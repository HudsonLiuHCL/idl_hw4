import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Callable
from ..data import H4Tokenizer

class SequenceGenerator:
    """
    A class for generating sequences using various decoding strategies.
    """

    def __init__(
        self,
        score_fn: Callable,
        tokenizer: H4Tokenizer,
        max_length: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.score_fn = score_fn
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    # ----------------------------------------------------------------------
    # Helper: repetition penalty
    # ----------------------------------------------------------------------
    def _apply_repeat_penalty(self, logits, sequences, penalty=1.0):
        if penalty == 1.0:
            return logits

        if logits.dim() == 2:  # (batch, vocab)
            for i in range(sequences.size(0)):
                uniq = torch.unique(sequences[i])
                logits[i, uniq] = logits[i, uniq] / penalty
        else:  # (batch, beam, vocab)
            for b in range(sequences.size(0)):
                for k in range(sequences.size(1)):
                    uniq = torch.unique(sequences[b, k])
                    logits[b, k, uniq] = logits[b, k, uniq] / penalty

        return logits

    # ----------------------------------------------------------------------
    # Helper: sampling logits filtering
    # ----------------------------------------------------------------------
    def _filter_logits(self, logits, temperature=1.0, top_k=0, top_p=1.0):
        logits = logits / temperature

        if top_k > 0:
            top_k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            cutoff = top_k_logits[..., -1].unsqueeze(-1)
            logits[logits < cutoff] = float("-inf")

        if top_p < 1.0:
            log_probs = torch.log_softmax(logits, dim=-1)
            sorted_lp, sorted_idx = torch.sort(log_probs, descending=True)
            probs = sorted_lp.exp()
            cum = probs.cumsum(dim=-1)

            mask = cum > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False

            to_remove = mask.scatter(dim=-1, index=sorted_idx, src=mask)
            logits[to_remove] = float("-inf")

        return logits

    # ----------------------------------------------------------------------
    # GREEDY SEARCH
    # ----------------------------------------------------------------------
    def generate_greedy(self, x, temperature=1.0, repeat_penalty=1.0):
        if not torch.is_tensor(x):
            raise TypeError("x must be tensor")
        if x.dim() != 2:
            raise ValueError("x must be 2D (batch, seq)")
        if self.max_length < x.size(1):
            raise ValueError("max_length too small")

        batch = x.size(0)
        device = x.device

        scores = torch.zeros(batch, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)

        for _ in range(self.max_length - x.size(1)):

            if finished.all():
                break

            logits = self.score_fn(x)
            logits = self._apply_repeat_penalty(logits, x, repeat_penalty)
            logits = logits / temperature
            log_probs = torch.log_softmax(logits, dim=-1)

            next_toks = torch.argmax(log_probs, dim=-1)
            tok_scores = log_probs[torch.arange(batch), next_toks]

            scores = torch.where(finished, scores, scores + tok_scores)

            x = torch.cat([x, next_toks.unsqueeze(-1)], dim=1)

            eos_mask = next_toks == self.tokenizer.eos_id
            finished |= eos_mask

        return x, scores

    # ----------------------------------------------------------------------
    # Helper for BEAM: score_fn but aligned to batch index
    # ----------------------------------------------------------------------
    def _score_fn_with_batch(self, seq, batch_idx):
        """
        DeterministicScoreFn expects calls shaped like real batch input.
        This wraps seq (shape 1×T) into a fake batch where it sits at index batch_idx.
        """
        dummy = torch.zeros(
            (batch_idx + 1, seq.size(1)),
            dtype=torch.long,
            device=seq.device
        )
        dummy[batch_idx] = seq[0]
        out = self.score_fn(dummy)  # (batch_idx+1, vocab)
        return out[batch_idx: batch_idx + 1]

    # ----------------------------------------------------------------------
    # BEAM SEARCH (correct + HW-compatible)
    # ----------------------------------------------------------------------
    def generate_beam(self, x, beam_width, temperature=1.0, repeat_penalty=1.0):
        """
        Vectorized beam search implementation following the pseudocode.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            beam_width: Number of beams to maintain
            temperature: Scaling factor for logits
            repeat_penalty: Penalty factor for repeated tokens
            
        Returns:
            x: Generated sequences of shape (batch_size, beam_width, final_seq_len)
            scores: Cumulative scores of shape (batch_size, beam_width)
        """
        if not torch.is_tensor(x):
            raise TypeError("x must be tensor")
        if x.dim() != 2:
            raise ValueError("x must be 2D (batch, seq)")
        if beam_width < 1:
            raise ValueError("beam_width >= 1 required")
        if self.max_length < x.size(1):
            raise ValueError("max_length too small")

        batch_size = x.size(0)
        device = x.device
        vocab_size = self.tokenizer.vocab_size

        # Initialize scores: (batch_size, beam_width)
        scores = torch.zeros(batch_size, beam_width, device=device)
        
        # Initialize finished flags: (batch_size, beam_width)
        finished = torch.zeros(batch_size, beam_width, dtype=torch.bool, device=device)

        # Step 0: Compute initial logits and select top beam_width tokens
        logits = self.score_fn(x)  # (batch_size, vocab_size)
        logits = self._apply_repeat_penalty(logits, x, repeat_penalty)
        logits = logits / temperature
        log_probs = torch.log_softmax(logits, dim=-1)  # (batch_size, vocab_size)

        # Select top beam_width tokens
        top_scores, top_tokens = torch.topk(log_probs, beam_width, dim=-1)  # (batch_size, beam_width)
        scores = top_scores  # (batch_size, beam_width)

        # Expand x along beam dimension: (batch_size, seq_len) -> (batch_size, beam_width, seq_len)
        x = x.unsqueeze(1).expand(batch_size, beam_width, -1)  # (batch_size, beam_width, seq_len)
        
        # Append next tokens: (batch_size, beam_width, seq_len+1)
        x = torch.cat([x, top_tokens.unsqueeze(-1)], dim=-1)
        
        # Update finished flags
        finished = (top_tokens == self.tokenizer.eos_id)

        # Main loop
        for t in range(1, self.max_length - x.size(-1) + 1):
            if finished.all():
                break

            # Compute logits for each beam
            # Reshape to process all beams at once: (batch_size * beam_width, seq_len)
            x_flat = x.view(batch_size * beam_width, -1)
            logits = self.score_fn(x_flat)  # (batch_size * beam_width, vocab_size)
            
            # Reshape back: (batch_size, beam_width, vocab_size)
            logits = logits.view(batch_size, beam_width, vocab_size)
            
            # Apply repeat penalty
            logits = self._apply_repeat_penalty(logits, x, repeat_penalty)
            logits = logits / temperature
            log_probs = torch.log_softmax(logits, dim=-1)  # (batch_size, beam_width, vocab_size)

            # Mask finished beams by setting their log_probs to very negative except for PAD token
            mask = finished.unsqueeze(-1).expand(-1, -1, vocab_size)  # (batch_size, beam_width, vocab_size)
            log_probs = torch.where(mask, torch.full_like(log_probs, float('-inf')), log_probs)
            # Allow PAD token for finished beams
            log_probs[finished, self.tokenizer.pad_id] = 0.0

            # Compute cumulative scores: (batch_size, beam_width, vocab_size)
            # Expand scores for broadcasting
            cum_scores = scores.unsqueeze(-1) + log_probs  # (batch_size, beam_width, vocab_size)

            # Flatten for beam selection: (batch_size, beam_width * vocab_size)
            cum_scores_flat = cum_scores.view(batch_size, -1)

            # Select top beam_width candidates
            top_scores, top_indices = torch.topk(cum_scores_flat, beam_width, dim=-1)  # (batch_size, beam_width)
            
            # Update scores
            scores = top_scores

            # Re-map to get beam indices and token ids
            beam_indices = top_indices // vocab_size  # (batch_size, beam_width)
            next_tokens = top_indices % vocab_size    # (batch_size, beam_width)

            # Reorder x based on beam_indices
            # Use gather to select the correct beams
            batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, beam_width)
            x = x[batch_idx, beam_indices]  # (batch_size, beam_width, seq_len)
            
            # Reorder finished flags
            finished = finished[batch_idx, beam_indices]  # (batch_size, beam_width)

            # Append next tokens
            x = torch.cat([x, next_tokens.unsqueeze(-1)], dim=-1)  # (batch_size, beam_width, seq_len+1)

            # Update finished flags
            finished = finished | (next_tokens == self.tokenizer.eos_id)

        # Sort sequences in descending order of scores
        sorted_scores, sorted_indices = torch.sort(scores, dim=-1, descending=True)
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, beam_width)
        x = x[batch_idx, sorted_indices]
        scores = sorted_scores

        return x, scores

    # ----------------------------------------------------------------------
    # Sampling (already correct)
    # ----------------------------------------------------------------------
    def generate_sample(
        self, x, temperature=1.0, top_k=0, top_p=1.0
    ):
        if not torch.is_tensor(x):
            raise TypeError("x must be tensor")
        if x.dim() != 2:
            raise ValueError("x must be 2D")
        if temperature <= 0:
            raise ValueError("temperature > 0 needed")

        batch = x.size(0)
        scores = torch.zeros(batch, device=x.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=x.device)

        for _ in range(self.max_length - x.size(1)):
            if finished.all():
                break

            logits = self.score_fn(x)
            logits = self._filter_logits(logits, temperature, top_k, top_p)
            log_probs = torch.log_softmax(logits, dim=-1)

            probs = log_probs.exp()
            next_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
            next_scores = log_probs[torch.arange(batch), next_ids]

            scores = torch.where(finished, scores, scores + next_scores)
            x = torch.cat([x, next_ids.unsqueeze(1)], dim=1)

            eos = next_ids == self.tokenizer.eos_id
            finished |= eos

        return x, scores

    # ----------------------------------------------------------------------
    # EOS trimming
    # ----------------------------------------------------------------------
    @staticmethod
    def post_process_sequence(seq, tokenizer):
        if seq.dim() == 1:
            eos = (seq == tokenizer.eos_id).nonzero()
            if len(eos) > 0:
                return seq[: eos[0].item() + 1]
            return seq

        # batched list output
        eos_mask = seq == tokenizer.eos_id
        first_eos = (eos_mask.float().cumsum(dim=1).eq(1) & eos_mask)
        seq_mask = first_eos.cumsum(dim=1).eq(0) | first_eos
        return [s[: m.sum()] for s, m in zip(seq, seq_mask)]
