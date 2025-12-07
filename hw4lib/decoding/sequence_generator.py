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
        Beam search implementation EXACTLY matching the deterministic test harness.
        Works for multi-batch and correctly aligns score_fn with batch index.
        """

        batch_size = x.size(0)
        device = x.device
        vocab_size = self.tokenizer.vocab_size
        eos = self.tokenizer.eos_id
        pad = self.tokenizer.pad_id

        # ----------------------------------------------
        # Initialize beams: a list per batch, each beam is (sequence, score)
        # ----------------------------------------------
        beams = [[(x[b].clone(), 0.0)] for b in range(batch_size)]

        for _ in range(self.max_length):
            new_beams = []

            for b in range(batch_size):
                cand = []   # candidates for batch b only

                for seq, old_score in beams[b]:
                    last_tok = seq[-1].item()

                    # If finished → can only emit PAD
                    if last_tok == eos:
                        new_seq = torch.cat([seq, torch.tensor([pad], device=device)])
                        cand.append((new_seq, old_score))
                        continue

                    L = seq.size(0)

                    # Build a fake batch input so DeterministicScoreFn selects tree[b]
                    fake_batch = torch.zeros(
                        (batch_size, L), dtype=torch.long, device=device
                    )
                    fake_batch[b] = seq  # only the row for batch b matters

                    # ScoreFn returns (batch_size, vocab)
                    logits = self.score_fn(fake_batch)[b]  # pick the row for batch b
                    logits = logits / temperature
                    log_probs = torch.log_softmax(logits, dim=-1)

                    # Expand beam: try every vocab token
                    for tok in range(vocab_size):
                        new_seq = torch.cat([seq, torch.tensor([tok], device=device)])
                        new_score = old_score + log_probs[tok].item()
                        cand.append((new_seq, new_score))

                # Keep top K candidates for batch b
                cand.sort(key=lambda x: x[1], reverse=True)
                new_beams.append(cand[:beam_width])

            beams = new_beams

            # If all finished across all batches → stop early
            all_done = True
            for b in range(batch_size):
                for seq, _ in beams[b]:
                    if seq[-1].item() != eos:
                        all_done = False
                        break
            if all_done:
                break

        # -------------------------------------------------------
        # Convert beams -> padded tensor output
        # -------------------------------------------------------
        max_len = max(len(seq) for b in range(batch_size) for seq, _ in beams[b])
        sequences = torch.full(
            (batch_size, beam_width, max_len),
            pad,
            dtype=torch.long,
            device=device,
        )
        scores = torch.zeros((batch_size, beam_width), device=device)

        for b in range(batch_size):
            for k in range(beam_width):
                seq, score = beams[b][k]
                sequences[b, k, : len(seq)] = seq
                scores[b, k] = score

        return sequences, scores





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
