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
        if not torch.is_tensor(x):
            raise TypeError("x must be tensor")
        if x.dim() != 2:
            raise ValueError("x must be 2D")
        if beam_width < 1:
            raise ValueError("beam_width >= 1 required")
        if self.max_length < x.size(1):
            raise ValueError("max_length too small")

        batch = x.size(0)
        device = x.device

        # beams[b] = [(seq_tensor, score_float)]
        beams = [
            [(x[b:b+1], 0.0)]
            for b in range(batch)
        ]

        finished = [
            [False] * beam_width
            for _ in range(batch)
        ]

        # ---- expand step by step ----
        for _ in range(self.max_length - x.size(1)):
            all_done = True
            new_beams = [[] for _ in range(batch)]

            for b in range(batch):
                candidates = []

                for i, (seq, score) in enumerate(beams[b]):
                    # Keep finished beams unchanged
                    if finished[b][i]:
                        candidates.append((seq, score, i))
                        continue

                    # At least one not finished
                    all_done = False

                    # Predict next
                    logits = self._score_fn_with_batch(seq, b)
                    logits = self._apply_repeat_penalty(logits, seq, repeat_penalty)
                    logits = logits / temperature
                    log_probs = torch.log_softmax(logits, dim=-1)

                    top_lp, top_ids = torch.topk(log_probs, beam_width, dim=-1)

                    for k in range(beam_width):
                        next_id = top_ids[0, k]
                        new_score = score + top_lp[0, k].item()
                        new_seq = torch.cat([seq, next_id.view(1, 1)], dim=1)
                        candidates.append((new_seq, new_score, i))

                # choose top beams
                candidates.sort(key=lambda x: -x[1])
                chosen = candidates[:beam_width]
                new_beams[b] = [(s, sc) for (s, sc, _) in chosen]

                # update finished flags
                finished[b] = [
                    seq[0, -1].item() == self.tokenizer.eos_id
                    for (seq, _) in new_beams[b]
                ]

            beams = new_beams
            if all_done:
                break

        # ------------------------------------------------------------------
        # COLLATE RESULT
        # ------------------------------------------------------------------
        max_len = max(seq.size(1) for b in beams for (seq, _) in b)

        out_seqs = torch.full(
            (batch, beam_width, max_len),
            self.tokenizer.pad_id,
            dtype=torch.long,
            device=device
        )
        out_scores = torch.zeros(batch, beam_width, device=device)

        for b in range(batch):
            for i, (seq, sc) in enumerate(beams[b]):
                out_scores[b, i] = sc
                out_seqs[b, i, :seq.size(1)] = seq

        return out_seqs, out_scores

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
