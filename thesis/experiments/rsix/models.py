"""Recommender models (CPU, deterministic, seeded).

* ItemPop      -- popularity ranking baseline
* BPR-MF       -- Rendle et al., UAI 2009, matrix factorization
* GRU4Rec-lite -- Hidasi et al., ICLR 2016, 1-layer GRU sequential model

All models expose ``rank(user, seq)`` returning scores over all items,
so metrics are model-agnostic.  Full ranking: every non-interacted item
is a candidate (no sampled negatives anywhere in evaluation).
"""

import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn


class ItemPop:
    def __init__(self, n_items: int):
        self.pop = np.zeros(n_items, dtype=np.float64)

    def fit(self, train: Dict[int, List[int]], epochs: int = 0, **kw):
        for items in train.values():
            for i in items:
                self.pop[i] += 1.0
        return {}

    def rank(self, user: int, seq: List[int]) -> np.ndarray:
        return self.pop.copy()


class BPRMF:
    """Bayesian personalized ranking matrix factorization."""

    def __init__(self, n_users: int, n_items: int, dim: int = 64,
                 lr: float = 0.01, l2: float = 1e-4, seed: int = 0):
        torch.manual_seed(seed)
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        self.lr, self.l2 = lr, l2
        self.params = list(self.user_emb.parameters()) + \
            list(self.item_emb.parameters())

    def fit(self, train: Dict[int, List[int]], epochs: int = 60,
            batch: int = 4096, rng: random.Random = None, **kw):
        rng = rng or random.Random(0)
        pairs = [(u, i) for u, items in train.items() for i in items]
        opt = torch.optim.Adam(self.params, lr=self.lr)
        n_items = self.item_emb.weight.shape[0]
        for _ in range(epochs):
            rng.shuffle(pairs)
            for start in range(0, len(pairs), batch):
                chunk = pairs[start:start + batch]
                u = torch.tensor([p[0] for p in chunk])
                pos = torch.tensor([p[1] for p in chunk])
                neg = torch.tensor([rng.randrange(n_items)
                                    for _ in chunk])
                uu, pp, nn_ = (self.user_emb(u), self.item_emb(pos),
                               self.item_emb(neg))
                pos_logits = (uu * pp).sum(-1)
                neg_logits = (uu * nn_).sum(-1)
                # BPR: maximize sigma(pos) and sigma(-neg)
                loss = -(pos_logits.sigmoid().log().mean()
                         + (-neg_logits).sigmoid().log().mean())
                reg = (uu ** 2).mean() + (pp ** 2).mean() + (nn_ ** 2).mean()
                (loss + self.l2 * reg).backward()
                opt.step()
                opt.zero_grad()
        return {}

    def rank(self, user: int, seq: List[int]) -> np.ndarray:
        with torch.no_grad():
            u = self.user_emb.weight[user]
            return (self.item_emb.weight @ u).numpy()


class GRU4Rec(nn.Module):
    """1-layer GRU sequential recommender (Hidasi et al. 2016)."""

    def __init__(self, n_items: int, dim: int = 64, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=n_items)
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.out = nn.Linear(dim, dim)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        with torch.no_grad():
            self.item_emb.weight[self.item_emb.padding_idx].zero_()

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        emb = self.item_emb(seqs)
        hidden, _ = self.gru(emb)
        return self.out(hidden)  # [B, L, D]

    def fit(self, train: Dict[int, List[int]], epochs: int = 20,
            batch: int = 128, lr: float = 1e-3, pad: int = None,
            rng: random.Random = None, items_pool=None, **kw):
        """Next-item training with full cross entropy over all items.

        Scores = hidden · item_emb^T (tied output embeddings), the
        GRU4Rec-style dot-product readout.
        """
        rng = rng or random.Random(0)
        if pad is None:
            pad = self.item_emb.padding_idx
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        users = sorted(u for u, items in train.items() if items)
        losses = []
        for _ in range(epochs):
            rng.shuffle(users)
            for start in range(0, len(users), batch):
                chunk = [train[u] for u in users[start:start + batch]]
                L = max(2, max(len(s) for s in chunk))
                seqs = torch.full((len(chunk), L), pad, dtype=torch.long)
                for bi, s in enumerate(chunk):
                    seqs[bi, :len(s)] = torch.tensor(s)
                logits = self(seqs)                       # [B, L, D]
                targets = seqs.clone()                    # next-item
                mask = seqs != pad
                scores = logits @ self.item_emb.weight.T  # [B, L, V+1]
                # exclude pad column from softmax by masking
                scores[..., pad] = -1e9
                # predict target[t+1] from position t
                loss = nn.functional.cross_entropy(
                    scores[:, :-1].reshape(-1, scores.shape[-1]),
                    targets[:, 1:].reshape(-1),
                    reduction="none").view(targets[:, 1:].shape)
                m = mask[:, 1:].float()
                loss = (loss * m).sum() / m.sum().clamp(min=1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                opt.step()
                opt.zero_grad()
                losses.append(float(loss.detach()))
        return {"final_loss": losses[-1] if losses else None}

    @torch.no_grad()
    def rank(self, user: int, seq: List[int]) -> np.ndarray:
        if not seq:
            seq = [self.item_emb.padding_idx]
        device = next(self.parameters()).device
        x = torch.tensor([seq], dtype=torch.long, device=device)
        hidden = self(x)[0, -1]                     # last state
        scores = self.item_emb.weight @ hidden
        return scores.cpu().numpy()
