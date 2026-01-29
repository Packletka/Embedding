from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


@dataclass
class EmbedderConfig:
    model_name: str = "google/embeddinggemma-300m"
    max_length: int = 512
    batch_size: int = 16
    normalize: bool = True
    device: Optional[str] = None


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class Embedder:
    def __init__(self, cfg: EmbedderConfig):
        self.cfg = cfg

        if cfg.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = cfg.device

        token = os.getenv("HF_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True, token=token)
        self.model = AutoModel.from_pretrained(cfg.model_name, token=token)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        all_vecs = []
        bs = self.cfg.batch_size

        for i in range(0, len(texts), bs):
            batch = texts[i: i + bs]

            tok = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                return_tensors="pt",
            )
            tok = {k: v.to(self.device) for k, v in tok.items()}

            out = self.model(**tok)
            last_hidden = out.last_hidden_state
            emb = _mean_pool(last_hidden, tok["attention_mask"])

            if self.cfg.normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)

            all_vecs.append(emb.detach().cpu().numpy().astype(np.float32))

        return np.vstack(all_vecs)
