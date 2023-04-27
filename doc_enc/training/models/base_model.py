#!/usr/bin/env python3

import torch


class DualEncModelOutput:
    def __init__(
        self,
        sm: torch.Tensor,
        ivf_score_matrix: torch.Tensor | None = None,
        pq_score_matrix: torch.Tensor | None = None,
    ) -> None:
        self.dense_score_matrix: torch.Tensor = sm
        self.ivf_score_matrix: torch.Tensor | None = ivf_score_matrix
        self.pq_score_matrix: torch.Tensor | None = pq_score_matrix
