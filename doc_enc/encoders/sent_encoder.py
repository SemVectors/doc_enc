#!/usr/bin/env python3

import logging
from torch import nn


class SentEncoder(nn.Module):
    def __init__(self, embed, encoder):
        super().__init__()

        self.embed = embed
        self.encoder = encoder

    def forward(self, tokens, lengths, enforce_sorted=True, token_types=None):
        # embed tokens
        x = self.embed(tokens, token_types)
        return self.encoder.forward(x, lengths, enforce_sorted=enforce_sorted)
