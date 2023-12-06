"""
[Transformer 이해하기]
- ref : https://wikidocs.net/156986
"""
import torch
print(torch.__version__)

""" [torch.nn 모듈로 Transformer 만드는 간단한 예제] """
# S : source sequence length
# T : target sequence length
# N : batch size
# E : feature number
# example : output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)

# batch_first=False
import torch
import torch.nn as nn
import numpy as np

transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
src = torch.rand((10, 32, 512))         # if batch_first=False -> (S, N, E) / if batch_first=True -> (N, S, E)
tgt = torch.rand((20, 32, 512))         # if batch_first=False -> (T, N, E) / if batch_first=True -> (N, T, E)
out = transformer_model(src, tgt)       # if batch_first=False -> (T, N, E) / if batch_first=True -> (N, T, E)

# batch_first=True -> torch 1.9.0 이상부터 지원되는 parameter
transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12, batch_first=True)
src = torch.rand((32, 10, 512))         # if batch_first=False -> (S, N, E) / if batch_first=True -> (N, S, E)
tgt = torch.rand((32, 20, 512))         # if batch_first=False -> (T, N, E) / if batch_first=True -> (N, T, E)
out = transformer_model(src, tgt)       # if batch_first=False -> (T, N, E) / if batch_first=True -> (N, T, E)

