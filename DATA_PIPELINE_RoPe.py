from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Literal
import math
import tiktoken
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
from kernal import act_quant, weight_dequant, fp8_gemm
tokenizer = tiktoken.get_encoding("o200k_base")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class RotaryPositionalEncoding(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, "Embedding dimension must be even for RoPE."
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        sinusoid_inp = torch.einsum("i,j->ij", positions, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb

def apply_rope(x, rope):
    x1, x2 = x[..., ::2], x[..., 1::2]
    rope_cos, rope_sin = rope[..., :rope.shape[-1]//2], rope[..., rope.shape[-1]//2:]
    x_rotated = torch.cat((x1 * rope_cos - x2 * rope_sin,
                           x1 * rope_sin + x2 * rope_cos), dim=-1)
    return x_rotated

def tokenize(text):
    token_ids = tokenizer.encode(text)
    tokens = [tokenizer.decode([id]) for id in token_ids]
    return tokens

def embed_tokens(tokens):
    embeddings = embedding_model.encode(tokens, normalize_embeddings=True)
    return embeddings

def prepare_embeddings_with_rope(text, device='cpu'):
    tokens = tokenize(text)
    embeddings = embed_tokens(tokens)

    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float16, device=device)
    seq_len, dim = embeddings_tensor.shape

    #rope_module = RotaryPositionalEncoding(dim=dim).to(device)
    #rope = rope_module(seq_len, device=device)

    #embeddings_with_rope = apply_rope(embeddings_tensor, rope)

    return {
        "tokens": tokens,
        "original_embeddings": embeddings_tensor,
        #"rope_embeddings": embeddings_with_rope
    }

def prepare_llm_input(rope_embeddings):
    return rope_embeddings.unsqueeze(0)

if __name__ == "__main__":
    device = 'cpu'

    passinput = input(f"Enter your text: ")

    results = prepare_embeddings_with_rope(passinput, device=device)

    llm_input = prepare_llm_input(results['rope_embeddings'])

    print("Tokens:", results['tokens'])
    print("Original Embeddings shape:", results['original_embeddings'].shape)
    print("RoPE Embeddings shape:", results['rope_embeddings'].shape)
    print("\nExample token and embeddings (first token):")
    print("Token:", results['tokens'][0])
    print("Original embedding (first 20 dims):", results['original_embeddings'][0][:20])
    print("RoPE embedding (first 20 dims):", results['rope_embeddings'][0][:20])
    print("\nLLM input tensor shape (batch_size, seq_len, dim):", llm_input.shape)
    print(llm_input.shape)