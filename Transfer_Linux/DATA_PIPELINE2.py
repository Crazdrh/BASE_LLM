import sys
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import torch.nn as nn
import torch.distributed as dist
import math
import tiktoken
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

tokenizer = tiktoken.get_encoding("cl100k_base")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def tokenize(text: str):
    token_ids = tokenizer.encode(text)
    tokens = [tokenizer.decode([id]) for id in token_ids]
    return tokens

def embed_tokens(tokens: list[str]):
    embeddings = embedding_model.encode(tokens, normalize_embeddings=True)
    return embeddings

def prepare_embeddings(text: str, device='cpu'):
    tokens = tokenize(text)
    embeddings = embed_tokens(tokens)
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32, device=device)

    return {
        "tokens": tokens,
        "embeddings": embeddings_tensor
    }

def prepare_llm_input(embeddings_tensor: torch.Tensor):
    return embeddings_tensor.unsqueeze(0)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    passinput = input("Enter your text: ")
    results = prepare_embeddings(passinput, device=device)
    llm_input = prepare_llm_input(results['embeddings'])

    print("Tokens:", results['tokens'])
    print("Embeddings shape:", results['embeddings'].shape)
    print("\nExample token and embedding (first token):")
    print("Token:", results['tokens'][0])
    print("Embedding (first 5 dims):", results['embeddings'][0][:5])
    print("\nLLM input tensor shape (batch_size, seq_len, dim):", llm_input.shape)
