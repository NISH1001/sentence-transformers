#!/usr/bin/env python3

import torch


def pairwise_cosine_similarity(embeddings: torch.Tensor) -> torch.Tensor:
    normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return torch.matmul(normalized_embeddings, normalized_embeddings.T)


def main():
    pass


if __name__ == "__main__":
    main()
