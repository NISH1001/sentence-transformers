#!/usr/bin/env python3

from typing import List, Optional, Tuple

import torch


class NetworkHead(torch.nn.Module):
    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()


class ClassificationHead(NetworkHead):
    """
    Classifies whole sentence into label/s.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        labels: Optional[Tuple[str]] = None,
        multi_label: bool = False,
    ):
        super(ClassificationHead, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.multi_label = multi_label
        self.labels = labels or tuple()
        self.fc = torch.nn.Linear(input_dim, num_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.fc(embeddings)

    def _logits_to_labels(
        self, logits: torch.Tensor, threshold: float = 0.5
    ) -> List[List[str]]:
        res = []
        if self.multi_label:
            probabilities = torch.sigmoid(logits)
            predicted_indices = probabilities >= threshold
            for pred in predicted_indices:
                active_labels = [
                    self.labels[idx] for idx, active in enumerate(pred) if active.item()
                ]
                res.append(active_labels)
        else:
            probabilities = torch.softmax(logits, dim=-1)
            predicted_indices = torch.argmax(probabilities, dim=-1)
            res = [self.labels[idx] for idx in predicted_indices]
        return res


class NERHead(ClassificationHead):
    """
    This classifies each token into a NER tag/label.
    """

    def __init__(
        self,
        input_dim: int,
        num_tags: int,
        ner_tags: Optional[Tuple[str]] = None,
        multi_label: bool = True,
    ):
        super(NERHead, self).__init__(
            input_dim=input_dim,
            num_classes=num_tags,
            labels=ner_tags,
            multi_label=multi_label,
        )

    def _logits_to_labels(
        self, logits: torch.Tensor, threshold: float = 0.5
    ) -> List[List[List[str]]]:
        res = []
        # if single token could potentially have multiple entities (probably not realistic)
        if self.multi_label:
            probabilities = torch.sigmoid(logits)
            predicted_indices = probabilities >= threshold

            for batch_idx in predicted_indices:
                batch_labels = []
                for token_idx in batch_idx:
                    # active labels for the current token
                    active_labels = [
                        self.labels[label_idx]
                        for label_idx in range(token_idx.size(0))
                        if token_idx[label_idx].item()
                    ]
                    batch_labels.append(active_labels)
                res.append(batch_labels)
        else:
            probabilities = torch.softmax(logits, dim=-1)
            predicted_indices = torch.argmax(probabilities, dim=-1)

            for batch_idx in predicted_indices:
                batch_labels = []
                # just map each token to individual label
                for token_idx in batch_idx:
                    batch_labels.append(self.labels[token_idx.item()])
                res.append(batch_labels)
        return res


def main():
    pass


if __name__ == "__main__":
    main()
