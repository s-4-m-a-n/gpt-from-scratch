import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        all_sentences = positive + negative
        all_words = []
        for sent in all_sentences:
            all_words += sent.split(" ")

        unique_words = sorted(set(all_words))
        word_to_idx = {word:i+1 for i, word in enumerate(unique_words)}

        # 2. Encode each sentence by replacing words with their IDs
        encoded_tokens = []
        for sent in all_sentences:
            words = sent.split(" ")
            idx = [word_to_idx[word] for word in words]
            encoded_tokens.append(torch.tensor(idx))
        
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        return nn.utils.rnn.pad_sequence(encoded_tokens, batch_first=True)

