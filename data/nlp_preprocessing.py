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
        unique_words_ids = {word:i+1 for i, word in enumerate(unique_words)}

        # 2. Encode each sentence by replacing words with their IDs
        encoded_positive = []
        for pos in positive:
            pos_words = pos.split(" ")
            pos_ids = [unique_words_ids[word] for word in pos_words]
            encoded_positive.append(torch.tensor(pos_ids))
        
        encoded_negative = []
        for neg in negative:
            neg_words = neg.split(" ")
            neg_ids = [unique_words_ids[word] for word in neg_words]
            encoded_positive.append(torch.tensor(neg_ids))

        # 3. Combine positive + negative into one list of tensors
        combined_tokens = encoded_positive + encoded_negative

        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        return nn.utils.rnn.pad_sequence(combined_tokens, batch_first=True)

