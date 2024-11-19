from typing import Any

import numpy as np
import torch
import torch.utils.data

class ChunkedSequenceDataset(torch.utils.data.Dataset):

    def __init__(self, chunks: list[Any], sequence_length: int):
        """
        data_chunks must be preprocessed, normalized and of float32 type
        """
        self.chunks = chunks
        self.sequence_length = sequence_length

        # chunk_sequence_length[i] = chunks_length[i] - (sequence_length + 1 - 1)
        # ass 1 to sequence_length for the y value
        self.chunks_nsamples_cummulative= np.cumsum([len(chunk) - sequence_length for chunk in chunks])


    def __len__(self):
        return self.chunks_nsamples_cummulative[-1]

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        assert index < len(self)
        chunk_index = np.argmax(self.chunks_nsamples_cummulative > index)
        chunk_offset = index - self.chunks_nsamples_cummulative[chunk_index - 1] if chunk_index != 0 else index
        x = self.chunks[chunk_index][chunk_offset:chunk_offset + self.sequence_length]
        y = self.chunks[chunk_index][chunk_offset + self.sequence_length, [3]] # index 3 is the close column
        return x, y
