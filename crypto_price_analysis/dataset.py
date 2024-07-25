import typing
import numpy as np

from torch.utils.data import Dataset


class ChunkedSequenceDataset(Dataset):

	def __init__(self, chunks: list[typing.Any], sequence_length: int):
		"""
		data_chunks must be preprocessed, normalized and of right type
		"""
		self.chunks = chunks
		self.chunks_length = list(map(len, chunks))
		self.sequence_length = sequence_length

		# chunk_sequence_length[i] = chunks_length[i] - (sequence_length + 1 - 1)
		# ass 1 to sequence_length for the y value
		self.chunks_sequence_length = [l - sequence_length for l in self.chunks_length]

		self.chunks_sequence_length_cummulative= np.cumsum(self.chunks_sequence_length)


	def __len__(self):
		return self.chunks_sequence_length_cummulative[-1]

	def __getitem__(self, index) -> tuple[typing.Any, typing.Any]:
		assert(index < len(self))
		backward_chunk_index = np.argmin(self.chunks_sequence_length_cummulative[::-1] > index)
		chunk_index = len(self.chunks) - (backward_chunk_index or len(self.chunks))
		chunk_offset = index - self.chunks_sequence_length_cummulative[chunk_index - 1] if chunk_index != 0 else index
		return self.chunks[chunk_index][chunk_offset:chunk_offset + self.sequence_length], self.chunks[chunk_index][chunk_offset + self.sequence_length]
	
