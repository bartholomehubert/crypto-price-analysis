import time
from typing import Any
from numpy import empty
import torch
from torch.utils.data import DataLoader

class LogVars:

	def __init__(self):
		self.msg_len = 0

	def update(self, variables: dict[str, tuple[Any, str]] | dict[str, Any]):
		msg = ''
		for k, v in variables.items():
			if type(v) == tuple: 
				msg += f'{k}: {{:{v[1]}}} '.format(v[0])
			else:
				msg += f'{k}: {v} '

		padding = self.msg_len - len(msg) if len(msg) < self.msg_len else 0
		print('\r' + msg + ' ' * padding, end='')
		self.msg_len = len(msg)

	def new_line(self):
		self.msg_len = 0
		print()



def train_model(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: torch.optim.Optimizer, loss_function, epochs: int):

	loss_history = []
	logger = LogVars()

	for e in range(epochs):
		model.train()
		running_loss = 0.
		start_time = time.time()
		for b, (X_batch, y_batch) in enumerate(train_loader):
			optimizer.zero_grad()
			y_pred = model(X_batch)
			loss = loss_function(y_pred, y_batch)
			loss.backward()
			optimizer.step()

			loss_history.append(loss.item())
			running_loss += loss.item()
			logger.update({'epoch': e, 'batch_progress': (b / len(train_loader), '.2%'), 'loss': (running_loss / (b + 1), '.5f')})

		running_val_loss = 0

		with torch.no_grad():
			for X_batch, y_batch in val_loader:
				y_pred = model(X_batch)
				running_val_loss += loss_function(y_pred, y_batch).item()

		logger.update({'epoch': e, 'duration': (time.time() - start_time, '.2f'), 'loss': (running_loss / (len(train_loader)), '.5f'), 'val_loss': (running_val_loss / len(val_loader), '.5f')})
		logger.new_line()

	return loss_history