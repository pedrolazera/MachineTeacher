import numpy as np

from .Definitions import get_qtd_rows
from .Definitions import InputSpace
from .Definitions import Labels

class Teacher:
	name = "GenericTeacher"
	
	def start(self, X: InputSpace, y: Labels):
		raise NotImplementedError

	def keep_going(self, h: Labels) -> bool:
		raise NotImplementedError

	def get_first_examples(self) -> np.ndarray:
		raise NotImplementedError

	def get_new_examples(self, h: Labels) -> np.ndarray:
		raise NotImplementedError

	def get_log_header(self):
		return []

	def get_log_line(self, h: Labels):
		return []

	def get_params(self) -> dict:
		return dict()

	def _start(self, X: InputSpace, y: Labels):
		self.X = X
		self.y = y
		self.ids = np.arange(y.size, dtype=int)

		qtd_rows_X = get_qtd_rows(X)
		qtd_rows_y = get_qtd_rows(y)
		assert qtd_rows_X == qtd_rows_X

	def _get_wrong_labels_id(self, h: Labels):
		wrong_labels = self.y != h
		wrong_labels = wrong_labels.reshape(-1)
		return self.ids[wrong_labels]

	def _get_accuracy(self, h):
		wrong_labels = self._get_wrong_labels_id(h)
		accuracy = 1 - len(wrong_labels) / len(self.y)
		return accuracy
