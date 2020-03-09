import numpy as np

from .Definitions import get_qtd_rows
from .Definitions import InputSpace
from .Definitions import Labels

class Teacher:
	def start(self, X: InputSpace, y: Labels):
		self.X = X
		self.y = y
		self.ids = np.arange(y.size, dtype=int)

		qtd_rows_X = get_qtd_rows(X)
		qtd_rows_y = get_qtd_rows(y)
		assert qtd_rows_X == qtd_rows_X

	def keep_going(self, h: Labels) -> bool:
		raise NotImplementedError

	def get_new_examples(self, h: Labels):
		raise NotImplementedError

	def get_log_header(self):
		return []

	def get_log_line(self, h: Labels):
		return []

	def _get_wrong_labels_id(self, y2: Labels):
		wrong_labels = self.y != y2
		wrong_labels = wrong_labels.reshape(-1)
		return self.ids[wrong_labels]