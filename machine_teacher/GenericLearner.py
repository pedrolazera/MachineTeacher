import numpy as np

from .Definitions import get_qtd_rows
from .Definitions import InputSpace
from .Definitions import Labels
from .Definitions import join_input_spaces
from .Definitions import join_labels

class Learner:
	def __init__(self):
		self.X = None
		self.y = None

	def fit(self, X: InputSpace, y: Labels) -> None:
		raise NotImplementedError

	def predict(self, X: InputSpace) -> Labels:
		raise NotImplementedError

	def get_h0(self, X: InputSpace) -> Labels:
		raise NotImplementedError

	def _get_h0(self, X: InputSpace) -> Labels:
		m = get_qtd_rows(X)
		return np.full(m, 0)

	def update_X(self, Xi: InputSpace) -> None:
		if self.X is None:
			self.X = Xi
		else:
			self.X = join_input_spaces(self.X, Xi)

	def update_y(self, yi: Labels) -> None:
		if self.y is None:
			self.y = yi
		else:
			self.y = join_labels(self.y, yi)