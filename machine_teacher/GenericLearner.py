import numpy as np

from .Definitions import get_qtd_rows
from .Definitions import InputSpace
from .Definitions import Labels
from .Definitions import join_input_spaces
from .Definitions import join_labels

class Learner:
	name = "GenericLearner"

	def start(self):
		pass

	def fit(self, X: InputSpace, y: Labels) -> None:
		raise NotImplementedError

	def predict(self, X: InputSpace) -> Labels:
		raise NotImplementedError

	def get_params(self) -> dict:
		return dict()