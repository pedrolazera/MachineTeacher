from ..GenericTeacher import Teacher
import numpy as np
import warnings
from sklearn import preprocessing

class SingleBatchTeacher(Teacher):
	name = "SingleBatchTeacher"

	def __init__(self, frac_dataset = 1.0):
		self.frac_dataset = frac_dataset
		return

	def start(self, X, y, time_left: float):
		return self._start(X, y, time_left)

	def get_first_examples(self, time_left: float) -> np.ndarray:
		size = np.round(self.frac_dataset*len(self.y))
		return self.ids[:int(size)]

	def get_new_examples(self, test_ids,
		test_labels, time_left: float) -> np.ndarray:
		return np.array([])

	def get_new_test_ids(self, test_ids, test_labels,
		time_left: float) -> np.ndarray:
		return []