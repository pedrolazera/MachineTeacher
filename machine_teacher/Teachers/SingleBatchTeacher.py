from ..GenericTeacher import Teacher
import numpy as np
import warnings
from sklearn import preprocessing

class SingleBatchTeacher(Teacher):
	name = "SingleBatchTeacher"

	def __init__(self):
		return

	def start(self, X, y, time_left: float):
		return self._start(X, y, time_left)

	def get_first_examples(self, time_left: float) -> np.ndarray:
		return self.ids

	def get_new_examples(self, test_ids,
		test_labels, time_left: float) -> np.ndarray:
		return np.array([])

	def get_new_test_ids(self, test_ids, test_labels,
		time_left: float) -> np.ndarray:
		return []