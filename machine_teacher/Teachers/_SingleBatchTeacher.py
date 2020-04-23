from .. import GenericTeacher
import numpy as np

class SingleBatchTeacher(GenericTeacher.Teacher):
	name = "SingleBatchTeacher"

	def start(self, X, y):
		return self._start(X, y)

	def get_first_examples(self):
		return self.ids

	def get_first_test_ids(self) -> np.ndarray:
		test_ids = np.array([0]) # dumb test
		return test_ids

	def get_new_examples(self, h):
		return np.array([])