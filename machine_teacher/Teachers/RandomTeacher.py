from ..GenericTeacher import Teacher
import numpy as np

class RandomTeacher(Teacher):
	name = "RandomTeacher"
	_MAX_ITERS = 1000

	def __init__(self, seed: int, batch_relative_size: float,
		max_iters: int):
		self.seed = seed
		self.max_iters = max_iters
		self.batch_relative_size = batch_relative_size

		assert 0 < batch_relative_size <= 1.0

	def start(self, X, y):
		self._start(X, y)
		self.iters = 0
		self.m = y.size
		self.batch_size = self._get_batch_size(self.m,
			self.batch_relative_size)
		self._random = np.random.RandomState(self.seed)
		self.selected = np.full(self.m, False)

	def _keep_going(self) -> bool:
		if self.iters >= self.max_iters:
			return False
		elif np.all(self.selected):
			return False
		else:
			return True

	def get_new_examples(self, test_ids, test_labels):
		assert len(test_labels) == self.m

		if not self._keep_going():
			return np.array([])

		self.iters += 1
		wrong_labels = self.get_wrong_and_unselected_labels_id(test_labels)

		assert wrong_labels.size > 0

		batch_size = min(self.batch_size, wrong_labels.size)
		new_ids = self._random.choice(wrong_labels, batch_size,
			replace=False)
		self._update_selected_ids(new_ids)
		return new_ids

	def get_first_examples(self):
		h_dumb = np.full(None, self.m)
		return self.get_new_examples(h_dumb)

	def _update_selected_ids(self, new_ids):
		self.selected[new_ids] = True

	def get_wrong_and_unselected_labels_id(self, test_labels):
		assert len(test_labels) == self.m
		
		wrong_labels_id = self._get_wrong_labels_id(test_labels)
		unselected = [i for i in wrong_labels_id if not self.selected[i]]
		unselected = np.array(unselected)
		return unselected

	def get_params(self):
		return {
			"seed": self.seed,
			"max_iters": self.max_iters,
			"batch_relative_size": self.batch_relative_size
		}

	@staticmethod
	def _get_batch_size(m, relative_size):
		batch_size = np.ceil(relative_size * m)
		batch_size = int(batch_size)
		batch_size = min(batch_size, m)
		return batch_size 
		