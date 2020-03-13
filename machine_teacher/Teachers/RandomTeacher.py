from .. import GenericTeacher
import numpy as np

class RandomTeacher(GenericTeacher.Teacher):
	_MAX_ITERS = 1000

	def __init__(self, seed: int, batch_relative_size: float,
		max_iters: int):
		self.max_iters = max_iters
		self.batch_relative_size = batch_relative_size
		self.seed = seed

		assert 0 < batch_relative_size <= 1.0

	def start(self, X, y):
		self._start(X, y)
		self.iters = 0
		self.m = y.size
		self.batch_size = self._get_batch_size(self.m,
			self.batch_relative_size)
		self._random = np.random.RandomState(self.seed)
		self.selected = np.full(self.m, False)

	def keep_going(self, h):
		if self.iters >= self.max_iters:
			return False
		elif np.all(self.selected):
			return False
		else:
			return True

	def get_new_examples(self, h):
		self.iters += 1
		wrong_labels = self._get_wrong_labels_id(h)

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

	def _get_wrong_labels_id(self, h):
		wrong_labels_id = super()._get_wrong_labels_id(h)

		# elimina ids que ja escolhidos anteriormente
		wrong_labels_id = [i for i in wrong_labels_id if not self.selected[i]]
		wrong_labels_id = np.array(wrong_labels_id)
		return wrong_labels_id

	@staticmethod
	def _get_batch_size(m, relative_size):
		batch_size = np.ceil(relative_size * m)
		batch_size = int(batch_size)
		batch_size = min(batch_size, m)
		return batch_size 
		