from .. import GenericTeacher
import numpy as np

class RandomTeacher(GenericTeacher.Teacher):
	_MAX_ITERS = 1000

	def __init__(self, batch_size=1,
		max_iters=_MAX_ITERS):
		self.max_iters = max_iters
		self.batch_size = batch_size

	def start(self, X, y):
		self.iters = 0
		super().start(X, y)

	def keep_going(self, h):
		return self.iters < self.max_iters

	def get_new_examples(self, h):
		self.iters += 1
		wrong_labels = self._get_wrong_labels_id(h)
		batch_size = min(self.batch_size, wrong_labels.size)
		i = np.random.choice(wrong_labels, batch_size, replace=False)
		return i
		