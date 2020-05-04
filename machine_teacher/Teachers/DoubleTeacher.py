import numpy as np
from ..GenericTeacher import Teacher
from ..Utils.Sampler import get_first_examples
from sklearn import preprocessing
import warnings

_SEED = 0
_FRAC_START = 0.01

class DoubleTeacher(Teacher):
	name = "DoubleTeacher"

	def __init__(self, seed: int = _SEED,
		frac_start: float = _FRAC_START,
		scale: bool = True):
		self.seed = seed
		self.frac_start = frac_start
		self.scale = scale

	def start(self, X, y, time_left: float):
		if self.scale:
			warnings.warn("X is being scaled inplace!")
			preprocessing.scale(X, copy = False)

		self._start(X, y, time_left)
		self.num_iters = 0
		self.m = y.size
		self.S_current_size = 0
		self.shuffled_ids = self._get_shuffled_ids()
		self.batch_size = 1
		
	def _keep_going(self):
		return self.S_current_size < self.m

	def get_first_examples(self, time_left: float):
		classes = np.unique(self.y)
		f_shuffle = np.random.RandomState(self.seed).shuffle
		new_ids = get_first_examples(self.frac_start, self.m,
			classes, self.y, f_shuffle)
		new_ids = np.array(new_ids)
		
		# update shuffled_ids
		_new_ids = set(new_ids)
		self.shuffled_ids = [i for i in self.shuffled_ids if i not in _new_ids]

		return self._send_new_ids(new_ids)

	def get_new_examples(self, test_ids, test_labels, time_left: float):
		if not self._keep_going():
			return np.array([])

		_start = self.S_current_size
		_end = min(_start + self.batch_size, self.m)
		new_ids = self.shuffled_ids[_start:_end]
		self.batch_size *= 2
		return self._send_new_ids(new_ids)

	def get_new_test_ids(self, test_ids,
		test_labels, time_left: float) -> np.ndarray:
		return np.array([])

	def get_log_header(self):
		return ["iter_number", "training_set_size", "accuracy"]

	def get_log_line(self, h):
		accuracy = 1 - self._get_wrong_labels_id(h).size/self.y.size
		log_line = [self.num_iters, self.S_current_size, accuracy]
		return log_line

	def _send_new_ids(self, new_ids):
		self.num_iters += 1
		self.S_current_size += len(new_ids)
		return new_ids

	def _get_shuffled_ids(self):
		ids = np.arange(self.m, dtype=int)
		f_shuffle = np.random.RandomState(self.seed).shuffle
		f_shuffle(ids)
		return ids
