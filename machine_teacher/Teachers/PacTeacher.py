from .. import GenericTeacher
from ..Utils.Sampler import get_first_examples
import numpy as np
import warnings

_FRAC_START = 0.01
_FRAC_STOP = 0.2

class PacTeacher(GenericTeacher.Teacher):
	def __init__(self, seed, batch_relative_size,
		frac_start=_FRAC_START,
		frac_stop=_FRAC_STOP):
		self.seed = seed
		self.batch_relative_size = batch_relative_size
		self.frac_start = frac_start
		self.frac_stop = frac_stop

		assert 0.0 <= frac_start <= 1.0, "frac start most be in [0, 1]"
		assert frac_start <= frac_stop <= 1.0, "frac start most be in [frac_start, 1]"

	def start(self, X, y):
		super()._start(X, y)
		self._random = np.random.RandomState(self.seed)
		self.m = y.size

		self.S_max_size = self._get_S_max_size()
		self.batch_size = self._get_batch_size()
		self.shuffled_ids = self._get_shuffled_ids()
		self.free_spot = 0
		self.S_current_size = 0
		self.classes = np.unique(y)
		self.num_iters = 0
		
	def keep_going(self, h):
		if self.S_current_size >= self.S_max_size:
			return False
		elif len(self._get_wrong_labels_id(h)) == 0: # delta_h = {}
			return False
		else:
			return True

	def get_first_examples(self):
		warnings.warn("gambiarra temporaria para o shuffle", Warning)
		_gambiarra_tmp_shuffle = np.random.RandomState(self.seed).shuffle
		new_ids = get_first_examples(self.frac_start, self.m,
			self.classes, self.y, _gambiarra_tmp_shuffle)
		new_ids = np.array(new_ids)
		
		# update shuffled_ids
		_new_ids = set(new_ids)
		self.shuffled_ids = [i for i in self.shuffled_ids if i not in _new_ids]

		# update size of S
		self.S_current_size += len(new_ids)

		self.num_iters += 1

		return new_ids

	def get_new_examples(self, h):
		# build slice
		_start = self.free_spot
		_batch_size = min(self.batch_size, self.S_max_size - self.S_current_size)
		_end = min(self.free_spot + _batch_size, self.m)
		new_ids = self.shuffled_ids[_start:_end]

		self.free_spot += len(new_ids)
		self.S_current_size += len(new_ids)
		self.num_iters += 1

		print("tmp", self.num_iters, self.S_current_size)

		return new_ids

	def get_log_header(self):
		return ["iter_number", "training_set_size", "accuracy"]

	def get_log_line(self, h):
		accuracy = 1 - self._get_wrong_labels_id(h).size/self.y.size
		log_line = [self.num_iters, self.S_current_size, accuracy]
		return log_line

	def _get_shuffled_ids(self):
		ids = np.arange(self.m, dtype=int)
		self._random.shuffle(ids)
		return ids

	def _get_batch_size(self):
		batch_size = np.ceil(self.batch_relative_size * self.m)
		batch_size = int(batch_size)
		batch_size = min(batch_size, self.m)
		return batch_size

	def _get_S_max_size(self):
		S_max_size = int(self.m*self.frac_stop)
		return S_max_size





