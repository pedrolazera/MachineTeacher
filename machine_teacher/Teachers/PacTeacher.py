from .. import GenericTeacher
import numpy as np

_FRAC_START = 0.01
_FRAC_STOP = 0.2

class PacTeacher(GenericTeacher.Teacher):
	def __init__(self, seed, batch_relative_size,
		frac_start=_FRAC_START,
		frac_stop=_FRAC_STOP):
		self.seed = seed
		self.batch_relative_size = batch_relative_size

	def start(self, X, y):
		self.X = X
		self.y = y
		self.m = y.size
		self.batch_size = self._get_batch_size(m)
		self.__random = np.random.RandomState(self.seed)

		self.free_ids = np.arange(m, dtype=int)
		self.__random.shuffle(self.free_ids)
		self.frac_stop = frac_stop
		self.frac_start = frac_start

	def keep_going(self, h):
		raise NotImplementedError

	def get_new_examples(self, h):
		if self.num_iters == 0:
			raise NotImplementedError
		else:
			return self.aux[i:i+qtd]

	def _get_first_batch(self):
		raise NotImplementedError

	def _get_batch_size(self, m):
		batch_size = np.ceil(self.batch_relative_size * m)
		batch_size = int(batch_size)
		return batch_size 