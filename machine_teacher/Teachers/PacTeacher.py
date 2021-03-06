from ..GenericTeacher import Teacher
from ..Utils.Sampler import get_first_examples
import numpy as np
import warnings

_SEED = 0
_FRAC_START = 0.01
_FRAC_STOP = 0.2
_BATCH_RELATIVE_SIZE = 0.005
_FIRST_EXAMPLE_SEED = 0

class PacTeacher(Teacher):
	name = "PacTeacher"
	
	def __init__(self, seed: int = _SEED,
		batch_relative_size: float = _BATCH_RELATIVE_SIZE,
		frac_start: float = _FRAC_START,
		frac_stop: float = _FRAC_STOP,
		first_example_seed: int = _FIRST_EXAMPLE_SEED):
		self.seed = seed
		self.batch_relative_size = batch_relative_size
		self.frac_start = frac_start
		self.frac_stop = frac_stop
		self.first_examples_seed = first_example_seed

		assert 0.0 <= frac_start <= 1.0, "frac start most be in [0, 1]"
		assert frac_start <= frac_stop <= 1.0, "frac start most be in [frac_start, 1]"

	def start(self, X, y, time_left: float):
		super()._start(X, y, time_left)
		self._random = np.random.RandomState(self.seed)
		self.m = y.size

		self.S_max_size = self._get_S_max_size()
		self.batch_size = self._get_batch_size()
		self.shuffled_ids = self._get_shuffled_ids()
		self.free_spot = 0
		self.S_current_size = 0
		self.classes = np.unique(y)
		self.num_iters = 0
		self.selected = np.full(self.m, False)

	def _keep_going(self, test_labels):
		assert len(test_labels) == self.m

		if self.S_current_size >= self.S_max_size:
			return False
		elif len(self.get_wrong_and_unselected_labels_id(test_labels)) == 0: # delta_h = {}
			return False
		else:
			return True

	def get_first_examples(self, time_left: float):
		f_shuffle = np.random.RandomState(self.first_examples_seed).shuffle
		new_ids = get_first_examples(self.frac_start, self.m,
			self.classes, self.y, f_shuffle)
		new_ids = np.array(new_ids)
		
		# update shuffled_ids
		_new_ids = set(new_ids)
		self.shuffled_ids = [i for i in self.shuffled_ids if i not in _new_ids]

		return self._send_new_ids(new_ids)

	def get_new_examples(self, test_ids, test_labels, time_left: float):
		if not self._keep_going(test_labels):
			return np.array([])

		# build slice
		_start = self.free_spot
		_batch_size = min(self.batch_size, self.S_max_size - self.S_current_size)
		_end = min(self.free_spot + _batch_size, self.m)
		new_ids = self.shuffled_ids[_start:_end]
		self.free_spot += len(new_ids)
		return self._send_new_ids(new_ids)

	def get_wrong_and_unselected_labels_id(self, test_labels):
		wrong_labels_id = self._get_wrong_labels_id(test_labels)
		unselected = [i for i in wrong_labels_id if not self.selected[i]]
		unselected = np.array(unselected)
		return unselected

	def _send_new_ids(self, new_ids):
		# updates
		self.num_iters += 1
		self.S_current_size += len(new_ids)
		self.selected[new_ids] = True
		
		return new_ids

	def get_log_header(self):
		return ["iter_number", "training_set_size",
			"accuracy", "accuracy do conj. de treino"]

	def get_log_line(self, test_labels):
		accuracy = 1 - self._get_wrong_labels_id(test_labels).size/self.y.size
		log_line = [self.num_iters, self.S_current_size, accuracy, 0.0]
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

	def get_params(self):
		return {
			"seed": self.seed,
			"batch_relative_size": self.batch_relative_size, 
			"frac_start": self.frac_start,
			"frac_stop": self.frac_stop,
		}
