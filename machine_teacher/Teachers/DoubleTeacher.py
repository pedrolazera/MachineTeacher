import numpy as np
from ..GenericTeacher import Teacher
from ..Utils.Sampler import get_first_examples
from sklearn import preprocessing
import warnings

_SEED = 0
_FRAC_START = 0.01
_STRATEGY_DOUBLE_INCREMENT = 0
_STRATEGY_DOUBLE_SIZE = 1

class DoubleTeacher(Teacher):
	name = "DoubleTeacher"

	def __init__(self, seed: int = _SEED,
		frac_start: float = _FRAC_START,
		scale: bool = True,
		strategy: int = _STRATEGY_DOUBLE_INCREMENT,
		shuffle: bool = True):
		self.seed = seed
		self.frac_start = frac_start
		self.scale = scale
		self.strategy = strategy

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
		classes = np.unique(self.y) # isso devia sair. devia ser computado for get_first_examples
		f_shuffle = np.random.RandomState(self.seed).shuffle
		new_ids = get_first_examples(self.frac_start, self.m,
			classes, self.y, f_shuffle)
		new_ids = np.array(new_ids)
		
		# update shuffled_ids. Aqui usamos ponteiros de ponteiros.
		# debugue as próximas duas linhas de cabeça vazia
		_new_ids = set(new_ids)
		self.shuffled_ids = np.append(new_ids,
							          [i for i in self.shuffled_ids if i not in _new_ids])
		self.unshuffled_ids = self._get_reverse_map(self.shuffled_ids)

		# update batch size, from 1 to len(new_ids), based on strategy
		if self.strategy == _STRATEGY_DOUBLE_SIZE:
			self.batch_size = len(new_ids)
		elif self.strategy == _STRATEGY_DOUBLE_INCREMENT:
			self.batch_size = 1
		else:
			raise ValueError("Estrategia desconhecida: " + str(self.strategy))

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

	#def _reordena_ids(self, ids, ids_ids_left):
	#	_set_ids_left = set(ids_ids_left)
	#	ids_left = ids[ids_ids_left]
	#	ids_ids_right = np.array([i for i in range(len(ids)) if i not in _set_ids_left])
	#	ids_right = ids[ids_ids_right]
	#	ids_reordenado = np.append(ids_left, ids_right)
	#	return ids_reordenado

	def _get_reverse_map(self, v):
		v2 = np.zeros(len(v), dtype = v.dtype)
		for (i, vi) in enumerate(v):
			v2[vi] = i
		return v2