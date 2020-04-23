import numpy as np
from . import DoubleTeacher
from ..Utils.Sampler import get_first_examples
from ..Utils.Timer import Timer

_SEED = 0
_FRAC_START = 0.01
_FRAC_TIME_CHANGE = 1.0/4.0

class Experiment1Teacher(DoubleTeacher):
	name = "Experiment1Teacher"
	_STATE_DOUBLE_EXAMPLES = 0
	_STATE_CHANGE_STRATEGY = 1
	_STATE_GET_WRONG_EXAMPLES = 2
	_STATE_DONE = 3

	def __init__(self, time_limit: float,
		safity: float,
		seed: int = _SEED,
		frac_start: float = _FRAC_START,
		frac_time_change: float = _FRAC_TIME_CHANGE,
		scale = True):
		super().__init__(seed, frac_start, scale)
		self.safity = safity
		self.timer = Timer()
		self.time_limit = time_limit
		self.frac_time_change = frac_time_change

		assert safity > -1.0

	def start(self, X, y):
		super().start(X, y)
		self.timer.start()
		self._state = self._STATE_DOUBLE_EXAMPLES
		self.c = None
		self.k = None
		
	def _keep_going(self):
		bool1 = self.S_current_size < self.m
		bool2 = self._state != self._STATE_DONE
		return (bool1 and bool2)

	def get_first_examples(self):
		self.timer.tick("iter{}".format(self.num_iters+1))
		return super().get_first_examples()

	def get_new_examples(self, test_ids, test_labels):
		# marcadores para computar o tempo de treino
		if not self._keep_going():
			return np.array([])

		if self._state == self._STATE_DOUBLE_EXAMPLES:
			self.timer.tick("iter{}".format(self.num_iters+1))
			return super().get_new_examples(test_ids, test_labels)
		elif self._state == self._STATE_GET_WRONG_EXAMPLES:
			# get ids of missclassified examples
			assert len(test_ids) > 0, str((self.seed, self.num_iters))

			gabarito_test_labels = self.y[test_ids]
			new_ids = test_ids[test_labels != gabarito_test_labels]
			self._state = self._STATE_DONE
			
			return self._send_new_ids(new_ids)

	def get_new_test_ids(self, test_ids, test_labels) -> np.ndarray:
		if self._state == self._STATE_DONE:
			return np.array([])

		# check if it is time to change strategy
		if self._get_frac_elapsed_time() >= self.frac_time_change:
			if self.num_iters >= 3:
				if self._state == self._STATE_DOUBLE_EXAMPLES:
					self._state = self._STATE_CHANGE_STRATEGY

		if self._state == self._STATE_DOUBLE_EXAMPLES:
			self.timer.tock()
			return np.array([])

		if self._state == self._STATE_CHANGE_STRATEGY:
			assert len(test_ids) == 0
			self.timer.tock()
			self.qtd_wrong_untrained_tested_examples = 0
			self.qtd_untrained_tested_examples = 0
			self._state = self._STATE_GET_WRONG_EXAMPLES
		elif self._state == self._STATE_GET_WRONG_EXAMPLES:
			last_id = test_ids[-1]
			last_label = test_labels[-1]
			if self.y[last_id] != last_label:
				self.qtd_wrong_untrained_tested_examples += 1

		# check if is possible (there is time) to increase training set
		new_test_id = self.S_current_size + self.qtd_untrained_tested_examples
		training_set_size = self.S_current_size + self.qtd_wrong_untrained_tested_examples
		bool1 = new_test_id < len(self.y)
		bool2 = self._increase_training_set(training_set_size, 1)

		if bool1 and bool2:
			self.qtd_untrained_tested_examples += 1
			return np.array([new_test_id])
		else:
			if self.qtd_wrong_untrained_tested_examples == 0:
				self._state = self._STATE_DONE
			return np.array([])

	def _increase_training_set(self, training_set_size, increase_amount):
		if self.c is None:
			(self.c,self.k) = self._get_c_and_k()

		c,k = self.c, self.k
		increased_training_set_size = training_set_size + increase_amount
		estimated_time = c * (increased_training_set_size**k) * (1 + self.safity)
		time_left = self.time_limit - self.timer.get_elapsed_time()
		return estimated_time < time_left

	def _get_frac_elapsed_time(self):
		elapsed_time = self.timer.get_elapsed_time()
		frac_elapsed_time = elapsed_time / self.time_limit
		return frac_elapsed_time

	def _get_c_and_k(self):
		""" Find (c,k) such that train(n) = c*n^k
		k = log(t1/t0) / log(n1/n0)
		c = t1 / (size_last_iter^k)
		wherw...
		(i) t1 = time spent on last iter
		(ii) t0 = time spent on prev last iter
		(iii) n1 = qtd_examples of last iter
		(iv) n0 = qtd_examples of prev last iter
		"""
		assert self.num_iters == len(self.timer._d), str((self.num_iters, len(self.timer._d)))
		assert self.num_iters >= 3

		# get times of last (t1) and prev last iters (t0)
		vals = [self.timer["iter{}".format(i)] for i in range(1, self.num_iters+1)]
		t1 = vals[-1]
		t0 = vals[-2]

		# get sizes of last (n1) and prev last iters (n0)
		n1 = self.S_current_size
		n0 = n1 - 2**(self.num_iters-2)

		# get c and k
		k = np.log2(t1/t0) / np.log2(n1/n0)
		k = max(k, 0)
		c = t1 / (n1 ** k)

		return (c,k)

