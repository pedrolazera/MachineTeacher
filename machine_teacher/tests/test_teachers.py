"""
This modules runs the protocol on a set of parameters
(Teachers, Learners, Datasets) previously tested in an
old project.

On top of that, this modules test the protocol method
behaviour when the teacher and the learner run out
of time

Author: Pedro Lazéra Cardoso
"""

import unittest
import numpy as np
from time import sleep

from .context import machine_teacher

from .datasets import car
from .datasets import crowdsourced

class PacTeacherTest(unittest.TestCase):
	""" Test cases for the PacTeacher """
	_LEARNER_SEED = 0
	_TEACHER_SEED = 4
	_K = 0.005
	_FRAC_START = 0.01
	_FRAC_STOP = 1.0
	_N_ESTIMATORS = 10

	def test_initial_conditions(self):
		""" Checks 'PachTeacher' initial conditions """

		T = machine_teacher.Teachers.PacTeacher(seed = self._TEACHER_SEED,
												batch_relative_size = self._K,
												frac_start = self._FRAC_START,
												frac_stop = self._FRAC_STOP)

		X = car.X.copy()
		y = car.y.copy()
		time_left = 10000 # dumb time_left

		T.start(X, y, time_left)

		self.assertEqual(T.batch_size, car.batch_size)

	def test_initial_teaching_set(self):
		""" Checks 'PacTeacher' first set of training examples """
		T = machine_teacher.Teachers.PacTeacher(seed = self._TEACHER_SEED,
												batch_relative_size = self._K,
												frac_start = self._FRAC_START,
												frac_stop = self._FRAC_STOP)

		X = car.X.copy()
		y = car.y.copy()
		time_left = 10000 # dumb time_left

		T.start(X, y, time_left)
		S_first_ids = T.get_first_examples(time_left)

		self.assertTrue(all(S_first_ids == car.S_first_ids))

	def test_full_teaching_set(self):
		""" Checks 'PacTeacher' full set of training examples for
		a the car dataset and the RandomForest Learner """
		T = machine_teacher.Teachers.PacTeacher(seed = self._TEACHER_SEED,
												batch_relative_size = self._K,
												frac_start = self._FRAC_START,
												frac_stop = self._FRAC_STOP)

		X = car.X.copy()
		y = car.y.copy()

		L = machine_teacher.Learners.RandomForestLearner(n_estimators = self._N_ESTIMATORS,
														 random_state = self._LEARNER_SEED)

		res = machine_teacher.teach(T, L, X, y)
		S_ids = res.S_ids

		self.assertTrue(all(S_ids == car.S_ids))

class WTFTeacherTest(unittest.TestCase):
	""" Test cases for the WTFTeacherTest """
	_SEED = 0
	_K = 0.09
	_FRAC_START = 0.01
	_FRAC_STOP = 0.2
	_N_ESTIMATORS = 10

	def test_initial_conditions(self):
		""" Checks 'WTFTeacherTest' initial conditions """
		T = machine_teacher.Teachers.WTFTeacher(seed = self._SEED,
												frac_start = self._FRAC_START,
												frac_stop = self._FRAC_STOP)
		X = crowdsourced.X.copy()
		y = crowdsourced.y.copy()
		time_left = 10000 # dumb time_left

		T.start(X, y, time_left)

		self.assertEqual(T.S_max_size, crowdsourced.S_max_size)

	def test_initial_teaching_set(self):
		""" Checks 'WTFTeacherTest' first set of training examples """
		T = machine_teacher.Teachers.WTFTeacher(seed = self._SEED,
												frac_start = self._FRAC_START,
												frac_stop = self._FRAC_STOP)

		X = crowdsourced.X.copy()
		y = crowdsourced.y.copy()
		time_left = 10000 # dumb time_left

		T.start(X, y, time_left)
		S_first_ids = T.get_first_examples(time_left)

		self.assertTrue(all(S_first_ids == crowdsourced.S_first_ids))

	def test_full_teaching_set(self):
		""" Checks 'WTFTeacher' full set of training examples for
		a the crowdsourced dataset and the RandomForest Learner """
		T = machine_teacher.Teachers.WTFTeacher(seed = self._SEED,
												frac_start = self._FRAC_START,
												frac_stop = self._FRAC_STOP)

		X = crowdsourced.X.copy()
		y = crowdsourced.y.copy()

		L = machine_teacher.Learners.RandomForestLearner(n_estimators = self._N_ESTIMATORS,
														 random_state = self._SEED)

		res = machine_teacher.teach(T, L, X, y)
		S_ids = res.S_ids

		self.assertTrue(all(S_ids == crowdsourced.S_ids))

class CustomTeacherTest(unittest.TestCase):
	""" Test cases for a custom teacher """
	
	def test_run_out_of_time(self):
		""" Checks the protocol behaviour when the teacher/learner
		intereactins run out of time """
		sleep_time = 0.7
		T = _CustomTeacher(sleep_time)
		L = _CustomLearner()

		dum_X = [[1],[2],[3],[4],[5],[6]]
		dum_y = [0,0,0,0,0,1]

		# 2 iteracoes
		time_limit = 2.0
		res = machine_teacher.teach(T, L, dum_X, dum_y,
			time_limit = time_limit)

		self.assertEqual(res.main_infos.qtd_iters, 2)
		self.assertTrue(res.main_infos.total_time, time_limit)

		# 3 iteracoes
		time_limit = 2.5
		res = machine_teacher.teach(T, L, dum_X, dum_y,
			time_limit = time_limit)

		self.assertEqual(res.main_infos.qtd_iters, 3)
		self.assertTrue(res.main_infos.total_time, time_limit)

class _CustomTeacher(machine_teacher.GenericTeacher.Teacher):
	def __init__(self, sleep_time):
		self.sleep_time = sleep_time

	def start(self, X, y, time_left):
		super()._start(X, y, time_left)
		self.num_iters = 0

	def get_first_examples(self, time_left: float) -> np.ndarray:
		ids = self.ids[self.num_iters:self.num_iters+1]
		sleep(self.sleep_time)
		self.num_iters += 1
		return ids

	def get_new_examples(self, test_ids,
		test_labels, time_left: float):
		return self.get_first_examples(time_left)

	def get_new_test_ids(self, test_ids,
		test_labels, time_left: float) -> np.ndarray:
		return []

class _CustomLearner(machine_teacher.GenericLearner.Learner):
	def start(self):
		pass

	def fit(self, X, y):
		pass

	def predict(self, X):
		qtd_rows = X.shape[0]
		labels = [0] * qtd_rows
		return labels