import unittest
import numpy as np
from .context import machine_teacher

from .datasets import car
from .datasets import crowdsourced

class PacTeacherTest(unittest.TestCase):
	_LEARNER_SEED = 0
	_TEACHER_SEED = 4
	_K = 0.005
	_FRAC_START = 0.01
	_FRAC_STOP = 1.0
	_N_ESTIMATORS = 10

	def test_initial_conditions(self):
		T = machine_teacher.Teachers.PacTeacher(seed = self._TEACHER_SEED,
												batch_relative_size = self._K,
												frac_start = self._FRAC_START,
												frac_stop = self._FRAC_STOP)

		X = car.X.copy()
		y = car.y.copy()

		T.start(X, y)

		self.assertEqual(T.batch_size, car.batch_size)

	def test_initial_teaching_set(self):
		T = machine_teacher.Teachers.PacTeacher(seed = self._TEACHER_SEED,
												batch_relative_size = self._K,
												frac_start = self._FRAC_START,
												frac_stop = self._FRAC_STOP)

		X = car.X.copy()
		y = car.y.copy()

		T.start(X, y)
		S_first_ids = T.get_first_examples()

		self.assertTrue(all(S_first_ids == car.S_first_ids))

	def test_full_teaching_set(self):
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
	_SEED = 0
	_K = 0.09
	_FRAC_START = 0.01
	_FRAC_STOP = 0.2
	_N_ESTIMATORS = 10

	def test_initial_conditions(self):
		T = machine_teacher.Teachers.WTFTeacher(seed = self._SEED,
												frac_start = self._FRAC_START,
												frac_stop = self._FRAC_STOP)
		X = crowdsourced.X.copy()
		y = crowdsourced.y.copy()

		T.start(X, y)

		self.assertEqual(T.S_max_size, crowdsourced.S_max_size)

	def test_initial_teaching_set(self):
		T = machine_teacher.Teachers.WTFTeacher(seed = self._SEED,
												frac_start = self._FRAC_START,
												frac_stop = self._FRAC_STOP)

		X = crowdsourced.X.copy()
		y = crowdsourced.y.copy()

		T.start(X, y)
		S_first_ids = T.get_first_examples()

		self.assertTrue(all(S_first_ids == crowdsourced.S_first_ids))

	def test_full_teaching_set(self):
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