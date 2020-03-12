import unittest
import numpy as np
from .context import machine_teacher
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris


class BasicTest1(unittest.TestCase):
	def test_nothing(self):
		return True

class LinRegTest(unittest.TestCase):
	def test_linear_regressor(self):
		X, y = load_boston(return_X_y=True)
		L = machine_teacher.Learners.LinearRegressionLearner()
		L.fit(X,y)
		h = L.predict(X)
		print("MSE =", _get_erro(y,h))
		#print(L1.model.coef_)
		return True

class RandomTeacherTest(unittest.TestCase):
	def test_random_teacher_1(self):
		X, y = load_boston(return_X_y=True)
		L = machine_teacher.Learners.LinearRegressionLearner()
		#T1 = machine_teacher.RandomTeacher(1,10)
		T = machine_teacher.Teachers.RandomTeacher(0, 1.0, 1)
		res = machine_teacher.teach(T,L,X,y)
		_print(X, y, res)
		return True

	def test_random_teacher_2(self):
		X, y = load_boston(return_X_y=True)
		L = machine_teacher.Learners.LinearRegressionLearner()
		#T = machine_teacher.RandomTeacher(1,506)
		T = machine_teacher.Teachers.RandomTeacher(0, 0.25, 4)
		res = machine_teacher.teach(T,L,X,y)
		_print(X, y, res)
		return True

	def test_random_teacher_3(self):
		X, y = load_iris(return_X_y=True)
		L1 = machine_teacher.Learners.RandomForestLearner(random_state=0)
		T1 = machine_teacher.Teachers.RandomTeacher(0, 0.01, 5)
		res = machine_teacher.teach(T1,L1,X,y)
		_print(X, y, res)
		return True

class WTFTeacherTest(unittest.TestCase):
	def _test_wtf_techer_1(self):
		X, y = load_iris(return_X_y=True)
		L = machine_teacher.Learners.RandomForestLearner(random_state=0)
		T = machine_teacher.Teachers.WTFTeacher(0.02, 0.5, 0)
		res = machine_teacher.teach(T,L,X,y)
		return True

def _get_erro(y1, y2):
	assert y1.size == y2.size, "y1 and y2 should have the same size"
	return np.sum(np.square(y1 - y2)) / y1.size

def _print(X, y, res):
	print("\n\n************************************\n")
	print("MSE =", _get_erro(y, res.h))
	print("----------")
	print(res)
	print("----------")
	print("len(X) =", X.shape[0])
		

if __name__ == "__main__":
	unittest.main()