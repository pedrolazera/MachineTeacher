import unittest
import numpy as np
from .context import project4
from sklearn.datasets import load_boston


class BasicTest1(unittest.TestCase):
	def test_nothing(self):
		return True

class LinRegTest(unittest.TestCase):
	def test_linear_regressor(self):
		X, y = load_boston(return_X_y=True)
		L1 = project4.Learners.LinearRegressionLearner()
		L1.fit(X,y)
		h = L1.predict(X)
		print("MSE =", _get_erro(y,h))
		#print(L1.model.coef_)
		return True

class RandomTeacherTest(unittest.TestCase):
	def test_random_teacher_1(self):
		X, y = load_boston(return_X_y=True)
		L1 = project4.Learners.LinearRegressionLearner()
		#T1 = project4.RandomTeacher(1,10)
		T1 = project4.Teachers.RandomTeacher(1,10)
		res = project4.teach(T1,L1,X,y)
		print("\n\n************************************\n")
		print("MSE =", _get_erro(y,res.h))
		print("----------")
		print(res)
		print("----------")
		print("len(X) =", X.shape[0])
		return True

	def test_random_teacher_2(self):
		X, y = load_boston(return_X_y=True)
		L1 = project4.Learners.LinearRegressionLearner()
		#T1 = project4.RandomTeacher(1,506)
		T1 = project4.Teachers.RandomTeacher(1,506)
		res = project4.teach(T1,L1,X,y)
		print("\n\n************************************\n")
		print("MSE =", _get_erro(y,res.h))
		print("----------")
		print(res)
		print("----------")
		print("len(X) =", X.shape[0])
		return True


def _get_erro(y1, y2):
	assert y1.size == y2.size, "y1 and y2 should have the same size"
	return np.sum(np.square(y1 - y2)) / y1.size
		

if __name__ == "__main__":
	unittest.main()