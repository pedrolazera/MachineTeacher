import unittest
import numpy as np
from time import sleep

from .context import machine_teacher
DoubleTeacher = machine_teacher.Teachers.DoubleTeacher

class TestesDoubleTeacher(unittest.TestCase):
	# testa os primeiros exemplos pegos
	# testa se testa todos os exemplos quando o tempo permite
	# testa se o shuffle está se comportando bem
	def test_first_examples(self):
		N = 100
		K = 2
		teacher_args = {
			"seed": 0,
			"frac_start": 0.1,
			"scale": True,
			"strategy": DoubleTeacher._STRATEGY_DOUBLE_INCREMENT,
			"shuffle": False
		}

		T = DoubleTeacher(**teacher_args)
		X, X_labels = _get_dataset_artificial(N, K)

	def test_shuffle_examples(self):
		pass

	def test_get_all_examples(self):
		pass

def _get_dataset_artificial(N, K):
	"""
	Retorna dataset artificial, com N exemplos distribuidos em K classes
	O vetor de classes X_labels é assim: X_labels[i] = i%K
	O conjunto de atributos X é apenas um vetor de zeros
	"""
	X = np.zeros((N,1), dtype=float)
	X_labels = np.zeros(N, dtype=int)
	for i in range(N):
		X_labels[i] = i%K

	return (X, X_labels)