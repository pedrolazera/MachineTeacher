import unittest
import numpy as np
from time import sleep

from .context import machine_teacher
DoubleTeacher = machine_teacher.Teachers.DoubleTeacher

class TestesDoubleTeacher(unittest.TestCase):
	# testa os primeiros exemplos pegos
	# testa se testa todos os exemplos quando o tempo permite
	def test_first_examples(self):
		N = 100
		K = 2
		teacher_args = {
			"seed": 0,
			"frac_start": 0.1,
			"scale": True,
			"strategy": DoubleTeacher._STRATEGY_DOUBLE_INCREMENT
		}

		time_left = 100000.0 # dumb time left
		X, X_labels = _get_dataset_artificial(N, K)
		T = DoubleTeacher(**teacher_args)
		T.start(X, X_labels, time_left)
		
		first_ids = T.get_first_examples(time_left) # time_left = dumb
		assert (X_labels[first_ids] == 1).sum() == 5

	def test_get_all_examples(self):
		"""
		Tamanhos do TS: [10, 20, 40, 80, 100]
		"""
		N = 100
		K = 2
		teacher_args = {
			"seed": 0,
			"frac_start": 0.1,
			"scale": True,
			"strategy": DoubleTeacher._STRATEGY_DOUBLE_SIZE
		}
		train_ids = []

		_dumb_time_left = 100000.0 # dumb time left
		X, X_labels = _get_dataset_artificial(N, K)
		T = DoubleTeacher(**teacher_args)
		T.start(X, X_labels, _dumb_time_left)
		
		# testa distribuicao do batch inicial
		first_train_ids = T.get_first_examples(_dumb_time_left) # time_left = dumb
		self.assertEqual( (X_labels[first_train_ids] == 1).sum(), 5)
		train_ids.append(first_train_ids)

		# testa tamanho do train_id a cada iteração
		_dumb_test_ids = _get_empty_array()
		_dumb_test_labels = _get_empty_array()
		for i in range(4):
			new_train_ids = T.get_new_examples(_dumb_test_ids,
				_dumb_test_labels, _dumb_time_left)
			train_ids.append(new_train_ids)

		self.assertEqual(len(train_ids[0]), 10)
		self.assertEqual(len(train_ids[1]), 10)
		self.assertEqual(len(train_ids[2]), 20)
		self.assertEqual(len(train_ids[3]), 40)
		self.assertEqual(len(train_ids[4]), 20)

def _get_dataset_artificial(N, K):
	"""
	Retorna dataset artificial, com N exemplos distribuidos em K classes
	O vetor de classes X_labels é assim: X_labels[i] = i%K
	O conjunto de atributos X é apenas um vetor gaussiano -- normal (0,1)
	"""
	X = np.random.normal(0, 1, (N,1))
	X_labels = np.zeros(N, dtype=int)
	for i in range(N):
		X_labels[i] = i%K

	return (X, X_labels)

def _get_empty_array():
	return np.array([], dtype=int)