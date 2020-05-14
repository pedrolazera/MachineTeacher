import numpy as np
from . import DoubleTeacher
from ..Utils.Sampler import get_first_examples
from ..Utils.Timer import Timer

class Experiment5Teacher(Experiment3Teacher):
	def _increase_training_set(self, training_set_size,
		increase_amount, time_left):
		assert len(self.iters_t_fim) >= 3
		# self.iters_t_fim[-1] # instante em que o ultimo treino terminou
		# self.iters_t_fim[-2] # instante em que o penultimo treino terminou
		# self.iters_n[-1] # tamanho do ultimo teaching set
		# self.iters_n[-2] # tamanho do penultimo teaching set