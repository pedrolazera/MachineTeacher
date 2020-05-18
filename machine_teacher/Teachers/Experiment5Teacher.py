import numpy as np
from . import Experiment3Teacher
from ..Utils.Sampler import get_first_examples
from ..Utils.Timer import Timer

class Experiment5Teacher(Experiment3Teacher):
	def _increase_training_set(self, training_set_size,
		increase_amount, time_left):
		assert len(self.iters_t_fim) >= 3

		x = [self.iters_n[-3], self.iters_n[-2], self.iters_n[-1]]
		y = [(self.iters_t_fim[-i] - self.iters_t_ini[-i]) for i in range(1,4)] 
		poly = np.poly1d(np.polyfit( x, y, deg=2)) 
		increased_training_set_size = training_set_size + increase_amount
		estimated_time = poly(increased_training_set_size)*(1+self.safity)
		
		return estimated_time < time_left