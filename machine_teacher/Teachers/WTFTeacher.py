from .. import GenericTeacher
from ..Utils.Sampler import get_first_examples
import numpy as np

# numpy convetions
_ROW_AXIS = 0
_FRAC_START = 0.01
_FRAC_STOP = 0.5
_SEED = 0

class WTFTeacher(GenericTeacher.Teacher):
	def __init__(self, frac_start: float = _FRAC_START,
		frac_stop: float = _FRAC_STOP,
		seed: int = _SEED):
		self.frac_start = frac_start
		self.frac_stop = frac_stop
		self.seed = seed
		
		assert 0.0 <= frac_start <= 1.0, "frac start most be in [0, 1]"
		assert frac_start <= frac_stop <= 1.0, "frac start most be in [frac_start, 1]"

	def start(self, X, y) -> None:
		super()._start(X, y)

		m = X.shape[_ROW_AXIS] # number of rows
		self.m = m
		self.S_max_size = int(m * self.frac_stop)
		self.first_batch_size = int(m * self.frac_start)
		self.num_iters = 0
		self.selected = np.full(m, False)
		self._random = np.random.RandomState(self.seed)
		self.w = np.full(m, 1/(2.0*m))	
		self.samples = []
		self.n = 1
		self.classes = np.unique(self.y)
		self.S_current_size = 0

	def keep_going(self, h) -> bool:
		if len(self._get_delta_h(h)) == 0:
			return False
		elif self.S_current_size >= self.S_max_size:
			return False
		else:
			return True

	def get_first_examples(self):
		new_ids = get_first_examples(self.frac_start, self.m,
			self.classes, self.y, self._random.shuffle)
		new_ids = np.array(new_ids)
		self.selected[new_ids] = True
		return new_ids

	def get_new_examples(self, h):
		self.num_iters += 1
		wrong_labels = self._get_wrong_labels_id(h)
		
		new_ids = []
		while new_ids == []:
			new_w = self._get_new_weights(wrong_labels)
			delta_w = (new_w - self.w)/2
			new_ids = self._select_examples(wrong_labels,
				delta_w[wrong_labels]) #cabe melhoria

		new_ids = np.array(new_ids)

		# updates
		self.S_current_size += len(new_ids)
		self.selected[new_ids] = True

		return new_ids	

	def _get_delta_h(self, h):
		delta_h = self._get_wrong_labels_id(h)
		delta_h = [example_id for example_id in delta_h if not self.selected[example_id]] #analisar se cabe melhoria com setdiff1d
		return delta_h

	def _restart_wights(self):
		self.n *= 2
		self.w.fill(1/(2*self.m))

	def _get_new_weights(self, wrong_labels):
		new_w = np.copy(self.w)
		v = np.sum(new_w[wrong_labels])

		if v >= 1.0: #The algorithm failed
			self.n *= 2
			new_w.fill(1/(2*self.m))
			v = (1/(2*self.m)) * wrong_labels.size
		
		k = 1
		while v*k < 1.0:
			k = k*2

		new_w[wrong_labels] *= k
		return new_w

	def _select_examples(self, wrong_labels, delta_w):
		random_numbers = self._random.rand(self.n)
		random_numbers = np.sort(random_numbers)
		j = 0
		i = 0
		aux = 0
		flag = True
		N = wrong_labels.size
		S = []
		while (j < self.n) and (i < N):
			if flag:
				aux += delta_w[i]
			flag = False
			if random_numbers[j] <= aux:				
				if random_numbers[j] > (aux-delta_w[i]):
					S.append(wrong_labels[i])
					flag = True
					i+=1
				j+=1      
			else:
				i+=1
				flag = True
		
		return S