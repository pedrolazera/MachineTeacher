from .. import GenericTeacher
import numpy as np

# numpy convetions
_ROW_AXIS = 0

class WTFTeacher(GenericTeacher.Teacher):
	def __init__(frac_start: float = 0.01, frac_stop: float = 0.5,
		seed: int = 0):
		self.frac_start = frac_start
		self.frac_stop = frac_stop
		self.seed = seed

	def start(self, X: InputSpace, y: Labels) -> None:
		self.X = X
		self.y = y

		m = X.shape[_ROW_AXIS] # number of rows
		self.m = m
		self.ids = np.arange(m)
		self.S_max_size = int(m * frac_stop)
		self.first_batch_size = int(m * self.frac_start)
		self.num_iters = 0
		self.selected = np.full(m, False)
		self._random = np.random.RandomState(self.seed)
		self.w = np.full(m, 1/(2.0*m))	
		self.samples = []
		self.n = 1

		# double checks
		qtd_rows_X = X.shape[_ROW_AXIS]
		qtd_rows_y = y.shape[_ROW_AXIS]
		assert qtd_rows_X == qtd_rows_y
		assert 0.0 <= frac_start <= 1.0, "frac start most be in [0, 1]"
		assert frac_start <= frac_stop <= 1.0, "frac start most be in [frac_start, 1]"

	def keep_going(self, h: Labels) -> bool:
		if len(self.get_delta_h(h)) == 0:
			return False
		elif len(self.S) >= self.S_max_size:
			return False
		else:
			return True

	def get_new_examples(self, h: Labels):
		self.num_iters += 1

		if self.num_iters == 1:
			n = self.S_first_batch_size
			selected_ids = self._random.choice(self.ids, n,	replace=False) # precisa ser estratificado
		else:
			wrong_labels = self._get_wrong_labels_id(h)
			S = []
			while S is not None:
				new_w = self._get_new_weights(wrong_labels)
				delta_w = (new_w - self.w[x])/2
				S = self._select_examples(wrong_labels, delta_w[wrong_labels]) #cabe melhoria

			selected_ids = np.array(S)

		self.selected[selected_ids] = True
		return selected_ids	


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
		
		if len(S)==0:
			S = None
		
		return S