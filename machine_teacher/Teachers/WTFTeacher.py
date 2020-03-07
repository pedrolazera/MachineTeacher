from .. import GenericTeacher
import numpy as np

class WTFTeacher(GenericTeacher.Teacher):
	def start(self, X: InputSpace, y: Labels,
		frac_start: float = 0.01, frac_stop: float = 0.5,
		seed: int = 0) -> None:
		self.X = X
		self.y = y

		m = X.shape[0] # nb of rows
		self.m = m
		self.ids = np.arange(m)
		self.S_max_size = int(m * frac_stop)
		self.first_batch_size = int(m * frac_start)
		self.num_iters = 0
		self.selected = np.full(m, False)
		self._random = np.random.RandomState(self.seed)
		self.w = np.full(m, 1/(2.0*m))	
		self.acc_list = [] # verificar se preciso disso mesmo
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
			selected_ids = self._random.choice(self.ids, n,
				replace=False)
			self.selected[selected_ids] = True
			return selected_ids
		else:
			raise NotImplementedError

	def _get_delta_h(self, h):
		return self.ids[self.h != self.y]

	def _update_weights(self, delta_h):
		raise NotImplementedError
