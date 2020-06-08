import numpy as np
from . import Experiment3Teacher

class Experiment4Teacher(Experiment3Teacher):
	name = "Experiment4Teacher"
	_BATCH_RELATIVE_SIZE = 0.005 # 0.5%

	def __init__(self,
		safity: float,
		seed: int = Experiment3Teacher._SEED,
		frac_start: float = Experiment3Teacher._FRAC_START,
		frac_time_change: float = Experiment3Teacher._FRAC_TIME_CHANGE,
		strategy: int = Experiment3Teacher.DoubleTeacher._STRATEGY_DOUBLE_SIZE,
		shuffle: bool = Experiment3Teacher._SHUFFLE,
		batch_relative_size = _BATCH_RELATIVE_SIZE):
		super().__init__(safity, seed, frac_start,
			frac_time_change, strategy)
		assert 0.0 < batch_relative_size <= 1.0, "batch_relative_size most be in (0, 1]"
		self.batch_relative_size = batch_relative_size

	def start(self, X, y, time_left: float):
		super().start(X, y, time_left)

		# calcula e guarda batch_size
		batch_size = self.m * self.batch_relative_size
		batch_size = np.ceil(batch_size)
		batch_size = min(self.m, self.batch_size)
		self.batch_size = batch_size

	def get_new_test_ids(self, test_ids, test_labels,
		time_left: float) -> np.ndarray:
		raise NotImplementedError

	def get_params(self) -> dict:
		_d = super().get_params()
		_d["batch_relative_size"] = self.batch_relative_size
		return _d

