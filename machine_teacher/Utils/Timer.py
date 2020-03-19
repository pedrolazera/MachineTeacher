from timeit import default_timer

class Timer:
	_OFF_STATE = 0
	_ON_STATE = 1
	_TICK_STATE = 2

	def __init__(self):
		self._d = dict()
		self._d_t0 = dict()
		self._state = Timer._OFF_STATE

	def start(self):
		self.total_time = 0.0
		self.others_time = 0.0

		self._t0_total_time = default_timer()

		self._d.clear()
		self._d_t0.clear()

		self._state = Timer._ON_STATE

	def tick(self, field):
		assert (self._state == Timer._ON_STATE), "cannot tick twice or tick before start"
		
		self._d_t0[field] = default_timer()
		self.curr_field = field

		self._state = Timer._TICK_STATE

	def tock(self):
		assert (self._state == Timer._TICK_STATE), "cannot tock before tick"

		curr_field = self.curr_field
		delta = default_timer() - self._d_t0[curr_field]
		self._d[curr_field] = self._d.get(curr_field, 0.0) + delta

		self._state = Timer._ON_STATE

	def finish(self):
		assert (self._state != Timer._OFF_STATE), "cannot finish before start"
		
		if self._state == Timer._TICK_STATE:
			self.tock()

		self.total_time = default_timer() - self._t0_total_time
		self.others_time = self.total_time - sum(self._d.values())

		self._state = Timer._OFF_STATE

	def __str__(self):
		d_names = list(self._d.keys())
		d_values = [self._d[name] for name in d_names]
		v_names = ["total_time"] + d_names + ["others_time"]
		v_values = [self.total_time] + d_values + [self.others_time]
		s = '\n'.join('{} = {:.3f}'.format(n,v) for (n,v) in zip(v_names, v_values))
		return s

	def __add__(self, other):
		keys1 = list(self._d.keys())
		keys2 = list(other._d.keys())
		assert keys1 == keys2

		_d = dict()
		for k in keys1: # or keys2, whatever
			_d[k] = self._d[k] + other._d[k]

		total_time = self.total_time + other.total_time
		others_time = self.others_time + other.others_time

		new_timer = Timer()
		new_timer._d = _d
		new_timer.total_time = total_time
		new_timer.others_time = others_time

		return new_timer

	def __mul__(self, alpha):
		new_timer = Timer()

		for k in self._d.keys():
			new_timer._d[k] = self._d[k] * alpha

		new_timer.total_time = self.total_time * alpha
		new_timer.others_time = self.others_time * alpha

		return new_timer

	def __truediv__(self, alpha):
		return self.__mul__(1/alpha)

	def __getitem__(self, key):
		return self._d[key]
