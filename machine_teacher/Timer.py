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
		s = '\n'.join('{} = {}'.format(n,v) for (n,v) in zip(v_names, v_values))
		return s