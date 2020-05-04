import unittest
from .context import Timer
from math import isclose
from timeit import default_timer
from time import sleep
from copy import copy

_EPS = 1e-1

class States(unittest.TestCase):
	def test_do_stuff_before_start(self):
		t = Timer.Timer()
		with self.assertRaises(AssertionError):
			t.tick("blabla")

		with self.assertRaises(AssertionError):
			t.tock()

		with self.assertRaises(AssertionError):
			t.stop()

		with self.assertRaises(AssertionError):
			t.unstop()

		with self.assertRaises(AssertionError):
			t.finish()

	def test_do_forbidden_stuff_during_tick(self):
		t = Timer.Timer()
		t.start()
		t.tick("xpto")

		with self.assertRaises(AssertionError):
			t.unstop()

		with self.assertRaises(AssertionError):
			t.tick("xpto2")

	def test_do_forbidden_stuff_during_stop(self):
		t = Timer.Timer()
		t.start()
		t.tick("xpto")
		t.stop()

		with self.assertRaises(AssertionError):
			t.tock()

		with self.assertRaises(AssertionError):
			t.stop()

	def test_tick_times(self):
		t = Timer.Timer()
		t.start()

		# single 1 sec
		t.tick("xpto1")
		sleep(1.0)
		t.tock()

		self.assertAlmostEqual(t["xpto1"], 1.0, delta = _EPS)

		# 2.0 sec, divided in two parts
		t.tick("xpto2")
		sleep(0.5)
		t.tock()

		t.tick("xpto2")
		sleep(1.5)
		t.tock()

		self.assertAlmostEqual(t["xpto2"], 2.0, delta = _EPS)

		# 6.0 sec, divided in two parts
		t.tick("xpto3")
		sleep(0.5)
		t.tock()

		t.tick("xpto3")
		sleep(1.5)
		t.tock()

		t.tick("xpto3")
		sleep(4.0)
		t.tock()

		self.assertAlmostEqual(t["xpto3"], 6.0, delta = _EPS)

		# alternating ticks
		t.tick("xpto4")
		sleep(0.5)
		t.tock()

		t.tick("xpto5")
		sleep(0.5)
		t.tock()

		t.tick("xpto4")
		sleep(1.5)
		t.tock()

		t.tick("xpto5")
		sleep(2.0)
		t.tock()

		self.assertAlmostEqual(t["xpto4"], 2.0, delta = _EPS)
		self.assertAlmostEqual(t["xpto5"], 2.5, delta = _EPS)

	def test_tick_times_with_stoppage(self):
		t = Timer.Timer()
		t.start()

		t.tick("xpto1")
		sleep(1.0)
		t.tock()

		t.stop()
		sleep(0.4) # should change nothing
		t.unstop()

		t.tick("xpto2")
		sleep(0.5)
		t.tock()

		sleep(0.3) # should change only total time elapsed

		t.stop()
		sleep(0.3)
		t.unstop()

		t.tick("xpto3")
		sleep(0.5)
		t.tock()

		t.tick("xpto2")
		sleep(0.7)
		

		t.stop() # this should causes t.tock()

		self.assertAlmostEqual(t["xpto1"], 1.0, delta = _EPS)
		self.assertAlmostEqual(t["xpto2"], 1.2, delta = _EPS)
		self.assertAlmostEqual(t["xpto3"], 0.5, delta = _EPS)


	def test_total_time(self):
		t = Timer.Timer()
		t.start()

		t.tick("xpto1")
		sleep(1.0)
		t.tock()

		elaps1 = 1.0
		elaps1_timer = t.get_elapsed_time()

		t.stop()
		sleep(0.4)
		t.unstop()

		elaps2 = 1.0
		elaps2_timer = t.get_elapsed_time()

		t.tick("xpto2")
		sleep(0.5)
		t.tock()

		sleep(0.3) # should change only total time elapsed

		elaps3 = 1.0 + 0.5 + 0.3
		elaps3_timer = t.get_elapsed_time()

		t.stop()
		sleep(0.3)
		elaps4 = 1.0 + 0.5 + 0.3
		elaps4_timer = t.get_elapsed_time()
		t.unstop()

		t.tick("xpto3")
		sleep(0.5)
		t.tock()

		t.tick("xpto2")
		sleep(0.7)
		t.stop() # this should causes t.tock()

		t.finish()

		elaps5 = 1.0 + 0.5 + 0.3 + 0.5 + 0.7
		elaps5_timer = t.get_elapsed_time()

		self.assertAlmostEqual(elaps1, elaps1_timer, delta = _EPS)
		self.assertAlmostEqual(elaps2, elaps2_timer, delta = _EPS)
		self.assertAlmostEqual(elaps3, elaps3_timer, delta = _EPS)
		self.assertAlmostEqual(elaps4, elaps4_timer, delta = _EPS)
		self.assertAlmostEqual(elaps5, elaps5_timer, delta = _EPS)
		self.assertAlmostEqual(elaps5, t.total_time, delta = _EPS)

	def test_finished(self):
		t = Timer.Timer()
		t.start()
		t.tick("blabla")
		sleep(1.3)
		t.finish()

		elaps1 = 1.3
		elaps1_timer = t.get_elapsed_time()
		elaps2_timer = t.total_time

		with self.assertRaises(AssertionError):
			t.tick("blabla")

		with self.assertRaises(AssertionError):
			t.tock()

		with self.assertRaises(AssertionError):
			t.stop()

		with self.assertRaises(AssertionError):
			t.unstop()
		
		self.assertAlmostEqual(elaps1, elaps1_timer, delta = _EPS)
		self.assertAlmostEqual(elaps1_timer, elaps2_timer, delta = _EPS)

	def test_copy(self):
		# copy after start
		t = Timer.Timer()
		t.start()
		sleep(0.5)
		t2 = copy(t)
		t.finish()
		self.assertEqual(t2._state, t._ON_STATE)

		# copy after tick
		t = Timer.Timer()
		t.start()
		sleep(0.5)
		t.tick("xpto")
		t2 = copy(t)
		t.finish()
		sleep(0.7)

		self.assertEqual(t._state, t._FINISHED_STATE)
		self.assertEqual(t2._state, t2._TICK_STATE)

		t2.tock()
		self.assertEqual(t2._state, t._ON_STATE)

		t2.finish()
		self.assertEqual(t2._state, t._FINISHED_STATE)
		self.assertAlmostEqual(t2.total_time, 0.5 + 0.7, delta = _EPS)


