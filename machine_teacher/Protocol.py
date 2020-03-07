import numpy as np

from .Timer import Timer

from .GenericTeacher import Teacher
from .GenericLearner import Learner

from .Definitions import InputSpace
from .Definitions import Labels
from .Definitions import join_input_spaces
from .Definitions import join_labels
from .Definitions import wrapp_labels
from .Definitions import wrapp_input_space
from .Definitions import get_empty_labels
from .Definitions import get_empty_input_space
from .Definitions import get_qtd_columns
from .Definitions import get_qtd_rows

class TeachResult:
	def __init__(self, S, h, timer,
		num_iters, teacher_log):
		self.S = S
		self.h = h
		self.timer = timer
		self.num_iters = num_iters
		self.teacher_log = teacher_log

	def __str__(self):
		s1 = "sample size = {}".format(get_qtd_rows(self.S))
		s2 = "num_iters = {}".format(self.num_iters)
		s3 = str(self.timer)
		return '\n'.join((s1,s2,s3))

def teach(T: Teacher, L: Learner,
	X: InputSpace, X_labels: Labels) -> TeachResult:
	timer = Timer()
	timer.start()
	teacher_log = [T.get_header()]

	# wrappers
	X = wrapp_input_space(X)
	X_labels = wrapp_labels(X_labels) # reshape to 1d array if needed

	# generate empty input subspace and respective labels
	S = get_empty_input_space(X)
	S_labels = get_empty_labels(X_labels)

	timer.tick("preprocess")
	T.start(X, X_labels)
	timer.tock()

	timer.tick("get_h0")
	h = L.get_h0(X)
	timer.tock()
	
	num_iters = 0
	while T.keep_going(h):
		num_iters += 1

		timer.tick("get_examples")
		new_i = T.get_new_examples(h) # refs to the examples in X
		timer.tock()
		
		# update input subspace S and respective labels S_labels
		timer.tick("update sample S")
		S_i = X[new_i]
		S_labels_i = X_labels[new_i]

		S = join_input_spaces(S, S_i)
		S_labels = join_labels(S_labels, S_labels_i)
		timer.tock()

		# fit
		timer.tick("fit")
		L.fit(S_i, S_labels_i)
		timer.tock()

		# preditc
		timer.tick("predict")
		h = L.predict(X)
		timer.tock()

		# get teacher iteration log
		timer.tick("teacher log")
		_teacher_log_line = T.get_log_line(h)
		teacher_log.append(_teacher_log_line)
		timer.tock()

		h = wrapp_labels(h) # reshape to 1d array if needed
		assert len(h) == len(X_labels)
		assert get_qtd_columns(S) == get_qtd_columns(X)

	timer.finish()

	assert num_iters+1 == len(teacher_log)

	return TeachResult(S, h, timer, num_iters, teacher_log)