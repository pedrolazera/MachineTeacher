import numpy as np
from datetime import datetime

from .Utils.Timer import Timer

from .GenericTeacher import Teacher
from .GenericLearner import Learner

from .Definitions import InputSpace
from .Definitions import Labels
from .Definitions import wrapp_labels
from .Definitions import wrapp_input_space
from .Definitions import get_qtd_columns
from .Definitions import get_qtd_rows

_DATASET_STD_NAME = "???"

class TeachResult:
	_DT_FORMAT = "%Y-%m-%d %H:%M"
	def __init__(self, T: Teacher, L: Learner,
		S_ids, h: Labels, timer: Timer,
		num_iters: int,
		teacher_log,
		dataset_name: str = _DATASET_STD_NAME):

		# output
		self.S_ids = S_ids
		self.h = h
		self.timer = timer
		self.num_iters = num_iters

		# teacher
		self.teacher_log = teacher_log
		self.teacher_name = T.name
		self.teacher_params = T.get_params()

		# learner
		self.learner_name = L.name
		self.learner_params = L.get_params()

		# other stuff
		self.date = datetime.today().strftime(self._DT_FORMAT)
		self.dataset_name = dataset_name

	def __str__(self):
		s0 = "-- summary"
		s1 = "date: {}".format(self.date)
		s2 = "teacher: {}".format(self.teacher_name)
		s3 = "learner: {}".format(self.learner_name)
		s4 = "dataset: {}".format(self.dataset_name)
		s5 = "sample size: {}".format(len(self.S_ids))
		s6 = "num_iters: {}".format(self.num_iters)

		s7 = "\n-- times (in seconds)"
		s8 = str(self.timer)
		
		s9 = "\n-- teacher parameters"
		s10 = "\n".join("{}: {}".format(a,b) for (a,b) in self.teacher_params.items())

		s11 = "\n-- learner parameters"
		s12 = "\n".join("{}: {}".format(a,b) for (a,b) in self.learner_params.items())

		return '\n'.join((s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12))		

def teach(T: Teacher, L: Learner,
	X: InputSpace, X_labels: Labels,
	dataset_name = _DATASET_STD_NAME) -> TeachResult:
	timer = Timer()
	timer.start()
	teacher_log = [T.get_log_header()]

	# wrappers
	X = wrapp_input_space(X)
	X_labels = wrapp_labels(X_labels)

	S_ids = np.array([], dtype=int)

	timer.tick("preprocess")
	T.start(X, X_labels)
	timer.tock()

	# get initial training set
	timer.tick("get first examples")
	new_ids = T.get_first_examples()
	timer.tock()

	S_ids = np.append(S_ids, new_ids)
	h = _run_one_round(T, L, X, X_labels, new_ids,
		timer, teacher_log)
	
	num_iters = 1
	while T.keep_going(h):
		num_iters += 1

		timer.tick("get examples")
		new_ids = T.get_new_examples(h) # examples ids
		timer.tock()

		S_ids = np.append(S_ids, new_ids)
		h = _run_one_round(T, L, X, X_labels, new_ids,
			timer, teacher_log)

	timer.finish()

	assert num_iters+1 == len(teacher_log)

	return TeachResult(T, L, S_ids, h, timer, num_iters,
		teacher_log, dataset_name)

def _run_one_round(T, L, X, X_labels, new_ids,
	timer, teacher_log):
	# update input subspace S and respective labels S_labels
	timer.tick("build training set and labels for new examples")
	S_i = X[new_ids]
	S_labels_i = X_labels[new_ids]
	timer.tock()

	# fit
	timer.tick("fit")
	L.fit(S_i, S_labels_i)
	timer.tock()

	# preditc
	timer.tick("predict")
	h = L.predict(X)
	timer.tock()
	h = wrapp_labels(h) # reshape to 1d array, if needed

	# get teacher iteration log
	timer.tick("build teacher log")
	_teacher_log_line = T.get_log_line(h)
	teacher_log.append(_teacher_log_line)
	timer.tock()

	# double checks
	assert len(h) == len(X_labels)
	assert get_qtd_columns(S_i) == get_qtd_columns(X)

	return h
