import numpy as np
from datetime import datetime

from .Utils.Timer import Timer
from .Utils.TeachResult import TeachResult

from .GenericTeacher import Teacher
from .GenericLearner import Learner

from .Definitions import InputSpace
from .Definitions import Labels
from .Definitions import wrapp_labels
from .Definitions import wrapp_input_space
from .Definitions import get_qtd_columns
from .Definitions import get_qtd_rows

_TIMER_KEYS = (
	"preprocess", "fit", "predict", "get_examples")

_TIME_LIMIT = 1000000000.0 # in seconds

def teach(T: Teacher, L: Learner,
	X: InputSpace, X_labels: Labels, *,
	dataset_name = TeachResult._DATASET_STD_NAME,
	time_limit = _TIME_LIMIT) -> TeachResult:

	# timer and teacher log
	timer = Timer()
	timer.start()
	_set_timer_keys_to_zero(timer, _TIMER_KEYS)
	teacher_log = [T.get_log_header()] # not being used so far

	# wrappers
	X = wrapp_input_space(X)
	X_labels = wrapp_labels(X_labels)

	# start with empty set of <training example ids>
	train_ids = np.array([], dtype=int)

	# initialization
	timer.tick("preprocess")
	L.start()
	T.start(X, X_labels)
	timer.tock()

	# first teaching interaction
	
	## get first examples
	timer.tick("get_examples")
	new_train_ids = T.get_first_examples()
	assert 0 < len(new_train_ids) <=  get_qtd_rows(X)
	timer.tock()

	# fit first examples
	train_ids = np.append(train_ids, new_train_ids)
	timer.tick("fit")
	L.fit(X[train_ids], X_labels[train_ids])
	timer.tock()

	# other teaching interactions
	qtd_iters = 1
	while (timer.get_elapsed_time() < time_limit):
		timer.tick("predict")
		test_ids, test_labels = _run_tests(T, L, X)
		timer.tock()

		# get log - time counting stops
		#timer.stop()
		#h = L.predict(X)
		#teacher_log.append(T.get_log_line(h))
		#timer.unstop()

		timer.tick("get_examples")
		new_train_ids = T.get_new_examples(test_ids, test_labels)
		timer.tock()

		if len(new_train_ids) > 0:
			timer.tick("fit")
			train_ids = np.append(train_ids, new_train_ids)

			assert len(train_ids) <= get_qtd_rows(X) 
			
			L.fit(X[train_ids], X_labels[train_ids])
			timer.tock()
			qtd_iters += 1
		else:
			break

	timer.finish()

	# get last log line
	h = L.predict(X)
	#teacher_log.append(T.get_log_line(h))

	return TeachResult(T, L, train_ids, h, timer, qtd_iters,
		get_qtd_columns(X), teacher_log, time_limit, dataset_name)

def _run_tests(T, L, X):
	test_ids = np.array([], dtype=int)
	test_labels = np.array([], dtype=int)

	while len(test_ids) <= get_qtd_rows(X):
		new_test_ids = T.get_new_test_ids(test_ids, test_labels)
		if len(new_test_ids) > 0:
			assert len(new_test_ids) + len(test_ids) <= get_qtd_rows(X)

			new_test_labels = L.predict(X[new_test_ids])
			test_ids = np.append(test_ids, new_test_ids)
			test_labels = np.append(test_labels, new_test_labels)
		else:
			break

	return (test_ids, test_labels)

def _set_timer_keys_to_zero(timer, keys):
	for key in keys:
		timer.tick(key)
		timer.tock()