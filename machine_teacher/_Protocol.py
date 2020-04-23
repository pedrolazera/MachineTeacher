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
	"preprocess", "get first examples", "get examples",
	"build training set and labels for new examples",
	"fit", "predict", "build teacher log")

_TIME_LIMIT = 1000000000.0 # in seconds

def teach(T: Teacher, L: Learner,
	X: InputSpace, X_labels: Labels, *,
	dataset_name = TeachResult._DATASET_STD_NAME,
	time_limit = _TIME_LIMIT) -> TeachResult:
	timer = Timer()
	timer.start()
	_set_timer_keys_to_zero(timer, _TIMER_KEYS)
	teacher_log = [T.get_log_header()]

	# wrappers
	X = wrapp_input_space(X)
	X_labels = wrapp_labels(X_labels)

	S_ids = np.array([], dtype=int)

	timer.tick("preprocess")
	L.start()
	T.start(X, X_labels)
	timer.tock()

	# get initial training set
	timer.tick("get first examples")
	new_ids = T.get_first_examples()
	timer.tock()
	
	qtd_iters = 0
	while (timer.get_elapsed_time() < time_limit) and (len(new_ids) > 0):
		qtd_iters += 1

		S_ids = np.append(S_ids, new_ids)

		timer.tick("fit")
		L.fit(X[new_ids], X_labels[new_ids])
		timer.tock()

		teacher_log.append(T.get_log_line(L.predict(X)))

		timer.tick("predict")
		new_ids = get_new_examples(T, L, X)
		timer.tock()

	h = L.predict(X)
	teacher_log.append(T.get_log_line(L.predict(X)))

	timer.finish()

	#assert qtd_iters+1 == len(teacher_log)

	return TeachResult(T, L, S_ids, h, timer, qtd_iters,
		get_qtd_columns(X), teacher_log, dataset_name)

def get_new_examples(T, L, X):
	test_ids = T.get_first_test_ids()
	h = L.predict(X[test_ids])

	while T.keep_testing():
		test_ids = T.get_test_ids(h)
		h = L.predict(X[test_ids])

	return T.get_new_examples(h)

def _set_timer_keys_to_zero(timer, keys):
	for key in keys:
		timer.tick(key)
		timer.tock()