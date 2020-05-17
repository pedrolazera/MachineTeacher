import numpy as np
from datetime import datetime
from copy import copy
from sklearn.utils import shuffle

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

_TIMER_KEYS = ("training", "classification", "get_examples")

_HEADER = ("iter", "TS_size", "accuracy", "elapsed_time",
	"get_examples_time", "training_time", "classification_time",
	"qtd_classified_examples", "TS_qtd_classes", "TS_dist_classes")

_TIME_LIMIT = 1000000000.0 # in seconds

_SHUFFLE_RANDOM_STATE = 0
_SHUFFLE_DATASET = False

def teach(T: Teacher, L: Learner,
	X: InputSpace, X_labels: Labels, *,
	dataset_name = TeachResult._DATASET_STD_NAME,
	time_limit = _TIME_LIMIT,
	shuffle_dataset = _SHUFFLE_DATASET,
	shuffle_random_state = _SHUFFLE_RANDOM_STATE) -> TeachResult:
	if shuffle_dataset:
		X, X_labels = shuffle(X, X_labels, random_state = shuffle_random_state)

	# timer
	timer = Timer()
	ok_timer = None
	timer.start()
	get_time_left = lambda: time_limit - timer.get_elapsed_time()
	_set_timer_keys_to_zero(timer, _TIMER_KEYS)

	# teacher log
	log = [_HEADER] # not being used so far
	test_ids = np.array([], dtype=int)

	# wrappers
	X = wrapp_input_space(X)
	X_labels = wrapp_labels(X_labels)

	# checks
	assert len(np.unique(X_labels)) > 1 # tem que existir mais de uma classe no dataset
	assert np.min(X_labels) == 0

	# start with empty set of <training example ids>
	train_ids = np.array([], dtype=int)
	ok_train_ids = None

	# initialization
	L.start()
	T.start(X, X_labels, get_time_left())

	# first teaching interaction
	
	## get first examples
	timer.tick("get_examples")
	new_train_ids = T.get_first_examples(get_time_left())
	assert 0 < len(new_train_ids) <=  get_qtd_rows(X)
	timer.tock()

	## fit first examples
	train_ids = np.append(train_ids, new_train_ids)
	timer.tick("training")
	L.fit(X[train_ids], X_labels[train_ids])
	timer.tock()

	# other teaching interactions
	qtd_iters = 0
	while (get_time_left() > 0):
		# copy last "ok" state and build log line
		qtd_iters += 1
		timer.stop()
		ok_timer = copy(timer)
		ok_timer.finish()
		ok_train_ids = train_ids[:]
		_accuracy = T._get_accuracy(L.predict(X))
		_qtd_classes_iter = len(np.unique(X_labels[ok_train_ids]))
		_dist_classes_iter = np.bincount(X_labels[ok_train_ids])
		_dist_classes_iter = ",".join(["{:.2f}".format(i/len(ok_train_ids)) for i in _dist_classes_iter])
		_log_line = (qtd_iters, len(ok_train_ids), _accuracy,
			ok_timer.get_elapsed_time(), ok_timer["get_examples"],
			ok_timer["training"], ok_timer["classification"],
			len(test_ids), _qtd_classes_iter, _dist_classes_iter)
		log.append(_log_line)
		timer.unstop()

		# run next iteration
		timer.tick("classification")
		test_ids, test_labels = _run_tests(T, L, X, get_time_left)
		timer.tock()

		timer.tick("get_examples")
		new_train_ids = T.get_new_examples(test_ids, test_labels, get_time_left())
		timer.tock()

		if len(new_train_ids) > 0:
			timer.tick("training")
			train_ids = np.append(train_ids, new_train_ids)

			assert len(train_ids) <= get_qtd_rows(X)
			
			L.fit(X[train_ids], X_labels[train_ids])
			timer.tock()
		else:
			break

	# adiciona ultima linha do log caso o último estado tenha sido revertido
	if get_time_left() < 0:
		timer.finish()
		_qtd_classes_iter = len(np.unique(X_labels[train_ids]))
		_dist_classes_iter = np.bincount(X_labels[train_ids])
		_dist_classes_iter = ",".join(["{:.2f}".format(i/len(train_ids)) for i in _dist_classes_iter])
		_log_line = (qtd_iters+1, len(train_ids), T._get_accuracy(L.predict(X)),
			timer.get_elapsed_time(), timer["get_examples"],
			timer["training"], timer["classification"],
			len(test_ids), _qtd_classes_iter, _dist_classes_iter)
		log.append(_log_line)

	# sanity checks
	assert qtd_iters >= 1, "there was no training..." + str((T.name, L.name, dataset_name))
	assert ok_timer is not None
	assert ok_train_ids is not None
	assert len(ok_train_ids) == len(set(ok_train_ids))

	# monta o teaching result
	# # hipótese final do learner
	L.fit(X[ok_train_ids], X_labels[ok_train_ids])
	h = L.predict(X)

	# # qtd classes e distribuicao das classes
	qtd_classes = len(np.unique(X_labels))
	dist_classes = np.bincount(X_labels)
	dist_classes = ",".join(["{:.2f}".format(i/len(X_labels)) for i in dist_classes])

	return TeachResult(T, L, ok_train_ids, h, ok_timer, qtd_iters,
		get_qtd_columns(X), log, time_limit, qtd_classes, dist_classes, dataset_name)

def _run_tests(T: Teacher, L: Learner,
	X: InputSpace, get_time_left):
	test_ids = np.array([], dtype=int)
	test_labels = np.array([], dtype=int)

	while len(test_ids) <= get_qtd_rows(X):
		new_test_ids = T.get_new_test_ids(test_ids, test_labels, get_time_left())
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