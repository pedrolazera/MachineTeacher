from datetime import datetime
from copy import deepcopy
from copy import copy

from ..GenericTeacher import Teacher
from ..GenericLearner import Learner
from ..Definitions import Labels
from .Timer import Timer

class TeachResult:
	_DT_FORMAT = "%Y-%m-%d %H:%M"
	_DATASET_STD_NAME = "???"

	def __init__(self, T: Teacher, L: Learner,
		S_ids, h: Labels, timer: Timer,
		num_iters: int,
		teacher_log,
		dataset_name: str = _DATASET_STD_NAME):

		# output
		self.h = h
		self.S_ids = S_ids

		# stats
		self.main_infos = _MainInfos(
			T.name, # teacher_name
			L.name, # learner_name
			dataset_name, # dataset_name
			timer.total_time, # total_time
			num_iters, # num_iters
			len(S_ids), # sample_size
			T._get_accuracy(h), # accuracy,
			timer["get examples"], # get_examples_time
			timer["fit"], # fit time
			timer["predict"] # predict time
			)
		
		self.timer = timer

		# teacher info
		self.teacher_log = teacher_log
		self.teacher_params = copy(T.get_params())

		# learner info
		self.learner_params = copy(L.get_params())

		# other stuff
		self.date = datetime.today().strftime(self._DT_FORMAT)

	def __str__(self):
		s1 = "-- main infos"
		s2 = "date: {}".format(self.date)
		s3 = str(self.main_infos)

		s4 = "\n-- times (in seconds)"
		s5 = str(self.timer)
		
		s6 = "\n-- teacher parameters"
		s7 = "\n".join("{}: {}".format(a,b) for (a,b) in self.teacher_params.items())

		s8 = "\n-- learner parameters"
		s9 = "\n".join("{}: {}".format(a,b) for (a,b) in self.learner_params.items())

		return '\n'.join((s1,s2,s3,s4,s5,s6,s7,s8,s9))

	def __add__(self, other):
 		new = deepcopy(self)

 		# Nones, things that does not make sense anymore
 		new.teacher_log = None
 		new.teacher_params = dict()
 		new.learner_params = dict()
 		new.date = None

 		# stats
 		new.main_infos += other.main_infos
 		new.timer += other.timer

 		return new

	def __mul__(self, alpha):
		new = deepcopy(self)
		new.main_infos *= alpha
		new.timer *= alpha
		return new

	def __truediv__(self, alpha):
		return self.__mul__(1/alpha)


class _MainInfos:
	def __init__(self, teacher_name: str, learner_name: str,
		dataset_name: str, total_time: float,
		num_iters: int, sample_size: int, accuracy: float,
		get_examples_time: float, fit_time: float, predict_time: float):
		self.teacher_name = teacher_name
		self.learner_name = learner_name
		self.dataset_name = dataset_name

		self.total_time = total_time
		self.sample_size = sample_size
		self.accuracy = accuracy
		self.num_iters = num_iters
		self.get_examples_time = get_examples_time
		self.fit_time = fit_time
		self.predict_time = predict_time

	@staticmethod
	def get_header():
		return ["teacher_name", "learner_name", "dataset_name",
			"total_time", "num_iters", "sample_size", "accuracy",
			"get_examples_time", "fit_time", "predict_time"]

	def get_infos_list(self):
		return [self.teacher_name, self.learner_name,
			self.dataset_name, self.total_time, self.num_iters,
			self.sample_size, self.accuracy,
			self.get_examples_time, self.fit_time, self.predict_time]

	def __add__(self, other):
		assert self.teacher_name == other.teacher_name
		assert self.learner_name == other.learner_name
		assert self.dataset_name == other.dataset_name

		new = deepcopy(self)

		new.total_time += other.total_time
		new.sample_size += other.sample_size
		new.accuracy += other.accuracy
		new.num_iters += other.num_iters
		new.get_examples_time += other.get_examples_time
		new.fit_time += other.fit_time
		new.predict_time += other.predict_time
		
		return new

	def __mul__(self, alpha):
		new = deepcopy(self)

		new.total_time *= alpha
		new.sample_size *= alpha
		new.accuracy *= alpha
		new.num_iters *= alpha
		new.get_examples_time *= alpha
		new.fit_time *= alpha
		new.predict_time *= alpha
		
		return new

	def __truediv__(self, alpha):
		return self.__mul__(1/alpha)

	def __str__(self):
		_v = []
		_v.append("teacher: {}".format(self.teacher_name))
		_v.append("learner: {}".format(self.learner_name))
		_v.append("dataset: {}".format(self.dataset_name))
		_v.append("total_time: {:.3f}".format(self.total_time))
		_v.append("num_iters: {}".format(self.num_iters))
		_v.append("sample size: {}".format(self.sample_size))
		_v.append("accuracy: {:.3f}".format(self.accuracy))
		_v.append("get_examples_time: {:.3f}".format(self.get_examples_time))
		_v.append("fit_time: {:.3f}".format(self.fit_time))
		_v.append("predict_time: {:.3f}".format(self.predict_time))

		return "\n".join(_v)

