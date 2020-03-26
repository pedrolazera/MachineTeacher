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
		qtd_iters: int,
		qtd_attributes: int,
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
			len(h), #"dataset_qtd_examples"
			qtd_attributes, # qtd_attributes
			timer.total_time, # total_time
			qtd_iters, # qtd_iters
			len(S_ids), # teaching_set_size
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
		dataset_name: str, dataset_qtd_examples: int,
		qtd_attributes: int, total_time: float, qtd_iters: int, 
		teaching_set_size: int, accuracy: float,
		get_examples_time: float, fit_time: float, predict_time: float):
		self.teacher_name = teacher_name
		self.learner_name = learner_name
		self.dataset_name = dataset_name
		self.dataset_qtd_examples = dataset_qtd_examples
		self.qtd_attributes = qtd_attributes

		self.total_time = total_time
		self.teaching_set_size = teaching_set_size
		self.accuracy = accuracy
		self.qtd_iters = qtd_iters
		self.get_examples_time = get_examples_time
		self.fit_time = fit_time
		self.predict_time = predict_time

	@staticmethod
	def get_header():
		return ["teacher_name", "learner_name", "dataset_name",
			"dataset_qtd_examples", "dataset_qtd_attributes",
			"total_time", "qtd_iters",
			"teaching_set_size", "accuracy",
			"get_examples_time", "fit_time", "predict_time"]

	def get_infos_list(self):
		return [self.teacher_name, self.learner_name,
			self.dataset_name, self.dataset_qtd_examples,
			self.qtd_attributes, self.total_time, self.qtd_iters,
			self.teaching_set_size, self.accuracy,
			self.get_examples_time, self.fit_time, self.predict_time]

	def __add__(self, other):
		assert self.teacher_name == other.teacher_name
		assert self.learner_name == other.learner_name
		assert self.dataset_name == other.dataset_name
		assert self.dataset_qtd_examples == other.dataset_qtd_examples
		assert self.qtd_attributes == other.qtd_attributes

		new = deepcopy(self)

		new.total_time += other.total_time
		new.teaching_set_size += other.teaching_set_size
		new.accuracy += other.accuracy
		new.qtd_iters += other.qtd_iters
		new.get_examples_time += other.get_examples_time
		new.fit_time += other.fit_time
		new.predict_time += other.predict_time
		
		return new

	def __mul__(self, alpha):
		new = deepcopy(self)

		new.total_time *= alpha
		new.teaching_set_size *= alpha
		new.accuracy *= alpha
		new.qtd_iters *= alpha
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
		_v.append("dataset qtd examples: {}".format(self.dataset_qtd_examples))
		_v.append("qtd attributes: {}".format(self.qtd_attributes))
		_v.append("total_time: {:.3f}".format(self.total_time))
		_v.append("qtd iters: {}".format(self.qtd_iters))
		_v.append("teaching set size: {}".format(self.teaching_set_size))
		_v.append("accuracy: {:.3f}".format(self.accuracy))
		_v.append("get_examples_time: {:.3f}".format(self.get_examples_time))
		_v.append("fit_time: {:.3f}".format(self.fit_time))
		_v.append("predict_time: {:.3f}".format(self.predict_time))

		return "\n".join(_v)

