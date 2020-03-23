from .. import GenericTeacher
import numpy as np

class SingleBatchTeacher(GenericTeacher.Teacher):
	name = "SingleBatchTeacher"

	def start(self, X, y):
		return self._start(X, y)
		
	def keep_going(self, h):
		return False

	def get_first_examples(self):
		return self.ids

	def get_new_examples(self, h):
		raise Error("this should never be called")