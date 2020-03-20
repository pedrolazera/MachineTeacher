from .. import GenericTeacher
import numpy as np

class MaxSizeTeacher(GenericTeacher.Teacher):
	name = "MaxSizeTeacher"
	pass

	def start(self, X, y):
		raise NotImplementedError
		
	def keep_going(self, h):
		raise NotImplementedError

	def get_first_examples(self):
		raise NotImplementedError

	def get_new_examples(self, h):
		raise NotImplementedError