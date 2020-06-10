import unittest
import os
from context import machine_teacher

_FILE_PATH = os.path.dirname(__file__)
_CONF_PATH = os.path.join(_FILE_PATH, "configurations")

class ReadConf(unittest.TestCase):
	def test_read_conf_1(self):
		pass

if __name__ == "__main__":
	path = os.path.join(_CONF_PATH, "conf1.conf")

	confs = machine_teacher.Reports.read_configuration_file(path)
	for c in confs:
		print("***************")
		print(c)