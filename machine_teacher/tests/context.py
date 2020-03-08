import sys
import os

_FILE_PATH = os.path.dirname(__file__)
_BASE_PATH = os.path.join(_FILE_PATH, os.path.pardir, os.path.pardir)
_BASE_PATH = os.path.abspath(_BASE_PATH)

sys.path.append(_BASE_PATH)

import machine_teacher