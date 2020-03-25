import os
import sys

sys.path.append(os.path.abspath(".."))

from  machine_teacher.Reports import create_reports_from_configuration_file

_CONFIGURATION_BASE_FOLDER = os.path.join(".", "configs")
_CONFIGURATION_BASE_FOLDER = os.path.abspath(_CONFIGURATION_BASE_FOLDER)
_DEST_FOLDER = os.path.join(".", "results")
_DEST_FOLDER = os.path.abspath(_DEST_FOLDER)


_ERROR_MSG = "should be 'python main_from_file.py <configuration_file_name>'"

if __name__ == "__main__":
	if len(sys.argv) != 2:
		raise KeyError(_ERROR_MSG)

	conf_file_name = sys.argv[1].strip()
	conf_file_path = os.path.join(_CONFIGURATION_BASE_FOLDER,
		conf_file_name)

	print("conf_file_path:", conf_file_path)

	create_reports_from_configuration_file(conf_file_path,
		_DEST_FOLDER, True)