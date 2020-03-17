import os
import sys

sys.path.append(os.path.abspath(".."))

import machine_teacher

configuration_folder = os.path.join(".", "configs", "config_folder_1")
configuration_folder = os.path.abspath(configuration_folder)
dest_folder = os.path.join(".", "results")
dest_folder = os.path.abspath(dest_folder)

machine_teacher.Reports.create_reports_from_configuration_folder(configuration_folder,
	dest_folder)