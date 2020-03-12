import csv
from ..Definitions import InputSpace
from ..Definitions import Labels
from ..GenericTeacher import Teacher
from ..GenericLearner import Learner

def create_report(T: Teacher, L: Learner,
	X: InputSpace, X_labels: Labels, dest_folder_path: str) -> None:
	raise NotImplementedError

def create_report_from_configuration_file(src_path: str,
	dest_path: str) -> None:
	raise NotImplementedError

def _convert_teacher_log_to_csv(log, path):
	dir_path = os.dirname(path)
	assert os.isdir(dir_path), "o path {} nao existe".format(dir_path)
	
	with open(path, "w", newline='') as csv_file:
		csv_writer = csv.writer(csv_file)
		csv_writer.writerows(log)

def _convert_teach_result_to_txt(result, path):
	raise NotImplementedError