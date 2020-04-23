import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath(os.path.join("..", "machine_teacher", "Utils")))

import  machine_teacher
from DatasetLoader import load_dataset_from_path

_PATH_FOLDER_DATASETS = os.path.abspath(os.path.join("..", "garagem", "datasets"))
_PATH_FOLDER_RESULTS = os.path.abspath(os.path.join(".", "results"))
_FAMILY_SUFIX_FORMAT = "%Y_%m_%d_%H_%M_%S"
_SEED = 0

def _get_dataset_path(dataset_name):
	dataset_path = os.path.join(_PATH_FOLDER_DATASETS, dataset_name)
	return dataset_path

def get_X_y(dataset_name, is_numeric):
	dataset_path = _get_dataset_path(dataset_name)
	(X, y) = load_dataset_from_path(dataset_path, is_numeric)
	return (X, y)

def get_avg_TR(TRs):
	if len(TRs) > 1:
		avg_TR = sum(TRs[1:], TRs[0]) / len(TRs)
	else:
		avg_TR = TRs[0]

	return avg_TR

def _get_dst_folder_and_sufix():
	_sufix = datetime.today().strftime(_FAMILY_SUFIX_FORMAT)
	folder_name = "family_" + _sufix
	folder_path = os.path.join(_PATH_FOLDER_RESULTS, folder_name)
	folder_path = os.path.abspath(folder_path)
	return (folder_path, _sufix)

def _convert_summary_file_to_xlsx(src, dst):
	assert os.path.isfile(src)
	assert src.endswith(".csv")
	
	df = pd.read_csv(src, header = 0)
	df.to_excel(dst, index = False)

def _get_summary_file_name(res_folder_path):
	sufix = os.path.basename(os.path.normpath(res_folder_path))[6:]
	summary_file_name = "reports_summary" + sufix + ".csv"
	return summary_file_name

# experiment parameters
qtd_runs_per_test = 5

# Learner
get_learners = [
	machine_teacher.Learners.RandomForestLearner,
	machine_teacher.Learners.SVMLearner
]

learner_params = [
	{
	"random_state": 0,
	"n_estimators": 10,
	},
	{
	"random_state": 0,
	}
]

# Teacher
get_teachers = [
	machine_teacher.Teachers.SingleBatchTeacher
]

teacher_params = [
	{}
]

# datasets
dataset_rel_sizes = [1/size for size in range(1,6)]
datasets_names = [
	("avila_tr.csv", True),
	("bank_marketing_dataset.csv", True),
	("default_of_credit_card_clients.csv", True),
	("Sensorless_drive_diagnosis.csv", True),
]

# create root subfolder
dst_folder, sufix = _get_dst_folder_and_sufix()
os.mkdir(dst_folder)

# run tests
TRs = []
for (get_learner, learner_dic_params) in zip(get_learners, learner_params):
	for (get_teacher, teacher_dic_params) in zip(get_teachers, teacher_params): 
		for (dataset_name, is_numeric) in datasets_names:
			for rel_size in dataset_rel_sizes:
				assert rel_size <= 1.0

				X, y = get_X_y(dataset_name, is_numeric)
				m_rel_size = int(y.size * rel_size)

				assert m_rel_size >= 1

				# get subset with rel_size and random examples
				shuffle_fun = np.random.RandomState(_SEED).shuffle
				ids = np.arange(y.size)
				shuffle_fun(ids)
				ids = ids[:m_rel_size]

				X = X[ids]
				y = y[ids]
			
				TRs_i = []

				for __ in range(qtd_runs_per_test):
					L = get_learner(**learner_dic_params)
					T = get_teacher(**teacher_dic_params)
					TR_i = machine_teacher.teach(T, L, X, y,
						dataset_name = dataset_name)
					TRs_i.append(TR_i)

					print(TR_i)

				#machine_teacher.Reports.create_reports(TRs_i, dst_folder)

				TRs_i_avg = get_avg_TR(TRs_i)

				TRs.append(TRs_i_avg)

machine_teacher.Reports.create_comparison_table_report(TRs, dst_folder, sufix)

# convert summary to excel
summary_file_name = _get_summary_file_name(dst_folder)
summary_file_path = os.path.join(dst_folder, summary_file_name)
excel_summary_file_path = summary_file_path.replace(".csv", ".xlsx")

_convert_summary_file_to_xlsx(summary_file_path, excel_summary_file_path)

