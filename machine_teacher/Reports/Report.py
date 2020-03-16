import csv
from datetime import datetime
from time import sleep
import os
from ..Definitions import InputSpace
from ..Definitions import Labels
from ..GenericTeacher import Teacher
from ..GenericLearner import Learner
from ..Protocol import TeachResult
from .ConfigurationReader import read_configuration_file

_SUFIX_FORMAT = "%Y_%m_%d_%H_%M_%S_%f"

def create_report_from_configuration_file(src_path: str,
    dest_path: str) -> None:
    raise NotImplementedError

def create_reports(v, dest_folder_path: str):
    assert os.path.isdir(dest_folder_path)

    _sufix = datetime.today().strftime(_SUFIX_FORMAT)

    # create subfolder
    new_folder_name = "experiment_set_{}".format(_sufix)
    new_folder_path = os.path.join(dest_folder_path, new_folder_name)
    os.mkdir(new_folder_path)

    create_comparison_table_report(v, new_folder_path)

    for teach_result in v:
        create_report(teach_result, new_folder_path)
        sleep(0.001) # avoid name colision

def create_comparison_table_report(v,
    dest_folder_path: str) -> None:
    # build report
    header = v[0].main_infos.get_header()
    report_lines = [header]
    for teach_result in v:
        report_lines.append(teach_result.main_infos.get_infos_list())

    file_path = os.path.join(dest_folder_path,
        "reports_summary.csv")

    # write report
    with open(file_path, "w", newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(report_lines)

def create_report(TR: TeachResult, dest_folder_path: str) -> None:
    assert os.path.isdir(dest_folder_path)

    # seconds since epoch, unique id
    _sufix = datetime.today().strftime(_SUFIX_FORMAT)

    # create subfolder
    new_folder_name = "experiment_{}".format(_sufix)
    new_folder_path = os.path.join(dest_folder_path, new_folder_name)
    os.mkdir(new_folder_path)

    # create summary file
    summary_file_name = "summary_{}.txt".format(_sufix)
    summary_file_path = os.path.join(new_folder_path,
        summary_file_name)
    _convert_teach_result_to_txt(TR, summary_file_path)

    # create csv file
    teacher_log_file_name = "teacher_log{}.csv".format(_sufix)
    teacher_log_file_path = os.path.join(new_folder_path,
        teacher_log_file_name)
    _convert_teacher_log_to_csv(TR.teacher_log, teacher_log_file_path)

def _convert_teacher_log_to_csv(log, path: str):
    assert os.path.isdir(os.path.dirname(path))

    with open(path, "w", newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(log)

def _convert_teach_result_to_txt(TR: TeachResult, path: str):
    assert os.path.isdir(os.path.dirname(path))

    with open(path, "w") as fp:
        fp.write(str(TR))
