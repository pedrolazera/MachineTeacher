import csv
import time
import os
from ..Definitions import InputSpace
from ..Definitions import Labels
from ..GenericTeacher import Teacher
from ..GenericLearner import Learner
from ..Protocol import TeachResult
from .ConfigurationReader import read_configuration_file

def create_report_from_configuration_file(src_path: str,
    dest_path: str) -> None:
    raise NotImplementedError

def create_report(TR: TeachResult, dest_folder_path: str) -> None:
    assert os.isdir(dest_folder_path)

    # seconds since epoch, unique id
    _seconds_since_epoch = int(time.time())

    # create subfolder
    new_folder_name = "experiment_{}".format(_seconds_since_epoch)
    new_folder_path = os.path.join(dest_folder_path, new_folder_name)
    os.mkdir(new_folder_path)

    # create summary file
    summary_file_name = "summary_{}.txt".format(_seconds_since_epoch)
    summary_file_path = os.path.join(new_folder_path,
        summary_file_name)
    _convert_teach_result_to_txt(TR, summary_file_path)

    # create csv file
    teacher_log_file_name = "teacher_log{}.csv".format(_seconds_since_epoch)
    teacher_log_file_pat = os.path.join(new_folder_path,
        teacher_log_file_name)
    _convert_teacher_log_to_csv(TR, teacher_log_file_pat)

def _convert_teacher_log_to_csv(log, path: str):
    assert os.isdir(os.dirname(path))

    with open(path, "w", newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(log)

def _convert_teach_result_to_txt(TR: TeachResult, path: str):
    assert os.isdir(os.dirname(path))

    with open(path, "w") as fp:
        fp.write(str(TR))

def _create_folder_name():
    seconds_since_epoch = int(time.time())
    folder_name = "teste_{}".format(seconds_since_epoch)
    return folder_name