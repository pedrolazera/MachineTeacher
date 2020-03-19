import csv
import os
from datetime import datetime
from time import sleep
from ..Definitions import InputSpace
from ..Definitions import Labels
from ..GenericTeacher import Teacher
from ..GenericLearner import Learner
from ..Utils import TeachResult
from .ConfigurationReader import read_configuration_file
from ..Protocol import teach
from ..Utils.TeacherLearnerLoader import get_teacher
from ..Utils.TeacherLearnerLoader import get_learner
from ..Utils.DatasetLoader import load_dataset_from_path

_SUFIX_FORMAT = "%Y_%m_%d_%H_%M_%S_%f"

def create_reports_from_configuration_folder(folder_path, dest_folder_path,
    verbose = False):
    assert os.path.isdir(dest_folder_path)

    _sufix = datetime.today().strftime(_SUFIX_FORMAT)

    # create subfolder
    new_folder_name = "family_{}".format(_sufix)
    new_folder_path = os.path.join(dest_folder_path, new_folder_name)
    os.mkdir(new_folder_path)

    TRs = []
    for file_name in os.listdir(folder_path):
        # ignore files that are not configuration files
        if not _is_valid_configuration_file(file_name):
            continue

        if verbose:
            print("\n" + "*"*20)
            print(file_name)

        file_path = os.path.join(folder_path, file_name)
        
        TRs_i = create_reports_from_configuration_file(file_path,
            new_folder_path, verbose)
        
        # get average result
        avg_TR = TRs_i[0]
        for TR in TRs_i[1:]:
            avg_TR += TR
        avg_TR *= 1/len(TRs_i)
        
        TRs.append(avg_TR)

    create_comparison_table_report(TRs, new_folder_path)

def create_reports_from_configuration_file(src_path: str,
    dest_folder_path: str = None, verbose = False):
    configs = read_configuration_file(src_path)

    if dest_folder_path is None:
        dest_folder_path = configs.dest_folder

    dataset_is_numeric = configs.dataset_is_numeric
    X, y = load_dataset_from_path(configs.dataset_path,
        dataset_is_numeric)
    dataset_name = configs.dataset_name
    
    TRs = []
    for conf in configs:
        T = get_teacher(conf.teacher_name, conf.teacher_kwargs)
        L = get_learner(conf.learner_name, conf.learner_kwargs)
        TR_i = teach(T, L, X, y, dataset_name)
        TRs.append(TR_i)

        if verbose:
            print("-"*20)
            print(TR_i.main_infos)

    if len(TRs) == 1:
        create_report(TRs[0], dest_folder_path)
    else:
        create_reports(TRs, dest_folder_path)

    return TRs

def create_reports(v, dest_folder_path: str):
    assert os.path.isdir(dest_folder_path)

    _sufix = datetime.today().strftime(_SUFIX_FORMAT)

    # create subfolder
    new_folder_name = "set_{}".format(_sufix)
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
    new_folder_name = "run_{}".format(_sufix)
    new_folder_path = os.path.join(dest_folder_path, new_folder_name)
    os.mkdir(new_folder_path)

    # create summary file
    summary_file_name = "summary_{}.txt".format(_sufix)
    summary_file_path = os.path.join(new_folder_path,
        summary_file_name)
    _convert_teach_result_to_txt(TR, summary_file_path)

    # create csv file
    teacher_log_file_name = "teacher_log_{}.csv".format(_sufix)
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

def _is_valid_configuration_file(file_name):
    return file_name.endswith("conf")