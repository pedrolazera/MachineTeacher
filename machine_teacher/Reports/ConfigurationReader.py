import configparser
import json

_SECTIONS = ('teacher', 'learner', 'dataset', 'destination')

class TestConfiguration:
	def __init__(self, teacher_name, learner_name, dataset_path,
		teacher_kwargs, learner_kwargs, dest_folder):
		self.teacher_name = teacher_name
		self.learner_name = learner_name
		self.dataset_path = dataset_path
		self.teacher_kwargs = teacher_kwargs
		self.learner_kwargs = learner_kwargs
		self.dest_folder = dest_folder

def read_configuration_file(path: str):
    config = configparser.ConfigParser()
    config.read(path)
    #return config

    #_sections_to_lowercase(config)

    # some asserts - sections most exist
    for section_name in _SECTIONS:
    	assert config.has_section(section_name)

    teacher_name, teacher_kwargs = _parse_teacher_section(config['teacher'])
    learner_name, learner_kwargs = _parse_learner_section(config['learner'])
    dataset_path = config['dataset']['path'].strip()
    dest_folder = config['destination']['path'].strip()

    return TestConfiguration(
    	teacher_name, learner_name, dataset_path,
		teacher_kwargs, learner_kwargs, dest_folder
		)

def _sections_to_lowercase(config):
	lst_sections = list(config.sections())
	for section_name in lst_sections:
		config[section_name.lower()] = config[section_name]

def _parse_teacher_section(section):
	kwargs = {key: json.loads(value) for (key,value) in dict(section).items()}
	assert 'name' in kwargs

	name = kwargs['name']
	del kwargs['name']

	return (name, kwargs)

def _parse_learner_section(section):
	learner_name, learner_kwargs = _parse_teacher_section(section)
	return (learner_name, learner_kwargs)