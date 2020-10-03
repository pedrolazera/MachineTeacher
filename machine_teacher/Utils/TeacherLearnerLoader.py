"""
This modules just loads Teachers and Learners from
a string name. It's used by the Report module, to load
objects from theirs names on configurations files
"""

from .. import Learners
from .. import Teachers

_D_LEARNERS = {
	Learners.LinearRegressionLearner.name: Learners.LinearRegressionLearner,
	Learners.LogisticRegressionLearner.name: Learners.LogisticRegressionLearner,
	Learners.RandomForestLearner.name: Learners.RandomForestLearner,
	Learners.SVMLinearLearner.name: Learners.SVMLinearLearner,
	Learners.SVMLearner.name: Learners.SVMLearner,
	Learners.LGBMLearner.name: Learners.LGBMLearner,
	Learners.XGBoostLearner.name: Learners.XGBoostLearner
}

_D_TEACHERS = {
	Teachers.RandomTeacher.name: Teachers.RandomTeacher,
	Teachers.PacTeacher.name: Teachers.PacTeacher,
	Teachers.WTFTeacher.name: Teachers.WTFTeacher,
	Teachers.SingleBatchTeacher.name: Teachers.SingleBatchTeacher,
	Teachers.DoubleTeacher.name: Teachers.DoubleTeacher,
	Teachers.Experiment3Teacher.name: Teachers.Experiment3Teacher,
	Teachers.SelectDistTeacher.name: Teachers.SelectDistTeacher,
	Teachers.FixedPercWrongTeacher.name: Teachers.FixedPercWrongTeacher,
	Teachers.DynamicPercWrongTeacher.name: Teachers.DynamicPercWrongTeacher
}

def get_teacher(teacher_name, args):
	return _D_TEACHERS[teacher_name](**args)

def get_learner(learner_name, args):
	return _D_LEARNERS[learner_name](**args)