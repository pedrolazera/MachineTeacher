from .. import Learners
from .. import Teachers

_D_LEARNERS = {
	Learners.LinearRegressionLearner.name: Learners.LinearRegressionLearner,
	Learners.LogisticRegressionLearner.name: Learners.LogisticRegressionLearner,
	Learners.RandomForestLearner.name: Learners.RandomForestLearner,
	Learners.SVMLinearLearner.name: Learners.SVMLinearLearner,
	Learners.SVMLearner.name: Learners.SVMLearner,
}

_D_TEACHERS = {
	Teachers.RandomTeacher.name: Teachers.RandomTeacher,
	Teachers.PacTeacher.name: Teachers.PacTeacher,
	Teachers.WTFTeacher.name: Teachers.WTFTeacher,
	Teachers.SingleBatchTeacher.name: Teachers.SingleBatchTeacher,
	Teachers.DoubleTeacher.name: Teachers.DoubleTeacher,
	Teachers.Experiment3Teacher.name: Teachers.Experiment3Teacher,
}

def get_teacher(teacher_name, args):
	return _D_TEACHERS[teacher_name](**args)

def get_learner(learner_name, args):
	return _D_LEARNERS[learner_name](**args)