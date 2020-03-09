import machine_teacher
from sklearn.datasets import load_boston

if __name__ == "__main__":
	X, y = load_boston(return_X_y=True)
	seed = 0
	batch_relative_size = 0.1
	max_iters = 5
	L1 = machine_teacher.Learners.LinearRegressionLearner()
	T1 = machine_teacher.Teachers.RandomTeacher(seed,
		batch_relative_size, max_iters)
	res = machine_teacher.teach(T1,L1,X,y)
	print(res)