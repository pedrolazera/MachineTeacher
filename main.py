import machine_teacher
from sklearn.datasets import load_boston

if __name__ == "__main__":
	X, y = load_boston(return_X_y=True)
	L1 = machine_teacher.Learners.LinearRegressionLearner()
	T1 = machine_teacher.Teachers.RandomTeacher(1,10)
	res = machine_teacher.teach(T1,L1,X,y)
	print(res)