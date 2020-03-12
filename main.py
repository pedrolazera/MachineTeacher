import machine_teacher
from sklearn.datasets import load_boston # regression
from sklearn.datasets import load_iris # classification
from sklearn.datasets import load_diabetes # regression
from sklearn.datasets import load_digits # classification
from sklearn.datasets import load_linnerud # multivariate regression
from sklearn.datasets import load_wine # classification
from sklearn.datasets import load_breast_cancer # classification

def _get_cars_dataset():
	from sklearn import preprocessing
	import pandas as pd
	import os

	path = os.path.dirname(__file__)
	path = os.path.join(path, "garagem", "datasets", "car.csv")
	path = os.path.abspath(path)
	data = pd.read_csv(path, header=None, sep=',')
	
	y = preprocessing.LabelEncoder().fit_transform(data[0].values)
	
	data = data.drop(columns=[0])
	X = pd.get_dummies(data, columns=data.columns).values

	return (X, y)

if __name__ == "__main__":
	#X, y = load_breast_cancer(return_X_y=True)
	X, y = _get_cars_dataset()
	seed = 0
	batch_relative_size = 0.005
	frac_start = 0.01
	frac_stop = 1.0
	L1 = machine_teacher.Learners.RandomForestLearner(random_state=0)
	T1 = machine_teacher.Teachers.WTFTeacher(frac_start, frac_stop, seed)
	res = machine_teacher.teach(T1,L1,X,y)
	print(res)
	for row in res.teacher_log:
		print(row)

	_get_cars_dataset()