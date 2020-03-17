import numpy as np
import pandas as pd
from sklearn import preprocessing

def load_dataset_from_path(path, is_numeric):
	return _tmp_load_dataset(path)

def _tmp_load_dataset(path, is_numeric=False):
	data  = pd.read_csv(path, header=None, sep=',')
	y = data[0].values
	le = preprocessing.LabelEncoder()		
	y = le.fit_transform(y)

	if is_numeric:	
		X = data.drop(columns=[0]).values
	else:
		data = data.drop(columns=[0])
		X = pd.get_dummies(data, columns=data.columns).values
			
	return (X,y)
		