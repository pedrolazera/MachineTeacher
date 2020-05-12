import numpy as np
import pandas as pd
from sklearn import preprocessing
import os

_SEP = ','

def load_dataset_from_path(path, is_numeric = None):
	if is_numeric is None:
		dataset_name = os.path.basename(path)
		is_numeric = _get_is_numeric(dataset_name)

	return _tmp_load_dataset(path, is_numeric)

def _get_is_numeric(dataset_name):
	_d = {
		"agaricus-lepiota.csv": False,
		"avila_tr.csv": True,
		"bank_marketing.csv": True,
		"car.csv": False,
		"ClaveVectors_Firm_Teacher_Model.csv": True,
		"covtype.csv": True,
		"crowdsourced.csv": True,
		"default_of_credit_card_clients.csv": True,
		"Electrical_grid_stability_simulated_data.csv": True,
		"HTRU.csv": True,
		"mnist_train.csv": True,
		"nursery.csv": False,
		"poker_hand_train.csv": False,
		"Sensorless_drive_diagnosis.csv": True,
		"shuttle.csv": True,
		"Skin_NonSkin.csv": True
	}

	assert dataset_name in _d, "dataset {} " + str(dataset_name) + "nao cadastrado"

	return _d[dataset_name]

def _tmp_load_dataset(path, is_numeric):
	data  = pd.read_csv(path, header = None, sep = _SEP)
	y = data[0].values
	le = preprocessing.LabelEncoder()		
	y = le.fit_transform(y)

	if is_numeric:
		X = data.drop(columns = [0]).values
	else:
		data = data.drop(columns = [0])
		X = pd.get_dummies(data, columns = data.columns).values
			
	return (X,y)
