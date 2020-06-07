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

def load_dataset_train_test_from_path(path_treino,
	path_teste, is_numeric = None):
	# carrega treino, aplica transformação em X e em Y
	# carrega teste, aplica as mesmas transformações em X e em Y
	raise NotImplementedError
	return (X_train, y_train, X_test, y_test)

def _get_is_numeric(dataset_name):
	_d = {
		"agaricus-lepiota.csv": False,
		"agaricus-lepiota_train.csv": False,
		"agaricus-lepiota_test.csv": False,

		"avila_tr.csv": True,
		"avila_train.csv": True,
		"avila_test.csv": True,

		"bank_marketing.csv": True,
		"car.csv": False,
		"ClaveVectors_Firm_Teacher_Model.csv": True,
		"covtype.csv": True,
		"covtype_train.csv": True,
		"covtype_test.csv": True,
		"crowdsourced.csv": True,
		"default_of_credit_card_clients.csv": True,
		"Electrical_grid_stability_simulated_data.csv": True,
		"HTRU.csv": True,
		"mnist_train.csv": True,
		"mnist_test.csv": True,

		"nursery.csv": False,
		"nursery_train.csv": False,
		"nursery_test.csv": False,

		"poker_hand_train.csv": False,
		"poker_hand_test.csv": False,

		"Sensorless_drive_diagnosis.csv": True,
		"shuttle.csv": True,
		"Skin_NonSkin.csv": True
	}

	assert dataset_name in _d, "dataset {} " + str(dataset_name) + "nao cadastrado"

	return _d[dataset_name]

def _tmp_load_dataset(path, is_numeric):
	data  = pd.read_csv(path, header = None, sep = _SEP)
	y = data[0].values

	# transforma os rótulos em inteiros a partir de zero
	le = preprocessing.LabelEncoder()
	y = le.fit_transform(y)

	# tira colunas (atributos) categóricos
	if is_numeric:
		X = data.drop(columns = [0]).values
	else:
		data = data.drop(columns = [0])
		X = pd.get_dummies(data, columns = data.columns).values
			
	return (X,y)
