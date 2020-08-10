from ..GenericLearner import Learner
import xgboost as XGBoost
import numpy as np


class XGBoostLearner(Learner):
	name = "XGBoostLearner"

	def __init__(self, *args, **kwargs):
		self.args = args
		self.kwargs = kwargs

	def start(self):
		self.model = XGBoost.XGBClassifier(*self.args, **self.kwargs)

	def fit(self, X, y):
		return self.model.fit(X, y)

	def predict(self, X):
		return self.model.predict(X)

	def get_params(self):
		return self.model.get_params()