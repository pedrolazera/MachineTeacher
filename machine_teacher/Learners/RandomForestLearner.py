from .. import GenericLearner
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RandomForestLearner(GenericLearner.Learner):
	def __init__(self, *args, **kwargs):
		self.model = RandomForestClassifier(*args, **kwargs)
		super().__init__()

	def fit(self, Xi, yi):
		self.update_X(Xi)
		self.update_y(yi)
		return self.model.fit(self.X, self.y)

	def predict(self, X):
		return self.model.predict(X)

	def get_h0(self, X):
		return super()._get_h0(X)