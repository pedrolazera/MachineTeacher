import numpy as np

# type annotation
InputSpace = np.ndarray
Labels = np.ndarray

# numpy axis ids
_ROW_AXIS = 0
_COL_AXIS = 1

def get_qtd_rows(v) -> int:
	return v.shape[_ROW_AXIS]

def get_qtd_columns(v) -> int:
	return v.shape[_COL_AXIS]

# Input Space funtions

def join_input_spaces(X1: InputSpace, X2: InputSpace):
	return np.vstack((X1, X2))

def wrapp_input_space(X: InputSpace):
	return np.array(X)

#def get_empty_input_space(X: InputSpace):
#	ncols = X.shape[_COL_AXIS]
#	return np.array([], dtype=X.dtype).reshape(0, ncols)

# Labels functions

def join_labels(y1: Labels, y2: Labels):
	if y1.size == 0:
		return y2
	elif y2.size == 0:
		return y1

	return np.concatenate((y1, y2), axis=_ROW_AXIS)

def wrapp_labels(y: Labels):
	return np.array(y).reshape(-1)

#def get_empty_labels(y: Labels):
#	return np.array([], dtype=y.dtype)