import numpy as np
from ..GenericTeacher import Teacher
from ..Utils.Sampler import get_first_examples
from sklearn import preprocessing
import warnings

class DynamicPercWrongTeacher(Teacher):
	"""docstring for DynamicPercWrongTeacher"""
	name = "DynamicPercWrongTeacher"
	_SEED = 0
	_FRAC_START = 0.05
	_FRAC_BATCH_SEARCH = 0.05

	_STATE_DONE = False
	_STATE_SEND_NEW_IDS = 0
	_STATE_SEND_EMPTY_NEW_IDS = 1
	_FRAC_SAMPLE = 0.2 #sample size sent to the student to analyze the accuracy

	def __init__(self, seed: int = _SEED,
		frac_start: float = _FRAC_START,
		frac_sample = _FRAC_SAMPLE,
		frac_batch_search = _FRAC_BATCH_SEARCH):
		self.seed = seed
		self.frac_start = frac_start
		self.frac_sample = frac_sample
		self.frac_batch_search = frac_batch_search

		

	def start(self, X, y, time_left: float):
		self._start(X, y, time_left)
		self.num_iters = 0
		self.m = y.size
		self.S_current_size = 0
		self.state_new_ids = self._STATE_SEND_NEW_IDS
		self._STATE_DONE = False
		self.sample_size = int(self.m * self.frac_sample)

		
		self.shuffled_ids = self._get_shuffled_ids()

		assert len(self.shuffled_ids) == len(self.ids)

	def _get_shuffled_ids(self):
		ids = np.arange(self.m, dtype=int)
		f_shuffle = np.random.RandomState(self.seed).shuffle
		f_shuffle(ids)
		return ids

	def _keep_going(self):
		return (self.S_current_size ) < self.m

	def get_first_examples(self, time_left: float):
		classes = np.unique(self.y) # isso devia sair. devia ser computado for get_first_examples
		f_shuffle = np.random.RandomState(self.seed).shuffle
		new_ids = get_first_examples(self.frac_start, self.m, classes, self.y, f_shuffle)
		new_ids = np.array(new_ids)
		
		# update shuffled_ids. Aqui usamos ponteiros de ponteiros.
		# debugue as próximas duas linhas de cabeça vazia
		_new_ids = set(new_ids)
		self.shuffled_ids = np.append(new_ids,
							          [i for i in self.shuffled_ids if i not in _new_ids])
		
		return self._send_new_ids(new_ids)

	def get_new_examples(self, test_ids, test_labels, time_left: float):
		if not self._keep_going() or len(test_ids)==0:
			return np.array([])

		self.batch_size = self.sample_size - self.S_current_size

		correct_test_labels = self.y[test_ids]
		wrong_ids = test_ids[test_labels != correct_test_labels]
		error = (len(wrong_ids)/len(test_ids))
		
		if np.isclose(error, 0): 			
			return np.array([])

		_start = self.S_current_size + len(test_ids)
		_end = _start + max(0, self.batch_size - len(wrong_ids))
		random_ids = self.shuffled_ids[_start:_end]
		new_ids = np.append(random_ids, wrong_ids)

		### TESTANDO SE FAZ DIFERENCAO COLOCAR ERRADOS ESPALHADOS
		#f_shuffle = np.random.RandomState(self.seed).shuffle
		#f_shuffle(new_ids)
		# ---------------------------------------------------
		return self._send_new_ids(new_ids)


	def get_new_test_ids(self, test_ids,
		test_labels, time_left: float) -> np.ndarray:
		if not self._keep_going():
			return np.array([])

		if self.state_new_ids == self._STATE_SEND_EMPTY_NEW_IDS:
			return np.array([])					

		_start = self.S_current_size
		_end = _start + int(self.frac_batch_search*self.m)
		
		new_ids = self.shuffled_ids[_start:_end]
		assert type(new_ids) == type(np.array([]))
		self.state_new_ids = self._STATE_SEND_EMPTY_NEW_IDS
		
		return new_ids
		
		

	def get_log_header(self):
		return ["iter_number", "training_set_size", "accuracy"]

	def get_log_line(self, h):
		accuracy = 1 - self._get_wrong_labels_id(h).size/self.y.size
		log_line = [self.num_iters, self.S_current_size, accuracy]
		return log_line

	def _send_new_ids(self, new_ids):
		self.num_iters += 1
		self.S_current_size += len(new_ids)
		return new_ids

	def _get_shuffled_ids(self):
		ids = np.arange(self.m, dtype=int)
		f_shuffle = np.random.RandomState(self.seed).shuffle
		f_shuffle(ids)
		return ids

	
	def get_params(self) -> dict:
		return {
			"seed": self.seed,
			"frac_start": self.frac_start
			}




################################################################################################
# COM TEMPO
##############################################################################################
# import numpy as np
# from ..GenericTeacher import Teacher
# from ..Utils.Sampler import get_first_examples
# from sklearn import preprocessing
# import warnings

# class DynamicPercWrongTeacher(Teacher):
# 	"""docstring for DynamicPercWrongTeacher"""
# 	name = "DynamicPercWrongTeacher"
# 	_SEED = 0
# 	_FRAC_START = 0.01
# 	_STRATEGY_DOUBLE_INCREMENT = 0
# 	_STRATEGY_DOUBLE_SIZE = 1
# 	_FRAC_WRONG_INCREMENT = 0.1

# 	_STATE_DONE = False
# 	_STATE_SEND_NEW_IDS = 0
# 	_STATE_SEND_EMPTY_NEW_IDS = 1
# 	_SAMPLE_SIZE = 300 #sample size sent to the student to analyze the accuracy

# 	def __init__(self, seed: int = _SEED,
# 		frac_start: float = _FRAC_START,
# 		frac_wrong_increment = _FRAC_WRONG_INCREMENT,
# 		sample_size = _SAMPLE_SIZE,
# 		strategy: int = _STRATEGY_DOUBLE_INCREMENT):
# 		self.seed = seed
# 		self.frac_start = frac_start
# 		self.strategy = strategy
# 		self.frac_wrong_increment = frac_wrong_increment
# 		self.sample_size = sample_size
		

# 	def start(self, X, y, time_left: float):
# 		self._start(X, y, time_left)
# 		self.num_iters = 0
# 		self.m = y.size
# 		self.S_current_size = 0
# 		self.batch_size = self.sample_size
# 		self.qtd_untrained_tested_examples = 0
# 		self.state_new_ids = self._STATE_SEND_NEW_IDS
# 		self._STATE_DONE = False
		
# 		self.time_limit = time_left
# 		self.iters_t_ini = []
# 		self.iters_t_fim = []
# 		self.iters_n = []
# 		self.safity = .0

# 		self.shuffled_ids = self._get_shuffled_ids()

# 		assert len(self.shuffled_ids) == len(self.ids)

# 	def _get_shuffled_ids(self):
# 		ids = np.arange(self.m, dtype=int)
# 		f_shuffle = np.random.RandomState(self.seed).shuffle
# 		f_shuffle(ids)
# 		return ids

# 	def _keep_going(self):
# 		return (self.S_current_size + self.qtd_untrained_tested_examples) < self.m

# 	def get_first_examples(self, time_left: float):
# 		self.iters_t_ini.append(self.time_limit - time_left)
# 		classes = np.unique(self.y) # isso devia sair. devia ser computado for get_first_examples
# 		f_shuffle = np.random.RandomState(self.seed).shuffle
# 		new_ids = get_first_examples(self.frac_start, self.m, classes, self.y, f_shuffle)
# 		new_ids = np.array(new_ids)
		
# 		# update shuffled_ids. Aqui usamos ponteiros de ponteiros.
# 		# debugue as próximas duas linhas de cabeça vazia
# 		_new_ids = set(new_ids)
# 		self.shuffled_ids = np.append(new_ids,
# 							          [i for i in self.shuffled_ids if i not in _new_ids])
		
# 		# update batch size, from 1 to len(new_ids), based on strategy
# 		if self.strategy == self._STRATEGY_DOUBLE_SIZE:
# 			self.batch_size = len(new_ids)
# 			if self.batch_size < self.sample_size:
# 				raise ValueError("batch_size ("+str(self.batch_size) +
# 					             ") deve ser maior ou igual sample_size ("+ 
# 					             str(self.sample_size)+")!") 

# 		elif self.strategy == self._STRATEGY_DOUBLE_INCREMENT:
# 			self.batch_size = self.sample_size
# 		else:
# 			raise ValueError("Estrategia desconhecida: " + str(self.strategy))

# 		return self._send_new_ids(new_ids)

# 	def get_new_examples(self, test_ids, test_labels, time_left: float):
# 		if not self._keep_going():
# 			return np.array([])

# 		if self._STATE_DONE:
# 			return self._send_new_ids(test_ids)

# 		self.iters_t_ini.append(self.time_limit - time_left)

# 		# Caso em que test_ids tem 2*batch_size elementos
# 		correct_test_labels = self.y[test_ids[:self.batch_size]]
# 		wrong_ids = test_ids[:self.batch_size][test_labels[:self.batch_size] != correct_test_labels]
# 		error = (len(wrong_ids)/len(test_ids))
		
# 		assert (error > 0 or len(wrong_labels) == 0)
# 		assert (error < 1 or len(wrong_labels) == len(test_ids))

# 		if np.isclose(error, 0): 			
# 			return np.array([])

# 		random_ids = test_ids[self.batch_size+len(wrong_ids):]
# 		new_ids = np.append(random_ids, wrong_ids)

# 		### TESTANDO SE FAZ DIFERENCAO COLOCAR ERRADOS ESPALHADOS
# 		#f_shuffle = np.random.RandomState(self.seed).shuffle
# 		#f_shuffle(new_ids)
# 		# ---------------------------------------------------

# 		self.qtd_untrained_tested_examples += (len(test_ids)-len(new_ids))
# 		self.batch_size *= 2
# 		self.state_new_ids = self._STATE_SEND_NEW_IDS
		
# 		return self._send_new_ids(new_ids)


# 	def get_new_test_ids(self, test_ids,
# 		test_labels, time_left: float) -> np.ndarray:
# 		if not self._keep_going():
# 			return np.array([])

# 		if self.state_new_ids == self._STATE_SEND_EMPTY_NEW_IDS:
# 			return np.array([])					

# 		self.iters_t_fim.append(self.time_limit - time_left)
# 		self.iters_n.append(self.S_current_size)

# 		_start = self.S_current_size + self.qtd_untrained_tested_examples
# 		_end = _start + 2*self.batch_size
# 		if (1 - time_left/self.time_limit) >= 1/4.0 or (_end > self.m):
# 			bound_max = self.m - self.S_current_size - self.qtd_untrained_tested_examples
# 			increase_amount = self._get_max_increase_amount(0, bound_max, 
# 															self.S_current_size, time_left)
# 			if increase_amount == 0:
# 				return np.array([])
# 			_end = min(_start + 2*self.batch_size, self.m)
# 			self._STATE_DONE = True

		
# 		new_ids = self.shuffled_ids[_start:_end]
# 		assert type(new_ids) == type(np.array([]))
# 		self.state_new_ids = self._STATE_SEND_EMPTY_NEW_IDS
		
# 		return new_ids
		
		

# 	def get_log_header(self):
# 		return ["iter_number", "training_set_size", "accuracy"]

# 	def get_log_line(self, h):
# 		accuracy = 1 - self._get_wrong_labels_id(h).size/self.y.size
# 		log_line = [self.num_iters, self.S_current_size, accuracy]
# 		return log_line

# 	def _send_new_ids(self, new_ids):
# 		self.num_iters += 1
# 		self.S_current_size += len(new_ids)
# 		return new_ids

# 	def _get_shuffled_ids(self):
# 		ids = np.arange(self.m, dtype=int)
# 		f_shuffle = np.random.RandomState(self.seed).shuffle
# 		f_shuffle(ids)
# 		return ids

# 	def _increase_training_set(self, training_set_size,
# 		increase_amount, time_left):
# 		assert len(self.iters_t_fim) >= 3

# 		x = [self.iters_n[-3], self.iters_n[-2], self.iters_n[-1]]
# 		y = [(self.iters_t_fim[-i] - self.iters_t_ini[-i]) for i in range(1,4)] 
# 		poly = np.poly1d(np.polyfit( x, y, deg=2)) 
# 		increased_training_set_size = training_set_size + increase_amount
# 		estimated_time = poly(increased_training_set_size)*(1+self.safity)
		
# 		return estimated_time < time_left

# 	def _get_max_increase_amount(self, bound_min, bound_max, training_set_size, time_left):
# 		if bound_min == bound_max:
# 			return bound_max
# 		k = int( (bound_min+bound_max)/2 ) + 1
# 		if self._increase_training_set(training_set_size, k, time_left):
# 			return self._get_max_increase_amount(k, bound_max, training_set_size, time_left)
# 		else:
# 			return self._get_max_increase_amount(bound_min, k-1, training_set_size, time_left)

# 	def get_params(self) -> dict:
# 		return {
# 			"seed": self.seed,
# 			"frac_start": self.frac_start,
# 			"strategy": self.strategy,
# 			}


########################################################
#SEM TEMPO
########################################################

# import numpy as np
# from ..GenericTeacher import Teacher
# from ..Utils.Sampler import get_first_examples
# from sklearn import preprocessing
# import warnings

# class DynamicPercWrongTeacher(Teacher):
# 	"""docstring for DynamicPercWrongTeacher"""
# 	name = "DynamicPercWrongTeacher"
# 	_SEED = 0
# 	_FRAC_START = 0.01
# 	_STRATEGY_DOUBLE_INCREMENT = 0
# 	_STRATEGY_DOUBLE_SIZE = 1
# 	_FRAC_WRONG_INCREMENT = 0.1
# 	_STATE_DONE = False

# 	_STATE_SEND_NEW_IDS = 0
# 	_STATE_CHOOSE_BATCH_SIZE_NEW_IDS = 1
# 	_STATE_SEND_EMPTY_NEW_IDS = 2
# 	_SAMPLE_SIZE = 300 #sample size sent to the student to analyze the accuracy

# 	def __init__(self, seed: int = _SEED,
# 		frac_start: float = _FRAC_START,
# 		frac_wrong_increment = _FRAC_WRONG_INCREMENT,
# 		sample_size = _SAMPLE_SIZE,
# 		strategy: int = _STRATEGY_DOUBLE_INCREMENT):
# 		self.seed = seed
# 		self.frac_start = frac_start
# 		self.strategy = strategy
# 		self.frac_wrong_increment = frac_wrong_increment
# 		self.sample_size = sample_size
		

# 	def start(self, X, y, time_left: float):
# 		self._start(X, y, time_left)
# 		self.num_iters = 0
# 		self.m = y.size
# 		self.S_current_size = 0
# 		self.batch_size = self.sample_size
# 		self.qtd_untrained_tested_examples = 0
# 		self.state_new_ids = self._STATE_SEND_NEW_IDS
# 		self._STATE_DONE = False
		
# 		self.shuffled_ids = self._get_shuffled_ids()

# 		assert len(self.shuffled_ids) == len(self.ids)

# 	def _get_shuffled_ids(self):
# 		ids = np.arange(self.m, dtype=int)
# 		f_shuffle = np.random.RandomState(self.seed).shuffle
# 		f_shuffle(ids)
# 		return ids

# 	def _keep_going(self):
# 		return (self.S_current_size + self.qtd_untrained_tested_examples) < self.m

# 	def get_first_examples(self, time_left: float):
# 		classes = np.unique(self.y) # isso devia sair. devia ser computado for get_first_examples
# 		f_shuffle = np.random.RandomState(self.seed).shuffle
# 		new_ids = get_first_examples(self.frac_start, self.m, classes, self.y, f_shuffle)
# 		new_ids = np.array(new_ids)
		
# 		# update shuffled_ids. Aqui usamos ponteiros de ponteiros.
# 		# debugue as próximas duas linhas de cabeça vazia
# 		_new_ids = set(new_ids)
# 		self.shuffled_ids = np.append(new_ids,
# 							          [i for i in self.shuffled_ids if i not in _new_ids])
		
# 		# update batch size, from 1 to len(new_ids), based on strategy
# 		if self.strategy == self._STRATEGY_DOUBLE_SIZE:
# 			self.batch_size = len(new_ids)
# 			if self.batch_size < self.sample_size:
# 				raise ValueError("batch_size ("+str(self.batch_size) +
# 					             ") deve ser maior ou igual sample_size ("+ 
# 					             str(self.sample_size)+")!") 

# 		elif self.strategy == self._STRATEGY_DOUBLE_INCREMENT:
# 			self.batch_size = self.sample_size
# 		else:
# 			raise ValueError("Estrategia desconhecida: " + str(self.strategy))

# 		return self._send_new_ids(new_ids)

# 	def get_new_examples(self, test_ids, test_labels, time_left: float):
# 		if not self._keep_going() or self._STATE_DONE:
# 			return np.array([])

# 		random_size = np.ceil(self.batch_size*(1-self.frac_wrong_increment))
# 		random_size = int(random_size)		
# 		correct_test_labels = self.y[test_ids[random_size:]]
# 		wrong_ids = test_ids[random_size:] [test_labels[random_size:] != correct_test_labels]
# 		new_ids = np.append(test_ids[:random_size], wrong_ids)

# 		### TESTANDO SE FAZ DIFERENCAO COLOCAR ERRADOS ESPALHADOS
# 		#f_shuffle = np.random.RandomState(self.seed).shuffle
# 		#f_shuffle(new_ids)
# 		# ---------------------------------------------------

# 		self.qtd_untrained_tested_examples += (len(test_ids)-len(new_ids))
# 		self.batch_size *= 2
# 		self.state_new_ids = self._STATE_SEND_NEW_IDS

# 		return self._send_new_ids(new_ids)


# 	def get_new_test_ids(self, test_ids,
# 		test_labels, time_left: float) -> np.ndarray:
# 		if not self._keep_going():
# 			return np.array([])
		
# 		if self.state_new_ids == self._STATE_SEND_NEW_IDS:
# 			_start = self.S_current_size + self.qtd_untrained_tested_examples
# 			_end = min(_start + self.sample_size, self.m)
# 			new_ids = self.shuffled_ids[_start:_end]
# 			self.state_new_ids = self._STATE_CHOOSE_BATCH_SIZE_NEW_IDS
# 			assert type(new_ids) == type(np.array([]))
# 			return new_ids
		
# 		if self.state_new_ids == self._STATE_CHOOSE_BATCH_SIZE_NEW_IDS:
# 			correct_test_labels = self.y[test_ids]
# 			wrong_labels = test_ids[test_labels != correct_test_labels]
# 			error = (len(wrong_labels)/len(test_ids))
			
# 			assert (error > 0 or len(wrong_labels) == 0)
# 			assert (error < 1 or len(wrong_labels) == len(test_ids))

# 			if np.isclose(error, 0): 
# 				self._STATE_DONE = True
# 				return np.array([])

# 			#self.frac_wrong_increment = min(error, 0.1)

# 			increment = int((self.frac_wrong_increment*self.batch_size)/error)
# 			increment -= int(self.frac_wrong_increment*self.batch_size)
# 			increment = max(increment, 0)
# 			_start = (self.S_current_size + 
# 					  self.qtd_untrained_tested_examples + 
# 					  self.sample_size)
# 			_end = (self.S_current_size +
# 					self.qtd_untrained_tested_examples + 
# 					self.batch_size +
# 					increment)

# 			_end = min(_end , self.m)
# 			new_ids = self.shuffled_ids[_start:_end]
# 			self.state_new_ids = self._STATE_SEND_EMPTY_NEW_IDS
# 			print(error, self.frac_wrong_increment, self.batch_size, len(new_ids))
# 			return new_ids

# 		return np.array([])

# 	def get_log_header(self):
# 		return ["iter_number", "training_set_size", "accuracy"]

# 	def get_log_line(self, h):
# 		accuracy = 1 - self._get_wrong_labels_id(h).size/self.y.size
# 		log_line = [self.num_iters, self.S_current_size, accuracy]
# 		return log_line

# 	def _send_new_ids(self, new_ids):
# 		self.num_iters += 1
# 		self.S_current_size += len(new_ids)
# 		return new_ids

# 	def _get_shuffled_ids(self):
# 		ids = np.arange(self.m, dtype=int)
# 		f_shuffle = np.random.RandomState(self.seed).shuffle
# 		f_shuffle(ids)
# 		return ids



# 	def get_params(self) -> dict:
# 		return {
# 			"seed": self.seed,
# 			"frac_start": self.frac_start,
# 			"strategy": self.strategy,
# 			}
