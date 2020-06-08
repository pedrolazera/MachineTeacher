import numpy as np
from . import Experiment3Teacher
from . import DoubleTeacher
from ..Utils.Sampler import get_first_examples


class SelectDistTeacher(Experiment3Teacher):
	name = "SelectDistTeacher"
	def __init__(self,
		safity: float,
		seed: int = Experiment3Teacher._SEED,
		frac_start: float = Experiment3Teacher._FRAC_START,
		frac_time_change: float = Experiment3Teacher._FRAC_TIME_CHANGE,
		strategy: int = DoubleTeacher._STRATEGY_DOUBLE_SIZE):
		super().__init__(safity, seed, frac_start, frac_time_change, strategy)


		assert safity > -1.0


	def _increase_training_set(self, training_set_size,
		increase_amount, time_left):
		assert len(self.iters_t_fim) >= 3

		x = [self.iters_n[-3], self.iters_n[-2], self.iters_n[-1]]
		y = [(self.iters_t_fim[-i] - self.iters_t_ini[-i]) for i in range(1,4)] 
		poly = np.poly1d(np.polyfit( x, y, deg=2)) 
		increased_training_set_size = training_set_size + increase_amount
		estimated_time = poly(increased_training_set_size)*(1+self.safity)
		
		return estimated_time < time_left


	def get_new_examples(self, test_ids,
		test_labels, time_left: float):
		# marcadores para computar o tempo de treino
		if not self._keep_going():
			return np.array([])

		if self._state == self._STATE_DOUBLE_EXAMPLES:
			self.iters_t_ini.append(self.time_limit - time_left)
			return super().get_new_examples(test_ids, test_labels, time_left)
		elif self._state == self._STATE_GET_WRONG_EXAMPLES:
			assert len(test_ids) > 0, str((self.seed, self.num_iters))
			
			self._state = self._STATE_DONE

			assert self.qtd_selected_untrained_tested_examples == len(self._selected_examples)
			#return self._send_new_ids(self._selected_examples)
			new_examples = self._get_selected_ids(test_ids, test_labels)
			return self._send_new_ids(new_examples)



	def get_new_test_ids(self, test_ids, test_labels,
		time_left: float) -> np.ndarray:
		if self._state == self._STATE_DONE:
			return np.array([])

		# check if it is time to change strategy
		if (1 - time_left/self.time_limit) >= self.frac_time_change:
			if self.num_iters >= 3:
				if self._state == self._STATE_DOUBLE_EXAMPLES:
					self._state = self._STATE_CHANGE_STRATEGY

		if self._state == self._STATE_DOUBLE_EXAMPLES:
			self.iters_t_fim.append(self.time_limit - time_left)
			self.iters_n.append(self.S_current_size)
			return np.array([])

		if self._state == self._STATE_CHANGE_STRATEGY:
			assert len(test_ids) == 0
			self.iters_t_fim.append(self.time_limit - time_left)
			self.iters_n.append(self.S_current_size)
			self.qtd_selected_untrained_tested_examples = 0
			self.qtd_untrained_tested_examples = 0
			self._state = self._STATE_GET_WRONG_EXAMPLES
			#bound do numero maximo de exemplos que devem ser testados
			self._last_increase_amount = len(self.y) 
			self.classes = np.unique(self.y)
			self._selected_examples = np.array([], dtype=int)
			#print("MUDANDO DE ESTRATEGIA...")
		elif self._state == self._STATE_GET_WRONG_EXAMPLES:
			last_ids = test_ids[-self._last_increase_amount:]
			last_labels = test_labels[-self._last_increase_amount:]
			new_ids = self._get_selected_ids(last_ids, last_labels)
			self.qtd_selected_untrained_tested_examples += len(new_ids)
			self._selected_examples = np.append(self._selected_examples, new_ids)	
			assert len(self._selected_examples) == len(set(self._selected_examples))

		# check if is possible (there is time) to increase training set
		new_test_id = self.S_current_size + self.qtd_untrained_tested_examples
		training_set_size = self.S_current_size + self.qtd_selected_untrained_tested_examples
		bound_max = min(self.m - new_test_id, self._last_increase_amount)
		self._last_increase_amount = self._get_max_increase_amount(0, bound_max, 
														training_set_size, time_left)
		if bound_max > 0:
			new_test_ids = self.shuffled_ids[new_test_id: new_test_id+self._last_increase_amount] # o double teacher embaralha referências
			if self._last_increase_amount == 0:
				assert new_test_ids.size == 0
			self.qtd_untrained_tested_examples += len(new_test_ids)
			return new_test_ids 
		else:
			if self.qtd_selected_untrained_tested_examples == 0:
				self._state = self._STATE_DONE
			return np.array([])

	def _get_max_increase_amount(self, bound_min, bound_max, training_set_size, time_left):
		if bound_min == bound_max:
			return bound_max
		k = int( (bound_min+bound_max)/2 ) + 1
		if self._increase_training_set(training_set_size, k, time_left):
			return self._get_max_increase_amount(k, bound_max, training_set_size, time_left)
		else:
			return self._get_max_increase_amount(bound_min, k-1, training_set_size, time_left)

	def _get_statistics(self, correct_labels, predict_labels):
		dict_class_size = {c:0 for c in self.classes}
		dict_class_errors = {c:0 for c in self.classes}
		for i,label in enumerate(predict_labels):
			c = correct_labels[i]
			dict_class_size[c] += 1
			if label != c:
				dict_class_errors[c] += 1
		prop = 0
		for c in self.classes:			
			if dict_class_size[c] == 0: continue
			p = dict_class_errors[c]/dict_class_size[c]
			if p > prop:
				prop = p
		
		dict_class_corrects = {c:0 for c in self.classes}
		qtd_new_examples = 0
		for c in self.classes:
			corrects = int(np.round(prop*dict_class_size[c])-dict_class_errors[c])
			corrects = min(corrects, dict_class_size[c]-dict_class_errors[c])
			dict_class_corrects[c] = corrects
			qtd_new_examples += (corrects + dict_class_errors[c])

		return qtd_new_examples, dict_class_size, dict_class_errors, dict_class_corrects

	def _get_selected_ids(self, test_ids, test_labels):
		correct_labels = self.y[test_ids]
		qtd_new_examples, dict_class_size, dict_class_errors, dict_class_corrects = self._get_statistics(correct_labels, test_labels)
		cont_selecteds = 0
		new_ids = np.zeros(qtd_new_examples, dtype=int)
		for i, label in enumerate(test_labels):
			e = test_ids[i]
			c = correct_labels[i]
			if label != c:
				new_ids[cont_selecteds] = e
				cont_selecteds += 1
			elif dict_class_corrects[c] > 0:
				new_ids[cont_selecteds] = e
				cont_selecteds += 1
				dict_class_corrects[c] -= 1
			if cont_selecteds == qtd_new_examples:
				break
		assert cont_selecteds == qtd_new_examples
		return new_ids

















