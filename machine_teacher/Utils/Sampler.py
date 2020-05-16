import numpy as np

def get_first_examples(prop, m, classes, y, shuffle_function):
	"""
	Seleciona uma amostra de prop*m exemplos, respeitando as restrições
	(1) tem de haver pelo menos um exemplo de cada classe
	(2) a distribuição de classes da amostra é igual à proporção de classes
	do dataset, módulo arredondamentos
	"""
	new_ids = []
	n_samples = prop*m
	class_distribution = [0 for c in classes]
	
	for c in y:
		class_distribution[c] += 1
	
	#class_samples = [np.ceil((c/m)*n_samples) for c in class_distribution]
	class_samples = _get_class_samples(n_samples, m, class_distribution)
	n_samples = np.sum(class_samples)
	
	#classes_number = len(classes)
	#v_cont = [0 for x in range(classes_number)]
	
	v_cont = [0] * len(classes)
	aux = [i for i in range(m)]
	shuffle_function(aux)
	
	cont = 0
	i = 0
	while (cont < n_samples):
		id_i = aux[i]
		class_i = y[id_i]
		if v_cont[class_i] < class_samples[class_i]:
			new_ids.append(id_i)
			cont += 1
			v_cont[class_i] += 1
		i+=1

	return new_ids

def _get_class_samples(n_samples, m, class_distribution):
	class_samples = [0]*len(class_distribution)

	for (i, tot_class_i) in enumerate(class_distribution):
		qtd_class_i = np.ceil(tot_class_i/m * n_samples)
		qtd_class_i = min(qtd_class_i, tot_class_i)
		class_samples[i] = qtd_class_i

	return class_samples

def choose_ids(population, weights, n):
	# cria elemento artificial para complementar probabilidade
	weights_2 = np.append(weights, 1.0 - np.sum(weights))
	population_2 = np.append(population, len(population))
	
	# faz selecao com repeticao
	new_ids = np.random.choice(population_2, n,
		replace = True, p = weights_2)

	# retira elemento artificial
	new_ids = np.unique(new_ids)
	new_ids = [i for i in new_ids if i != len(population)]
	new_ids = np.array(new_ids)
	
	return new_ids