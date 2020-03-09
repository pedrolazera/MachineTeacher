def get_first_examples(prop, m, classes, y, shuffle_function):
	new_ids = []
	n_samples = prop*m
	class_distribution = [0 for c in classes]
	for c in y:
		class_distribution[c] += 1
	class_samples = [np.ceil((c/m)*n_samples) for c in class_distribution]
	n_samples = np.sum(class_samples)
	cont = 0
	classes_number = len(classes)
	v_cont = [0 for x in range(classes_number)]
	aux = [i for i in range(m)]
	shuffle_function(aux)
	i = 0
	while (cont < n_samples):
		if v_cont[y[aux[i]]] < class_samples[y[aux[i]]]:
			new_ids.append(aux[i])
			cont+=1
			v_cont[y[aux[i]]] += 1
		i+=1

	return new_ids