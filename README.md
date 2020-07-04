# MachineTeacher

O programa é um pacote em Python que implementa o paradigma de Machine Learning do meu projeto de pesquisa. No projeto, estamos estudando Machine Teaching, em que procuramos entender como um Teacher pode ensinar um conceito-alvo a um Learner.

A finalidade do programa é possibilitar e padronizar (i) a interação entre Teacher e Learner, (ii) a criação de Teachers, (iii) a criaçãao de Learners e (iv) a criação de relatórios que demonstrem os resultados dessa interação para um determinado conjunto de treino.

## Documentação

A documentação do pacote está no arquivo [MachineTeacher_Documentacao.pdf](https://github.com/pedrolazera/MachineTeacher/blob/master/MachineTeacher_Documentacao.pdf), na raiz do projeto.

## Instalação

Ainda não é possível baixar o pacote pelo pip ou instalar o pacote. Em vez disso, faça o download do projeto e coloque a pasta raiz em algum diretório. Vou assumir que o local é /caminho/MachineTeacher.

## Dependências

As dependências do pacote estão listas em \\ \textit{/MachineTeacher/machine\_teacher/requirements.txt}. Para instalá-las, basta abrir um terminal e rodar o comando:

```bash
pip3 install -r caminho/MachineTeacher/machine\_teacher/requirements.txt
```

## Alguns exemplos

### Adicione o pacote ao seu path e importe o pacote

```python
import sys
sys.path.append("/caminho/MachineTeacher")

import machine_teacher
```

### Importe um dos teachers prontos

```python
T1 = machine_teacher.Teachers.DoubleTeacher()
```

### Importe um dos learners prontos ou crie um a partir de um modelos do sklearn

```python
L1 = machine_teacher.Learners.SVMLinearLearner()


from sklearn.svm import LinearSVC
class MyLearner(machine_teacher.GenericLearner):
	name = "MyLearner"

	def start(self):
		self.model = LinearSVC()

L2 = MyLearner()
```

### Rode um experimento com limite de tempo

```python
L1 = machine_teacher.Learners.SVMLinearLearner()
T1 = machine_teacher.Teachers.DoubleTeacher()
time_limit = 10.0

result = machine_teacher.protocol(T1, L1, X, y, time_limit)
```