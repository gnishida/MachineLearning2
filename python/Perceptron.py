import numpy as np
import math


# Example
class Example:
	def __init__(self, row, label, weight):
		self.row = row
		self.label = label
		self.weight = weight

# Create a feature vector for a given example based on the feature representation
#  vars   feature representation
#  row    example
#  return feature vector
def _x(vars, row):
	x = np.zeros(len(vars) + 1)

	for i in xrange(len(vars)):
		bit = 1
		for var in vars[i]:
			if row[var[0]] != var[1]:
				bit = 0
		x[i] = bit

	x[len(vars)] = 1

	return x

# Report the accuracy, precision, recall, and F1 for given examples and weight vector
def report(examples, w, vars):
	correct = 0
	incorrect = 0
	true_pos = 0
	false_pos = 0
	false_neg = 0

	for i in xrange(len(examples)):
		x = _x(vars, examples[i].row)

		if examples[i].label == "+": y = 1
		else: y = -1

		h = np.sign(np.dot(w, x))
		if h == y:
			correct += 1
			if y == 1:
				true_pos += 1
		else:
			incorrect += 1
			if y == 1:
				false_neg += 1
			else:
				false_pos += 1

	accuracy = float(correct) / float(correct + incorrect)
	precision = float(true_pos) / float(true_pos + false_pos)
	recall = float(true_pos) / float(true_pos + false_neg)
	F1 = 2 * precision * recall / (precision + recall)

	print("accuracy: " + str(accuracy) + " / precision: " + str(precision) + " / recall: " + str(recall) + " / F1: " + str(F1))
	return [accuracy, precision, recall, F1]

# Perceptron
def Perceptron(maxIterations, featureSet):
	attr_types = {0: "B", 1: "C", 2: "C", 3: "B", 4: "B", 5: "B", 6: "B", 7: "C", 8: "B", 9: "B", 10: "C", 11: "B", 12: "B", 13: "C", 14: "C"}

	examples = readData("train.txt")

	# get all the possible values for each attribute
	attr_values = {}
	for attr_index, attr_type in attr_types.iteritems():
		if attr_type == "C": continue

		print(attr_index)

		attr_values[attr_index] = []

		for example in examples:
			if example.row[attr_index] == "?": continue
			if example.row[attr_index] not in attr_values[attr_index]:
				attr_values[attr_index].append(example.row[attr_index])

		for val in attr_values[attr_index]:
			print("  " + val)

	# setup feature representation
	vars = []
	if featureSet == 1 or featureSet == 3:
		for attr_index, attr_type in attr_types.iteritems():
			if attr_type == "C": continue

			for val in attr_values[attr_index]:
				vars.append([[attr_index, val]])


	if featureSet == 2 or featureSet == 3:
		attr_indices = attr_types.keys()
		for i in xrange(len(attr_indices)):
			attr_type1 = attr_types[attr_indices[i]]
			if attr_type1 == "C": continue

			for j in xrange(i+1, len(attr_indices)):
				attr_type2 = attr_types[attr_indices[j]]
				if attr_type2 == "C": continue

				for val1 in attr_values[attr_indices[i]]:
					for val2 in attr_values[attr_indices[j]]:
						list = []
						list.append([attr_indices[i], val1])
						list.append([attr_indices[j], val2])
						vars.append(list)

	print("============= variables =======")
	for list in vars:
		for combination in list:
			print(str(combination[0]) + ":" + combination[1]),
		print


	# initialize the weight vector
	w = np.zeros(len(vars) + 1)
	#print(w)

	# learning rate
	r = 0.001

	correct = 0
	incorrect = 0
	for i in xrange(len(examples)):
		# if it reaches the maxIterations, stop learning.
		if i >= maxIterations: break

		# compute feature vector
		x = _x(vars, examples[i].row)
		#print(x)

		# get the true label
		if examples[i].label == "+": y = 1
		else: y = -1

		# predict the label
		h = np.sign(np.dot(w, x))
		if h == y:
			#print("OK")
			correct += 1
			continue
		else:
			#print("NG")
			incorrect += 1
			# update the weight vector
			w += r * y * x;
			#print(w)

	print("============== final weight ============")
	print(w)
	print("correct: " + str(correct) + " / incorrect: " + str(incorrect))

	ret = []
	print("============== test on the training data ========")
	ret.append(report(examples, w, vars))

	print("============== test on the validation data ========")
	examples = readData("validation.txt")
	ret.append(report(examples, w, vars))

	print("============== test on the test data ========")
	examples = readData("test.txt")
	ret.append(report(examples, w, vars))

	return ret

#readfile:
#   Input: filename
#   Output: return a list of rows.
def readData(filename):
	f = open(filename).read()
	examples = []
	for line in f.split('\r'):
		if line == "": continue
		row = line.split('\t')
		label = row[len(row) - 1]
		row.pop(len(row) - 1)
		examples.append(Example(row, label, 1.0))

	return examples

if __name__ == '__main__':
	for maxIterations in xrange(10, 491):
		results = Perceptron(maxIterations, 1)

