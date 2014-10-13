import numpy as np
import pylab as plt
import math
import sys

featureComputed = False
trainFeatures = []
valFeatures = []
testFeatures = []

###############################################################################
# Example class
#
# This class represents each example.
#
class Example:
	def __init__(self, row, label, weight):
		self.row = row
		self.label = label
		self.weight = weight

###############################################################################
# Create a feature vector for a given example based on the feature representation
#  vars   feature representation
#  row    example
#  return feature vector
def _x(attr_types, vars, examples):
	ret = []

	for example in examples:
		x = np.zeros(len(vars) + 1)

		for i in xrange(len(vars)):
			bit = 1
			for var in vars[i]:
				if example.row[var[0]] == "?":
					bit = 0
				elif attr_types[var[0]] == "B":
					if example.row[var[0]] != var[1]:
						bit = 0
				else:
					if not eval(example.row[var[0]] + str(var[1])):
						bit = 0

			x[i] = bit

		x[len(vars)] = 1
		ret.append(x)

	return ret

###############################################################################
# sgn function
#
# return 1 for x > 0, -1 otherwise.
#
def _sgn(x):
	if np.sign(x) > 0: return 1
	else: return -1

###############################################################################
# Report the accuracy, precision, recall, and F1 for given examples and weight vector
#
# Calculate performance values and return them in the following format.
# [accuracy, precision, recall, F1-score]
#
def report(examples, features, w, vars, attr_types):
	correct = 0
	incorrect = 0
	true_pos = 0
	false_pos = 0
	false_neg = 0

	for i in xrange(len(examples)):
		if examples[i].label == "+": y = 1
		else: y = -1

		h = _sgn(np.dot(w, features[i]))
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
	precision = 0
	recall = 0
	F1 = 0
	if true_pos + false_pos > 0:
		precision = float(true_pos) / float(true_pos + false_pos)
	if true_pos + false_neg > 0:
		recall = float(true_pos) / float(true_pos + false_neg)
	if precision + recall > 0:
		F1 = 2 * precision * recall / (precision + recall)

	return [accuracy, precision, recall, F1]

###############################################################################
# Find the thresholds
def findThresholds(examples, attr_index):
	thresholds = []

	# extract only valid values
	examples2 = []
	for example in examples:
		if example.row[attr_index] == "?": continue
		examples2.append(example)

	if len(examples2) == 0: return thresholds

	# sort
	examples2.sort(key=lambda example: float(example.row[attr_index]))

	previous_label = examples2[0].label
	previous_value = float(examples2[0].row[attr_index])
	for example2 in examples2:
		if previous_label == "?" and previous_value == float(example2.row[attr_index]): continue

		if example2.label != previous_label:
			previous_label = example2.label

			if previous_value == float(example2.row[attr_index]):
				previous_label = "?"
			else:
				thresholds.append((previous_value + float(example2.row[attr_index])) * 0.5)
		previous_value = float(example2.row[attr_index])

	return thresholds

def buildFeatureSet(attr_index, attr_type, attr_values):
	featureSet = []

	if attr_type == "B":
		for val in attr_values:
			featureSet.append([[attr_index, val]])
	else:
		for i in xrange(len(attr_values)):
			var = attr_values[i]

			if i == len(attr_values) - 1:
				featureSet.append([[attr_index, ">=" + str(var)]])
			else:
				if i == 0:
					featureSet.append([[attr_index, "<" + str(var)]])
				next_var = attr_values[i + 1]
				featureSet.append([[attr_index, ">=" + str(var)], [attr_index, "<" + str(next_var)]])

	return featureSet

###############################################################################
# Perceptron
#
# @param maxIterations	the maximum iteration.
# @param featureSet		1 - original attributes / 2 - feature pairs / 3 - use all the features in 1-2 as required in the instruction.
# @return performance results in the order of "training data", "validation data", and "test data". For each data, "accuracy", "precision", "recall", and "F1-score" are stored in the list.
def Perceptron(maxIterations, featureSet):
	global featureComputed
	global trainFeatures
	global valFeatures
	global testFeatures

	attr_types = ["B", "C", "C", "B", "B", "B", "B", "C", "B", "B", "C", "B", "B", "C", "C"]

	examples = readData("train.txt")

	# get all the possible values for each attribute
	# MODIFIED: for the continuous attributes, we use thresholds.
	attr_values = {}
	for attr_index in xrange(len(attr_types)):
		#if attr_type == "C": continue

		#print(attr_index)

		attr_values[attr_index] = []

		if attr_types[attr_index] == "B":
			for example in examples:
				if example.row[attr_index] == "?": continue
				if example.row[attr_index] not in attr_values[attr_index]:
					attr_values[attr_index].append(example.row[attr_index])
		else:
			attr_values[attr_index] = findThresholds(examples, attr_index)


		#for val in attr_values[attr_index]:
		#	print("  " + str(val))

	# setup feature representation
	vars = []
	if featureSet == 1 or featureSet == 3:
		for attr_index in xrange(len(attr_types)):
			#if attr_type == "C": continue

			vars += buildFeatureSet(attr_index, attr_types[attr_index], attr_values[attr_index])


	if featureSet == 2 or featureSet == 3:
		#attr_indices = attr_types.keys()
		for i in xrange(len(attr_types)):
			#attr_type1 = attr_types[attr_indices[i]]
			#if attr_type1 == "C": continue
			set1 = buildFeatureSet(i, attr_types[i], attr_values[i])

			for j in xrange(i+1, len(attr_types)):
				#attr_type2 = attr_types[attr_indices[j]]
				#if attr_type2 == "C": continue
				set2 = buildFeatureSet(j, attr_types[j], attr_values[j])

				for k in xrange(len(set1)):
					for l in xrange(len(set2)):
						vars.append(set1[k] + set2[l])

	#print("============= variables =======")
	#for list in vars:
	#	for combination in list:
	#		print(str(combination[0]) + ":" + combination[1]),
	#	print

	# initialize the weight vector
	w = np.zeros(len(vars) + 1)
	#print("w: " + str(w))

	# learning rate
	r = 0.01

	# compute the feature vectors for all the examples
	if not featureComputed:
		trainFeatures = _x(attr_types, vars, examples)

	correct = 0
	incorrect = 0
	for iter in xrange(maxIterations):
		for i in xrange(len(examples)):
			sys.stdout.write(".")

			# get the true label
			if examples[i].label == "+": y = 1
			else: y = -1

			# predict the label
			h = _sgn(np.dot(w, trainFeatures[i]))
			#print("x: " + str(x) + " h: " + str(h) + " y: " + str(y))
			if h == y:
				#print("OK")
				correct += 1
				continue
			else:
				#print("NG")
				incorrect += 1
				# update the weight vector
				w += r * y * trainFeatures[i];
				#print("w: " + str(w))
	print

	#print("============== final weight ============")
	#print(w)
	#print("correct: " + str(correct) + " / incorrect: " + str(incorrect))

	ret = []
	#print("============== test on the training data ========")
	ret.append(report(examples, trainFeatures, w, vars, attr_types))

	#print("============== test on the validation data ========")
	examples = readData("validation.txt")
	if not featureComputed:
		valFeatures = _x(attr_types, vars, examples)
	ret.append(report(examples, valFeatures, w, vars, attr_types))

	#print("============== test on the test data ========")
	examples = readData("test.txt")
	if not featureComputed:
		testFeatures = _x(attr_types, vars, examples)
	ret.append(report(examples, testFeatures, w, vars, attr_types))

	featureComputed = True

	return (ret, w)

###############################################################################
# readfile:
#   Input: filename
#   Output: return a list of rows.
def readData(filename):
	f = open(filename).read()
	examples = []
	for line in f.split('\r'):
		if line == "": continue
		row = line.split('\t')

		# ignore the example which contains missing values
		if "?" in row: continue

		label = row[len(row) - 1]
		row.pop(len(row) - 1)
		examples.append(Example(row, label, 1.0))

	return examples

###############################################################################
# Draw learning curves of Perceptron
def drawLearningCurve(startMaxIterations, endMaxIterations, featureSet, saveFile):
	nExamples = []
	list_t = []
	list_v = []
	list_ts = []
	max_performance = -1
	max_maxIterations = 0
	max_results = []

	# use the accuracy to draw the learning curve
	perfType = 0

	for maxIterations in xrange(startMaxIterations, endMaxIterations):
		print(maxIterations)
		(results, w) = Perceptron(maxIterations, featureSet)
		#print("final w: " + str(w))
		nExamples.append(maxIterations)
		list_t.append(results[0][perfType])
		list_v.append(results[1][perfType])
		list_ts.append(results[2][perfType])

		# keep the best performance
		if results[1][perfType] > max_performance:
			max_performance = results[1][perfType]
			max_maxIterations = maxIterations
			max_results = results

	# show the best
	print("maxIterations: " + str(max_maxIterations) + " (accuracy: " + str(max_performance) + ")")
	print("=== Training data ===")
	print("accuracy: " + str(max_results[0][0]) + " / precision: " +  str(max_results[0][1]) + " / recall: " + str(max_results[0][2]) + " / F1: " + str(max_results[0][3]))
	print("=== Validation data ===")
	print("accuracy: " + str(max_results[1][0]) + " / precision: " +  str(max_results[1][1]) + " / recall: " + str(max_results[1][2]) + " / F1: " + str(max_results[1][3]))
	print("=== Test data ===")
	print("accuracy: " + str(max_results[2][0]) + " / precision: " +  str(max_results[2][1]) + " / recall: " + str(max_results[2][2]) + " / F1: " + str(max_results[2][3]))

	# save the best
	f = open("best_results.txt", "w")
	f.write("maxIterations: " + str(max_maxIterations) + " (accuracy: " + str(max_performance) + ")\n")
	f.write("=== Training data ===\n")
	f.write("accuracy: " + str(max_results[0][0]) + " / precision: " +  str(max_results[0][1]) + " / recall: " + str(max_results[0][2]) + " / F1: " + str(max_results[0][3]) + "\n")
	f.write("=== Validation data ===\n")
	f.write("accuracy: " + str(max_results[1][0]) + " / precision: " +  str(max_results[1][1]) + " / recall: " + str(max_results[1][2]) + " / F1: " + str(max_results[1][3]) + "\n")
	f.write("=== Test data ===\n")
	f.write("accuracy: " + str(max_results[2][0]) + " / precision: " +  str(max_results[2][1]) + " / recall: " + str(max_results[2][2]) + " / F1: " + str(max_results[2][3]) + "\n")
	f.close()

	# show the accuracy graph
	plt.plot(nExamples, list_t, "-", label="training")
	plt.plot(nExamples, list_v, "-", label="validation")
	plt.plot(nExamples, list_ts, "-", label="test")
	plt.title("Learning Curve")
	plt.xlim(0, endMaxIterations)
	plt.ylim(0, 1.0)
	plt.legend(loc='lower left')

	if (saveFile):
		plt.savefig("result.eps")

	plt.show()


###############################################################################
# Main function
if __name__ == '__main__':
	# this flag is to avoid the redundant computation when extracting feature vectors
	featureComputed = False

	#Perceptron(1, 1)
	#(results, w) = Perceptron(1, 2)
	#print(results)

	# draw learning curves for each featureSet type
	drawLearningCurve(1, 11, 1, True)
	#drawLearningCurve(1, 11, 2, True)
	#drawLearningCurve(1, 11, 3, True)

