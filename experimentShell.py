import numpy as np
import math # For the floor

from sklearn import datasets

iris = datasets.load_iris()

#Combine together the iris data with iris targets
combinedData = np.column_stack((iris.data, iris.target))

#Randomize the data set 100 times
for i in range(0, 100):
	np.random.shuffle(combinedData)

# The length of the combined data set
setLength = len(combinedData)

#Split the data set size for the training and testing sets
seventySplit = int(math.floor(setLength * .7))
thirtySplit = setLength - seventySplit


# Initialize empty array with 0's for the training set and testing set
trainingSet = np.zeros(shape=(seventySplit, 5))
testSet = np.zeros(shape=(thirtySplit, 5))

# Fill the first seventy percent
for i in range(0, seventySplit):
	trainingSet[i] = combinedData[i]

# Fill the last thirty percent
for i in range(0, thirtySplit):
	testSet[i] = combinedData[i + seventySplit]

class HardCoded:
	def train(self, trainingSet):
		return

	def predict(self, testSet):
		return 0

	def accuracy(self, testSet):
		rate = 0
		for i in range(0, len(testSet)):
			if (testSet[i][4] == self.predict(testSet)):
				rate += 1
		return rate

hardCodedClassifier = HardCoded()
hardCodedClassifier.train(trainingSet)
hardCodedClassifier.predict(testSet)

correct = hardCodedClassifier.accuracy(testSet)

# Get the accuracy percentage
accuracy = (float(correct) / float(thirtySplit)) * 100
print (accuracy, "% accuracy rate")