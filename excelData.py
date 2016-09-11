# import csv

# def loadCsv(filename):
#     lines = csv.reader(open(filename, "rt"))
#     dataset = list(lines)
#     for i in range(len(dataset)):
#         dataset[i] = [float(x) for x in dataset[i]]
#     return dataset

# filename = 'pima-indians-diabetes.data.csv'
# dataset = loadCsv(filename)
# print('Loaded data file %s with %s rows' % (filename, len(dataset)))



# import random

# def splitDataset(dataset, splitRatio):
#     trainSize = int(len(dataset) * splitRatio)
#     trainSet = []
#     copy = list(dataset)
#     while len(trainSet) < trainSize:
#         index = random.randrange(len(copy))
#         trainSet.append(copy.pop(index))
#     return [trainSet, copy]

# dataset = [[1], [2], [3], [4], [5]]
# splitRatio = 0.67
# train, test = splitDataset(dataset, splitRatio)
# print('Split %s rows into train with %s and test with %s' % (len(dataset), train, test))



# def separateByClass(dataset):
#     separated = {}
#     for i in range(len(dataset)):
#         vector = dataset[i]
#         if (vector[-1] not in separated):
#             separated[vector[-1]] = []
#         separated[vector[-1]].append(vector)
#     return separated

# dataset = [[1,20,1], [2,21,0], [3,22,1]]
# separated = separateByClass(dataset)
# print('Separated instances: %s' % (format(separated)))


# import math
# def mean(numbers):
# 	return sum(numbers)/float(len(numbers))

# def stdev(numbers):
# 	avg = mean(numbers)
# 	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
# 	return math.sqrt(variance)

# numbers = [1,2,3,4,5]
# print('Summary of %s: mean=%s, stdev=%s' % (numbers, format(mean(numbers)), format(stdev(numbers))))

# def summarize(dataset):
#     summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
#     del summaries[-1]
#     return summaries

# dataset = [[1,20,0], [2,21,1], [3,22,0]]
# summary = summarize(dataset)
# print('Attribute summaries: %s' % (format(summary)))

# def summarizeByClass(dataset):
#     separated = separateByClass(dataset)
#     summaries = {}
#     for classValue, instances in separated.items():
#         summaries[classValue] = summarize(instances)
#     return summaries

# dataset = [[1,20,1], [2,21,0], [3,22,1], [4,22,0]]
# summary = summarizeByClass(dataset)
# print('Summary by class value: %s' % (format(summary)))




import math

def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

x = 71.5
mean = 73
stdev = 6.2
probability = calculateProbability(x, mean, stdev)
print('Probability of belonging to this class: %s' % (format(probability)))

def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities

summaries = {0:[(1, 0.5)], 1:[(20, 5.0)]}
inputVector = [1.1, '?']
probabilities = calculateClassProbabilities(summaries, inputVector)
print('Probabilities for each class: %s' % (format(probabilities)))

def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel
summaries = {'A':[(1, 0.5)], 'B':[(20, 5.0)]}
inputVector = [1.1, '?']
result = predict(summaries, inputVector)
print('Prediction: %s' % (format(result)))