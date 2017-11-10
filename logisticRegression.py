from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def kFoldCrossValidation(k, train, target, model):
	kf = KFold(n_splits=10, shuffle=True) # 10-folds, shuffle or genres will group together
	for trainIndex, testIndex in kf.split(train):
		print(trainIndex)
		print(testIndex)
		Xtrain, Xtest = train[trainIndex], train[testIndex]
		ytrain, ytest = target[trainIndex], target[testIndex]
		model.fit(Xtrain, ytrain) #fit on each round
	return model, Xtest, ytest

def setupTrainData(featureFile):
	df = pd.read_csv(featureFile)
	train = df.values.T #must transpose here! features were written transposed from numpy.vals
	return train

def internalTesting(model, expected, testData):
	predicted = model.predict(testData)
	print(metrics.classification_report(expected, predicted))
	return predicted

def crossValScoring(model, train, test, iter):
	scores = cross_val_score(model, train, test, iter)
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def confusionMatrix(expected, predicted, save):
	cm = metrics.confusion_matrix(expected, predicted)
	fig, ax = plt.subplots()
	fig.subplots_adjust(left=0.3)
	im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.ocean)
	fig.colorbar(im, ax=ax)
	fig.savefig(save)

#make the logistic regression model
model = LogisticRegression()

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [x for x in y for i in range(90)] #repeat each label 90 times, that's all our classifications
y = np.array(y)

X = setupTrainData('scaledSpec.csv')
model, Xtest, expected = kFoldCrossValidation(10, X, y, model)
predicted = internalTesting(model, expected, Xtest)
confusionMatrix(expected, predicted, 'spec2LogReg.pdf')

model = LogisticRegression()
X = setupTrainData('scaledMfcc.csv')
model, Xtest, expected = kFoldCrossValidation(10, X, y, model)
predicted = internalTesting(model, expected, Xtest)
confusionMatrix(expected, predicted, 'mfcc2LogReg.pdf')

model = LogisticRegression()
X = setupTrainData('scaledFft.csv')
model, Xtest, expected = kFoldCrossValidation(10, X, y, model)
predicted = internalTesting(model, expected, Xtest)
confusionMatrix(expected, predicted, 'fft2LogReg.pdf')

###############
model = LogisticRegression()
X = setupTrainData('scaledSpec.csv')
model, Xtest, expected = kFoldCrossValidation(10, X, y, model)
X = setupTrainData('scaledMfcc.csv')
model, Xtest, expected = kFoldCrossValidation(10, X, y, model)
X = setupTrainData('scaledFft.csv')
model, Xtest, expected = kFoldCrossValidation(10, X, y, model)
predicted = internalTesting(model, expected, Xtest)
confusionMatrix(expected, predicted, 'composite.pdf')




