from sklearn.model_selection import KFold
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt

def kFoldCrossValidation(k, train, target, model):
	kf = KFold(n_splits=10, shuffle=True) # 10-folds, shuffle or genres will group together
	for trainIndex, testIndex in kf.split(train):
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

def confusionMatrix(expected, predicted, save):
	cm = metrics.confusion_matrix(expected, predicted)
	fig, ax = plt.subplots()
	fig.subplots_adjust(left=0.3)
	im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.ocean)
	fig.colorbar(im, ax=ax)
	fig.savefig(save)