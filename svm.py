import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from logisticRegression import kFoldCrossValidation
from kaggle import writePredictionFromList

model = SVC(C=1.0, kernel='rbf', gamma=0.1)
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [x for x in y for i in range(90)] #repeat each label 90 times, that's all our classifications
y = np.array(y)

X = np.loadtxt('newFft.csv', delimiter=',')
model, Xtest, expected = kFoldCrossValidation(50, X, y, model)
scores = cross_val_score(model, X, y, cv=50)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
newX = np.loadtxt('fftKaggleProcess.csv', delimiter=',')
kagY = model.predict(newX)
writePredictionFromList(kagY)
