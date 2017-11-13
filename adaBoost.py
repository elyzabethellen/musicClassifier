import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from logisticRegression import kFoldCrossValidation
from kaggle import writePredictionFromList

model = AdaBoostClassifier(DecisionTreeClassifier( max_depth=2), n_estimators=100, learning_rate=1.5, algorithm="SAMME")

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [x for x in y for i in range(90)] #repeat each label 90 times, that's all our classifications
y = np.array(y)

X = np.loadtxt('newMfccs.csv', delimiter=',')
model, Xtest, expected = kFoldCrossValidation(30, X, y, model)
scores = cross_val_score(model, X, y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
newX = np.loadtxt('mfccKaggleProcess.csv', delimiter=',')
kagY = model.predict(newX)
writePredictionFromList(kagY)