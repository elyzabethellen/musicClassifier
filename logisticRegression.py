from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from classifierSupport import *
from kaggle import writePredictionFromList




#make the logistic regression model
model = LogisticRegression()

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [x for x in y for i in range(90)] #repeat each label 90 times, that's all our classifications
y = np.array(y)


X = np.loadtxt('100mfcc.csv', delimiter=',')
model, Xtest, expected = kFoldCrossValidation(25, X, y, model)
scores = cross_val_score(model, X, y, cv=50)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = internalTesting(model, expected, Xtest)
confusionMatrix(expected, predicted, '100mfcc.pdf')
newX = np.loadtxt('100mfccKaggleProcess.csv', delimiter=',')
kagY = model.predict(newX)
writePredictionFromList(kagY)

