from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from classifierSupport import *
from kaggle import writePredictionFromList

clf = MLPClassifier(hidden_layer_sizes=(15,), random_state=1, max_iter=1, warm_start=True)