import os
import fnmatch
import librosa
import csv
import scipy
import numpy as np
from sklearn import preprocessing

def makePrediction(model, type):
	genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
	y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	genresKey = dict(zip(y, genres))
	with open('predictions.csv', 'w') as f:
		writer = csv.writer(f)
		writer.writerow('id, class')
		dir = 'rename/'
		for file in os.listdir(dir):
			if fnmatch.fnmatch(file, '*.au'):
				if type is 'fft':
					fftFeat = abs(scipy.fft(y)[1:1000])
					toPredict = preprocessing.minmax_scale(fftFeat)
					toPredict = toPredict.reshape(-1, 1).T
				if type is 'spec':
					specFeat = librosa.feature.spectral_centroid(y, sr)
					spec = specFeat.tolist()
					toPredict = preprocessing.normalize(spec)
				pred = model.predict(toPredict)[0]
				result = genresKey.get(pred)
				lst = [file, result]
				writer.writerow(lst)

def makeRFEPrediction(model, idx):
	genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
	y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	genresKey = dict(zip(y, genres))
	with open('predictions.csv', 'w') as f:
		writer = csv.writer(f)
		writer.writerow('id, class')
		dir = 'rename/'
		for file in os.listdir(dir):
			if fnmatch.fnmatch(file, '*.au'):
				y, sr = librosa.load('' + dir + file)
				grab = preprocessing.minmax_scale(y[1:1000])
				res = grab[idx]
				pred = model.predict(res)[0]
				result = genresKey.get(pred)
				lst = [file, result]
				writer.writerow(lst)




