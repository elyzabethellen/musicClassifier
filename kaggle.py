#online predictions

import os
import fnmatch
import librosa
import csv
import scipy
import numpy as np
from sklearn import preprocessing

def makePrediction(model, feature):
	genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
	y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	genresKey = dict(zip(y, genres))
	with open('predictionsAfterAdjustment.csv', 'w') as f:
		writer = csv.writer(f)
		header = ['id', 'class']
		writer.writerow(header)
		dir = 'rename/'
		for file in os.listdir(dir):
			if fnmatch.fnmatch(file, '*.au'):
				y, sr = librosa.load('' + dir + file)
				if feature == 'fft':
					fftFeat = abs(scipy.fft(y)[1:1000])
					toPredict = preprocessing.minmax_scale(fftFeat)
				if feature == 'mfcc':
					mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
					toPredict = np.array(preprocessing.minmax_scale(mfccs))
				if feature == 'specContrast':
					stft = np.abs(librosa.stft(y))
					contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
					toPredict = preprocessing.minmax_scale(contrast)
				if feature == 'tonnetz':
					tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
					toPredict = preprocessing.minmax_scale(tonnetz)
				pred = model.predict([toPredict])
				result = genresKey.get(pred[0])
				lst = [file, result]
				print(lst)
				writer.writerow(lst)

def writePredictionFromList(l):
	genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
	y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	genresKey = dict(zip(y, genres))
	with open('list_validation.txt') as f:
		lines = f.read().splitlines()
	with open('predictionsAfterAdjustment.csv', 'w') as f:
		writer = csv.writer(f)
		header = ['id', 'class']
		writer.writerow(header)
		for i in range (len(l)):
			lst = [lines[i], genresKey.get(l[i])]
			writer.writerow(lst)







