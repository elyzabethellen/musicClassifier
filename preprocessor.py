#Elizabeth E. Esterly
#The University of New Mexico
#preprocessor.py
#preprocess the data to extract features and create a .csv for each approach

import os
import librosa
import fnmatch
import scipy
import csv
import pandas as pd

def fftExtractor(paths, classKeys):
	fftFeatures = []
	classifications = []
	for p in paths:
		for file in os.listdir(p):
			if fnmatch.fnmatch(file, '*.au'):
				y, sr = librosa.load('' + p + file)  # y, sr = np array, sample rate
				fftFeat = abs(scipy.fft(y)[1:1000])
				fftFeatures.append(fftFeat)
				classifications.append(classKeys.get(p))
	return fftFeatures, classifications

def mfccExtractor(paths, classKeys):
	mfccFeatures = []
	classifications = []
	for p in paths:
		for file in os.listdir(p):
			if fnmatch.fnmatch(file, '*.au'):
				y, sr = librosa.load('' + p + file)  # y, sr = np array, sample rate
				mfccFeat = librosa.feature.mfcc(y, n_mfcc=13)
				mfccFeatures.append(mfccFeat)
				classifications.append(classKeys.get(p))
	return mfccFeatures, classifications

def featureWriter(data, filename):
	df = pd.DataFrame(data)
	df.to_csv(filename)

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
dir = 'genres/'
paths = list(dir + g + '/' for g in genres)
classKeys = dict(zip(paths, genres)) #map the path back to the genre for classification

fftFeatures, classifications = fftExtractor(paths, classKeys)
data = {'fft': fftFeatures, 'xclass': classifications}
featureWriter(fftFeatures, 'fftFeatures.csv')




#index = list(range(0, len(classifications)))
#df = pd.DataFrame(data, index=index)
#print(df)


