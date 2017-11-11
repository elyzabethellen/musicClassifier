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
				mfccFeat = librosa.feature.mfcc(y[1:3000], n_mfcc=13) #take the abs here, we will normalize later also so don't want to lose vals
				mfcc = mfccFeat.tolist()[0] #librosa gives us a nested numpy array, we don't lose any info by doing this.
				mfccFeatures.append(mfcc)
				classifications.append(classKeys.get(p))
	return mfccFeatures, classifications

def specCentExtractor(data, filename): #spectral Centroids
	specFeatures = []
	classifications = []
	for p in paths:
		for file in os.listdir(p):
			if fnmatch.fnmatch(file, '*.au'):
				y, sr = librosa.load('' + p + file)  # y, sr = np array, sample rate
				y, sr = librosa.load('genres/blues/blues.00000.au')
				specFeat = librosa.feature.spectral_centroid(y, sr)
				spec = specFeat.tolist()[0]
				specFeatures.append(spec)
				classifications.append(classKeys.get(p))
	return specFeatures, classifications

def tempogramExtractor(data, filename): #spectral Centroids
	tempoFeatures = []
	classifications = []
	for p in paths:
		for file in os.listdir(p):
			if fnmatch.fnmatch(file, '*.au'):
				y, sr = librosa.load('' + p + file)  # y, sr = np array, sample rate
				y, sr = librosa.load('genres/blues/blues.00000.au')
				tempoFeat = librosa.feature.tempogram(y, sr)
				tempo = tempoFeat.tolist()[0]
				tempoFeatures.append(tempo)
				classifications.append(classKeys.get(p))
	return tempoFeatures, classifications

def featureWriter(data, filename):
	df = pd.DataFrame(data)
	df.to_csv(filename)

def featureWriterWithClass(data, filename, headers):
	df = pd.DataFrame(data, columns=headers)

def writeClassifications(classifications):
	with open('classifications.csv', 'w') as f:
		writer = csv.writer(f, quoting=csv.QUOTE_ALL)
		writer.writerow(classifications)

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
dir = 'genres/'
paths = list(dir + g + '/' for g in genres)
classKeys = dict(zip(paths, genres)) #map the path back to the genre for classification

'''''
fftFeatures, classifications = fftExtractor(paths, classKeys)
data = {'fft': fftFeatures, 'xclass': classifications}
featureWriter(fftFeatures, 'fftFeatures.csv')
headers = range(0, len(fftFeatures[0]))
featureWriterWithClass(data, 'fftFeaturesWithClass.csv', headers)


mfccFeatures, classifications = mfccExtractor(paths, classKeys)
featureWriter(mfccFeatures, 'mfccFeatures2.csv')


specFeatures, classifications = specCentExtractor(paths, classKeys)
featureWriter(specFeatures, 'specFeatures2.csv')
'''

tempoFeatures, classifications = tempogramExtractor(paths, classKeys)
featureWriter(tempoFeatures, 'tempoFeatures.csv')


#classifications = [c for c in genres for i in range(90)]
#print(classifications)
#writeClassifications(classifications)