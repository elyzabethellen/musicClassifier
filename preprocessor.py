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
				specFeat = librosa.feature.spectral_centroid(y, sr)
				spec = specFeat.tolist()[0]
				specFeatures.append(spec)
				classifications.append(classKeys.get(p))
	return specFeatures, classifications

def tempogramExtractor(data, filename): #estimate the tempo of the song
	tempoFeatures = []
	classifications = []
	for p in paths:
		for file in os.listdir(p):
			if fnmatch.fnmatch(file, '*.au'):
				y, sr = librosa.load('' + p + file)  # y, sr = np array, sample rate
				oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
				tempoFeat = librosa.feature.tempogram(y, sr, onset_envelope=oenv, hop_length=512)
				globalAutoCorr = librosa.autocorrelate(oenv, max_size=tempoFeat.shape[0])
				globalAutoCorr = librosa.util.normalize(globalAutoCorr)
				tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr, hop_length = 512)[0]
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
'''

specFeatures, classifications = specCentExtractor(paths, classKeys)
featureWriter(specFeatures, 'specFeatures2.csv')


tempoFeatures, classifications = tempogramExtractor(paths, classKeys)
featureWriter(tempoFeatures, 'tempoFeatures.csv')


#classifications = [c for c in genres for i in range(90)]
#print(classifications)
#writeClassifications(classifications)