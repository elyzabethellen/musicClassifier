#Elizabeth E. Esterly
#The University of New Mexico
#preprocessor.py
#preprocess the data to extract features, normalize, and create a .csv for each approach

import os
import librosa
import fnmatch
import scipy
import csv
import numpy as np
from sklearn import preprocessing

def rawDataExtractor(paths):
	with open ('rawData.csv', 'w') as f:
		writer = csv.writer(f)
		for p in paths:
			for file in os.listdir(p):
				if fnmatch.fnmatch(file, '*.au'):
					y, sr = librosa.load('' + p + file)  # y, sr = np array, sample rate
					writer.writerow(y[1:1000])

def extractAllLibrosaFeatures(paths):
	mfccFeatures = []
	specContrastFeat = []
	chromaFeatures = []
	tonnetzFeatures = []
	for p in paths:
		for file in os.listdir(p):
			if fnmatch.fnmatch(file, '*.au'):
				y, sr = librosa.load('' + p + file)



				mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
				mfccFeatures.append(mfccs)

				stft = np.abs(librosa.stft(y))

				contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
				specContrastFeat.append(contrast)

				chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
				chromaFeatures.append(chroma)

				tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
				tonnetzFeatures.append(tonnetz)

	#normalize with min-max scale
	mfccFeatures = preprocessing.minmax_scale(mfccFeatures)
	specContrastFeat = preprocessing.minmax_scale(specContrastFeat)
	tonnetzFeatures = preprocessing.minmax_scale(tonnetzFeatures)
	chromaFeatures = preprocessing.minmax_scale(chromaFeatures)

	#write out
	np.savetxt('newMfccs.csv', mfccFeatures, delimiter=',')
	np.savetxt('newSpecContrast.csv', specContrastFeat, delimiter=',')
	np.savetxt('newTonnetz.csv', tonnetzFeatures, delimiter=',')
	np.savetxt('newChroma.csv', chromaFeatures, delimiter=',')


#individual methods follow
def tonnetzExtractor(paths, outfile):
	tonnetzFeatures = []
	for p in paths:
		for file in os.listdir(p):
			if fnmatch.fnmatch(file, '*.au'):
				y, sr = librosa.load('' + p + file)
				tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
				tonnetzFeatures.append(tonnetz)
	tonnetzFeatures = preprocessing.minmax_scale(tonnetzFeatures)
	np.savetxt(outfile, tonnetzFeatures, delimiter=',')

def spectralContrastExtractor(paths, outfile):
	specContrastFeat = []
	for p in paths:
		for file in os.listdir(p):
			if fnmatch.fnmatch(file, '*.au'):
				y, sr = librosa.load('' + p + file)
				stft = np.abs(librosa.stft(y))
				contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
				specContrastFeat.append(contrast)
	specContrastFeat = preprocessing.minmax_scale(specContrastFeat)
	np.savetxt(outfile, specContrastFeat, delimiter=',')

def mfccExtractor(paths, outfile):
	mfccFeatures = []
	for p in paths:
		for file in os.listdir(p):
			if fnmatch.fnmatch(file, '*.au'):
				y, sr = librosa.load('' + p + file)
				mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=100).T, axis=0)
				mfccFeatures.append(mfccs)
	mfccFeatures = preprocessing.minmax_scale(mfccFeatures)
	np.savetxt(outfile, mfccFeatures, delimiter=',')

def chromaExtractor(paths, outfile):
	chromaFeatures = []
	for p in paths:
		for file in os.listdir(p):
			if fnmatch.fnmatch(file, '*.au'):
				y, sr = librosa.load('' + p + file)
				stft = np.abs(librosa.stft(y))
				chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
				chromaFeatures.append(chroma)
	chromaFeatures = preprocessing.minmax_scale(chromaFeatures)
	np.savetxt(outfile, chromaFeatures, delimiter=',')

def fftExtractor(paths, outfile):
	fftFeatures = []
	classifications = []
	for p in paths:
		for file in os.listdir(p):
			if fnmatch.fnmatch(file, '*.au'):
				y, sr = librosa.load('' + p + file)  # y, sr = np array, sample rate
				fftFeat = abs(scipy.fft(y)[1:1000])
				fftFeatures.append(fftFeat)
	fftFeatures = preprocessing.minmax_scale(fftFeatures)
	np.savetxt(outfile, fftFeatures, delimiter=',')


genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
dir = 'genres/'
paths = list(dir + g + '/' for g in genres)
mfccExtractor(paths, '100mfcc.csv')
paths = ['rename/']
mfccExtractor(paths, '100mfccKaggleProcess.csv')


