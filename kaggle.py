import os
import fnmatch
import librosa
from sklearn import preprocessing

def makePrediction(model):
	genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
	y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	genresKey = dict(zip(y, genres))
	with open('predictions.txt', 'w') as f:
		dir = 'rename/'
		for file in os.listdir(dir):
			if fnmatch.fnmatch(file, '*.au'):
				y, sr = librosa.load('' + dir + file)
				toPredict = preprocessing.minmax_scale(y)
				pred = model.predict(toPredict)
				result = genresKey.get(pred)
				print(result)
				lst = [file, ' ', result]
				f.write(','.join(map(str, lst))+'\n')




