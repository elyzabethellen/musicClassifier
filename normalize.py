import pandas as pd
from sklearn import preprocessing

def normalizeFeatures(sourcefile, outfile):
	df = pd.read_csv(sourcefile)
	normalized = preprocessing.normalize(df.values.T)
	df = pd.DataFrame(normalized)
	df.to_csv(outfile)

normalizeFeatures('fftFeatures.csv', 'normalizedFft.csv')
normalizeFeatures('mfccFeatures.csv', 'normalizedMfcc.csv')
normalizeFeatures('specFeatures.csv','normalizedSpec.csv')



