import pandas as pd
from sklearn import preprocessing

def normalizeFeatures(sourcefile, outfile):
	df = pd.read_csv(sourcefile)
	normalized = preprocessing.normalize(df.values.T)
	df = pd.DataFrame(normalized)
	df.to_csv(outfile)

#scaling to accommodate negative values
def scaleFeatures(sourcefile, outfile):
	df = pd.read_csv(sourcefile)
	scaled = preprocessing.minmax_scale(df.values.T)
	df = pd.DataFrame(scaled)
	df.to_csv(outfile)

scaleFeatures('fftFeatures.csv', 'scaledFft.csv')
scaleFeatures('mfccFeatures2.csv', 'scaledMfcc.csv')
scaleFeatures('specFeatures2.csv','scaledSpec.csv')



