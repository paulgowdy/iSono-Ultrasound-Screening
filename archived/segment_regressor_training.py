import csv
import numpy as np 
from numpy import loadtxt
import matplotlib.pyplot as plt
from random import shuffle
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score, confusion_matrix

import xgboost
from xgboost import XGBClassifier, XGBRegressor

segments = [35, 40, 45, 50, 55, 60]
betas = [5, 7, 8, 10] 

save_folder_prefix = 'models/segment_models/'

regressors = []

for n_seg in segments:

	for b in betas:

		dataset = []

		fn = 'segment_stats_parameter_specific_saves/segment_stats_' + str(n_seg) + '_' + str(b) + '.csv'

		with open(fn, 'rU') as fn:

			reader = csv.reader(fn, quoting=csv.QUOTE_NONNUMERIC)

			for row in reader:
				dataset.append([ float(i) for i in row ])

		#shuffle(dataset)

		dataset = np.asarray(dataset)

		#print(dataset.shape)

		# split data into X and y
		X = dataset[:,4:]
		Y = dataset[:,1:2]

		regressor = XGBRegressor()

		print('fitting...')
		regressor.fit(X, Y)

		regressors.append([n_seg, b, regressor])

print('Pickling Regressors...')

with open(save_folder_prefix + 'segment_regressors_multiparameter.pickle', 'wb') as handle:
    pickle.dump(regressors, handle)




