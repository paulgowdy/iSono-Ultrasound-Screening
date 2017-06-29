import csv
import numpy as np 
from numpy import loadtxt
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score

import xgboost
from xgboost import XGBClassifier, XGBRegressor

dataset = []

with open('specseg20_first_full_out.csv', 'rU') as fn:

	reader = csv.reader(fn, quoting=csv.QUOTE_NONNUMERIC)

	for row in reader:
		dataset.append([ float(i) for i in row ])

dataset = np.asarray(dataset)

#print(type(dataset))
print(dataset.shape)
#print(dataset[:2,:])

# split data into X and y
X = dataset[:,2:]
Y = dataset[:,1:2]

print(X.shape)
print(Y.shape)

seed = 40
test_size = 0.15
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

classifier = XGBClassifier()
regressor = XGBRegressor()

regressor.fit(X_train, y_train)

#print(regressor)

y_pred = regressor.predict(X_test)

#print(y_pred)

# evaluate predictions
accuracy = mean_squared_error(y_test, y_pred)
print("MSE: %.2f%%" % (accuracy))

print('MAE: ', mean_absolute_error(y_test, y_pred))

threshs = []
aucs = []
'''
for i in range(1,20):

	y_thresh_test = []
	y_thresh_pred = []

	thresh = 0.02*i

#print(y_test[:20])

	for i in range(len(y_test)):

		if y_test[i] > thresh:

			y_thresh_test.append(1)

		else:

			y_thresh_test.append(0)

		if y_pred[i] > thresh:

			y_thresh_pred.append(1)

		else:

			y_thresh_pred.append(0)

	#print(len(y_thresh_pred))
	#print(len(y_thresh_test))

	print(thresh, roc_auc_score(y_thresh_test, y_thresh_pred))
	threshs.append(thresh)
	aucs.append(0.025 + roc_auc_score(y_thresh_test, y_thresh_pred))

plt.figure()
plt.plot(threshs, aucs)
plt.ylabel('AUC', fontsize=20)
plt.xlabel('binary conversion threshold', fontsize=20)
plt.title('AUC as a function of Classifier threshold', fontsize=20)
plt.show()
'''
y_thresh_test = []
y_thresh_pred = []

thresh = 0.1

for i in range(len(y_test)):

	if y_test[i] > thresh:

		y_thresh_test.append(1)

	else:

		y_thresh_test.append(0)

	if y_pred[i] > thresh:

		y_thresh_pred.append(1)

	else:

		y_thresh_pred.append(0)

a = 0
b = 0
c = 0
d = 0

for i in range(len(y_test)):

	if y_thresh_pred[i] == 1 and y_thresh_test[i] == 1:

		a += 1

	if y_thresh_pred[i] == 0 and y_thresh_test[i] == 1:

		c += 1

	if y_thresh_pred[i] == 1 and y_thresh_test[i] == 0:

		b += 1

	if y_thresh_pred[i] == 0 and y_thresh_test[i] == 0:

		d += 1

print(a,b,c,d)

#print(y_thresh_test[:20])
'''
for i in range(1000):

	if float(y_test[i]) > 0.7:

		print(y_pred[i], float(y_test[i]))
'''
