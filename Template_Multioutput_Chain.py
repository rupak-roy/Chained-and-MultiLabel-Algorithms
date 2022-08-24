#MultiOutputClassifier-----------------------------------------------
import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

#create dataset
X, y = make_multilabel_classification(n_classes=3, random_state=0)
clf = MultiOutputClassifier(LogisticRegression()).fit(X, y)
clf.predict(X[-2:])

#ClassifierChain---------------------------------------------
from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import ClassifierChain

#make dataset
X, Y = make_multilabel_classification(n_samples=12, n_classes=3, random_state=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

base_lr = LogisticRegression(solver='lbfgs', random_state=0)
chain = ClassifierChain(base_lr, order='random', random_state=0)
chain.fit(X_train, Y_train).predict(X_test)

chain.predict_proba(X_test)

#machine learning algorithms support multiple outputs directly
#1 linear regression for multioutput regression-----------------
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

# create datasets
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1, noise=0.5)

# define model
model = LinearRegression()
# fit model
model.fit(X, y)

# make a prediction
row = [0.21947749, 0.32948997, 0.81560036, 0.440956, -0.0606303, -0.29257894, -0.2820059, -0.00290545, 0.96402263, 0.04992249]
yhat = model.predict([row])

# summarize prediction
print(yhat[0])

#2 k-nearest neighbors for multioutput regression------------------
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor

# create datasets
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1, noise=0.5)

# define model
model = KNeighborsRegressor()
# fit model
model.fit(X, y)

# make a prediction
row = [0.21947749, 0.32948997, 0.81560036, 0.440956, -0.0606303, -0.29257894, -0.2820059, -0.00290545, 0.96402263, 0.04992249]
yhat = model.predict([row])

# summarize prediction
print(yhat[0])

#3 decision tree for multioutput regression----------------
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor

# create datasets
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1, noise=0.5)

# define model
model = DecisionTreeRegressor()
# fit model
model.fit(X, y)

# make a prediction
row = [0.21947749, 0.32948997, 0.81560036, 0.440956, -0.0606303, -0.29257894, -0.2820059, -0.00290545, 0.96402263, 0.04992249]
yhat = model.predict([row])

# summarize prediction
print(yhat[0])

#Multioutput sklearn-Wrapper Approach----------------------------------
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR

# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1, noise=0.5)

# define base model
model = LinearSVR()
#define the multioutput wrapper model
wrapper = MultiOutputRegressor(model)
# fit the model
wrapper.fit(X, y)

# make a single prediction
row = [0.21947749, 0.32948997, 0.81560036, 0.440956, -0.0606303, -0.29257894, -0.2820059, -0.00290545, 0.96402263, 0.04992249]
yhat = wrapper.predict([row])

# summarize the prediction
print('Predicted: %s' % yhat[0])

#Regressor Chain-----------------------------------------------------
from numpy import mean
from numpy import std
from numpy import absolute
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.multioutput import RegressorChain
from sklearn.svm import LinearSVR

# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, n_targets=2, random_state=1, noise=0.5)

# define base model
model = LinearSVR()
# define the chain model
wrapper = RegressorChain(model)

# define the evaluation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model and collect the scores
n_scores = cross_val_score(wrapper, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force the scores to be positive
n_scores = absolute(n_scores)

# summarize performance
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

