import numpy as np
from sklearn.linear_model import SGDClassifier

from error_tools import *
from grouse import Grouse
from petrels import Petrels
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import normalize
from joblib import Memory
from sklearn.metrics import classification_report
from tqdm import tqdm

mem = Memory("./mycache")

@mem.cache
def get_data(data):
    data = load_svmlight_file(data)
    return data[0], data[1]

X, y = get_data("data/spam-sms")

X = normalize(X)

ob_count = X.shape[0]

print(X.shape)

num_features = X.shape[1]
rank = 100

tracker = Grouse(num_features, rank, 1 ) # Petrels(num_features, rank, 0.98 )

sgd = SGDClassifier()

total = 0.0
clf_err = 0.0
y_pred = []
y_true = []

for i in range(ob_count):

    x = np.array(X[i, :].todense().T)
    y_sample = y[i].reshape(1, )
    ss = np.ones_like(x)

    tracker.consume(x, ss)

    proj = tracker._project(x, ss)

    error = calc_observation_error(x, tracker.U @ proj)

    if i > 0:
        y_pred.append(sgd.predict(proj.T))
        y_true.append(y_sample)

    sgd.partial_fit(proj.T, y_sample, classes=[0, 1])

    total += error

    if i % 50 == 1:
        print(classification_report(y_true, y_pred, labels=[0, 1]))
        print(total/(i+1))
