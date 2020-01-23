import scipy
import numpy as np
from joblib import Memory

from sklearn.datasets import load_svmlight_file

mem = Memory("./mycache")


@mem.cache
def get_data(data):
    data = load_svmlight_file(data)
    return data[0], data[1]


X, y = get_data("data/spam-sms")

X = np.asarray(X.todense())

scipy.io.savemat('data/spam-sms.mat', dict(X=X, y=y.flatten()))
