from sklearn.datasets import load_svmlight_file

from joblib import Memory
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np

from hs_chains import HSChains
from petrels_project import PETRELSProjection
from sklearn.preprocessing import minmax_scale

from sh_project import StreamhashProjection

mem = Memory("./mycache")

@mem.cache
def get_data(data):
    data = load_svmlight_file(data)
    return data[0], data[1]

X, y = get_data("data/spam-sms")

X = np.asarray(X.todense())

print(X.shape)

algorithm = "xstream"

if algorithm == "xstream":
    projector = StreamhashProjection(n_components=100, density=1/3.0, random_state=42)

    projected_X = projector.fit_transform(X)

    projected_X = minmax_scale(projected_X)

    cf = HSChains()

    cf.fit(projected_X)

    anomalyscores = -cf.score(projected_X)
    ap = average_precision_score(y, anomalyscores)
    auc = roc_auc_score(y, anomalyscores)
    print ("xstream: AP =", ap, "AUC =", auc)
