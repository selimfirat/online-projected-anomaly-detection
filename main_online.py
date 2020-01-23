"""Online Anomaly Detection via Subspace Tracking

Usage:
    main_online.py   [--window_percentage=<integer_value>] [--projection=<string_value>] [--algorithm=<string_value>] [--projection_dimensions=<integer_value] [--nchains=<integer_value>] [--depth=<integer_value>] [--data=<string_value>]

Options:
    --window_percentage=<integer_value> Window percentage [default: 100].
    --projection Type of the projection "streamhash" "petrels" "grouse" [default: streamhash].
    --algorithm Name of the algorithm [default: hsc].
    --projection_dimensions Projection dimensions [default: 50].
    --nchains Projection dimensions [default: 100].
    --depth Projection dimensions [default: 25].
    --data path to the data [default: data/spam-sms]
"""
import random

from docopt import docopt
from sklearn.datasets import load_svmlight_file

from joblib import Memory
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import minmax_scale

from algorithm_hsc import HSChains
from projection_grouse import GrouseProjection
from projection_streamhash import StreamhashProjection

random_state = 42

np.random.seed(random_state)
random.seed(random_state)

args = docopt(__doc__, version='Online Anomaly Detection via Subspace Tracking 1.0')

args["--window_percentage"] = float(args.get("--window_percentage"))
args["--projection_dimensions"] = int(args.get("--projection_dimensions"))
args["--depth"] = int(args.get("--depth"))
args["--nchains"] = int(args.get("--nchains"))

mem = Memory("./mycache")


@mem.cache
def get_data(data):
    data = load_svmlight_file(data)
    return data[0], data[1]


@mem.cache
def get_projected_X(X, data, projection, projection_dimension):
    if projection == "streamhash":
        projector = StreamhashProjection(n_components=projection_dimension, density=1 / 3.0, random_state=random_state)
        X = projector.fit_transform(X)
    elif projection == "grouse":
        projector = GrouseProjection(n_components=projection_dimension, random_state=random_state)
        X = projector.fit_transform(X)

    return X


X, y = get_data("data/" + args["--data"])

num_rows = X.shape[0]
num_features = X.shape[1]

window_size = int(np.round(1.0 * num_rows * args["--window_percentage"] / 100.0))

X = np.asarray(X.todense())

print(args)

# Projection

X = get_projected_X(X, args["--data"], args["--projection"], args["--projection_dimension"])

X = minmax_scale(X)

if args["--algorithm"] == "hsc":
    algorithm = HSChains(k=X.shape[1], nchains=args["--nchains"], depth=args["--depth"])

y_pred = np.empty((X.shape[0],))

last_window_idx = 0
for i in tqdm(range(num_rows), desc="Running Algorithm..."):

    x = np.expand_dims(X[i], axis=0)

    algorithm.partial_fit(x)

    y_pred[i] = -algorithm.score(x)

    if ((i + 1) % window_size) == 0 or i + 1 == num_rows:

        algorithm.next_window()

        print("Window AP:")

        ap = average_precision_score(y[last_window_idx:i+1], y_pred[last_window_idx:i+1])
        auc = roc_auc_score(y[last_window_idx:i+1], y_pred[last_window_idx:i+1])

        print(f"Window AP: {ap}; AUC: {auc} (at index {i+1})")
        last_window_idx = i + 1


oap = average_precision_score(y, y_pred)
auc = roc_auc_score(y, y_pred)
print(f"Overall AP: {oap}, AUC: {auc}")