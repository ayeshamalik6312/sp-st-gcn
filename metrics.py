import numpy as np
from scipy.special import rel_entr
from scipy.stats import spearmanr, pearsonr, wasserstein_distance
from sklearn.metrics import mean_squared_error, mean_absolute_error
from numpy.linalg import norm

def mse(y_true, y_pred): return mean_squared_error(y_true, y_pred)
def mae(y_true, y_pred): return mean_absolute_error(y_true, y_pred)

def ccc(y_true, y_pred):
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true, var_pred = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true-mean_true)*(y_pred-mean_pred))
    return 2*cov / (var_true+var_pred+(mean_true-mean_pred)**2+1e-8)

def jsd(p, q):
    m = 0.5 * (p + q)
    return 0.5 * np.sum(rel_entr(p, m)) + 0.5 * np.sum(rel_entr(q, m))

def hellinger(p, q):
    return (1/np.sqrt(2)) * norm(np.sqrt(p) - np.sqrt(q))

def evaluate_all(P, Q):
    results = {}
    results["MSE"] = mse(P, Q)
    results["MAE"] = mae(P, Q)
    results["CCC"] = ccc(P, Q)
    results["RMSE"] = np.sqrt(results["MSE"])
    results["JSD"] = np.mean([jsd(P[i], Q[i]) for i in range(P.shape[0])])
    results["Hellinger"] = np.mean([hellinger(P[i], Q[i]) for i in range(P.shape[0])])
    results["EMD"] = np.mean([wasserstein_distance(P[i], Q[i]) for i in range(P.shape[0])])
    results["Spearman"] = np.nanmean([spearmanr(P[i], Q[i])[0] for i in range(P.shape[0])])
    results["Pearson"] = np.nanmean([pearsonr(P[i], Q[i])[0] for i in range(P.shape[0])])
    results["Cosine"] = np.mean([np.dot(P[i], Q[i])/(norm(P[i])*norm(Q[i])+1e-10) for i in range(P.shape[0])])
    return results
