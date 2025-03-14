import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.covariance import GraphicalLassoCV, ledoit_wolf

n_samples = 60
n_features = 20

prng = np.random.RandomState(1)
prec = make_sparse_spd_matrix(n_features, alpha=0.98, smallest_coef=0.4, largest_coef=0.7, random_state=prng)
cov = linalg.inv(prec)
d = np.sqrt(np.diag(cov))
cov /= d
cov /= d[:, np.newaxis]
prec *= d
prec *= d[:, np.newaxis]
X = prng.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
X -= X.mean(axis=0)
X /= X.std(axis=0)

# 协方差估计
emp_cov = np.dot(X.T, X) / n_samples
model = GraphicalLassoCV()
model.fit(X)
cov_ = model.covariance_
prec_ = model.precision_
lw_cov_, _ = ledoit_wolf(X)
lw_prec_ = linalg.inv(lw_cov_)

plt.figure(figsize=(10, 6))
plt.subplots_adjust(left=0.02, right=0.98)

# plot the covariances
covs = [("Empirical", emp_cov), ("Ledoit-Wolf", lw_cov_), ("GraphicalLassoCV", cov_), ("True", cov),]
vmax = cov_.max()
for i, (name, this_cov) in enumerate(covs):
    plt.subplot(2, 4, i + 1)
    plt.xticks(())
    plt.yticks(())
    plt.title("%s covariance" % name)
    plt.imshow(this_cov, interpolation="nearest", vmin=-vmax, vmax=vmax, cmap=plt.cm.RdBu_r)

# plot the precisions
precs = [("Empirical", linalg.inv(emp_cov)), ("Ledoit-Wolf", lw_prec_), ("GraphicalLasso", prec_), ("True", prec),]
vmax = 0.9 * prec_.max()
for i, (name, this_prec) in enumerate(precs):
    ax = plt.subplot(2, 4, i + 5)
    plt.xticks(())
    plt.yticks(())
    plt.imshow(np.ma.masked_equal(this_prec, 0), interpolation="nearest", vmin=-vmax, vmax=vmax, cmap=plt.cm.RdBu_r,)
    plt.title("%s precision" % name)
    if hasattr(ax, "set_facecolor"):
        ax.set_facecolor(".7")
    else:
        ax.set_axis_bgcolor(".7")

plt.show()
