import matplotlib.lines as mlines
from sklearn.covariance import EmpiricalCovariance, MinCovDet
import matplotlib.pyplot as plt
import numpy as np
from numpy._typing._generic_alias import NDArray

# for consistent results
np.random.seed(7)

n_samples = 125
n_outliers = 25
n_features = 2

# generate Gaussian data of shape (125, 2)
gen_cov = np.eye(n_features)
gen_cov[0, 0] = 2.0
X = np.dot(np.random.randn(n_samples, n_features), gen_cov)
# add some outliers
outliers_cov = np.eye(n_features)
outliers_cov[np.arange(1, n_features), np.arange(1, n_features)] = 7.0
X[-n_outliers:] = np.dot(np.random.randn(n_outliers, n_features), outliers_cov)

# fit a MCD robust estimator to data
robust_cov = MinCovDet().fit(X)
# fit a MLE estimator to data
emp_cov = EmpiricalCovariance().fit(X)
print("Estimated covariance matrix:\nMCD (Robust):\n{}\nMLE:\n{}".format(robust_cov.covariance_, emp_cov.covariance_))

# fig, ax = plt.subplots(figsize=(10, 5))
# # Plot data set
# inlier_plot = ax.scatter(X[:, 0], X[:, 1], color="black", label="inliers")
# outlier_plot = ax.scatter(
#     X[:, 0][-n_outliers:], X[:, 1][-n_outliers:], color="red", label="outliers"
# )
# ax.set_xlim(ax.get_xlim()[0], 10.0)
# ax.set_title("Mahalanobis distances of a contaminated data set")

# # Create meshgrid of feature 1 and feature 2 values
# xx, yy = np.meshgrid(
#     np.linspace(plt.xlim()[0], plt.xlim()[1], 100),
#     np.linspace(plt.ylim()[0], plt.ylim()[1], 100),
# )
# zz = np.c_[xx.ravel(), yy.ravel()]
# # Calculate the MLE based Mahalanobis distances of the meshgrid
# mahal_emp_cov = emp_cov.mahalanobis(zz)
# mahal_emp_cov = mahal_emp_cov.reshape(xx.shape)
# emp_cov_contour = plt.contour(
#     xx, yy, np.sqrt(mahal_emp_cov), cmap=plt.cm.PuBu_r, linestyles="dashed"
# )
# # Calculate the MCD based Mahalanobis distances
# mahal_robust_cov = robust_cov.mahalanobis(zz)
# mahal_robust_cov = mahal_robust_cov.reshape(xx.shape)
# robust_contour = ax.contour(
#     xx, yy, np.sqrt(mahal_robust_cov), cmap=plt.cm.YlOrBr_r, linestyles="dotted"
# )

# # Add legend
# ax.legend(
#     [
#         mlines.Line2D([], [], color="tab:blue", linestyle="dashed"),
#         mlines.Line2D([], [], color="tab:orange", linestyle="dotted"),
#         inlier_plot,
#         outlier_plot,
#     ],
#     ["MLE dist", "MCD dist", "inliers", "outliers"],
#     loc="upper right",
#     borderaxespad=0,
# )

# plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2)
plt.subplots_adjust(wspace=0.6)

# Calculate cubic root of MLE Mahalanobis distances for samples
emp_mahal = emp_cov.mahalanobis(X - np.mean(X, 0)) ** (0.33)
# Plot boxplots
ax1.boxplot([emp_mahal[:-n_outliers], emp_mahal[-n_outliers:]], widths=0.25)
# Plot individual samples
ax1.plot(
    np.full(n_samples - n_outliers, 1.26),
    emp_mahal[:-n_outliers],
    "+k",
    markeredgewidth=1,
)
ax1.plot(np.full(n_outliers, 2.26), emp_mahal[-n_outliers:], "+k", markeredgewidth=1)
ax1.axes.set_xticklabels(("inliers", "outliers"), size=15)
ax1.set_ylabel(r"$\sqrt[3]{\rm{(Mahal. dist.)}}$", size=16)
ax1.set_title("Using non-robust estimates\n(Maximum Likelihood)")

# Calculate cubic root of MCD Mahalanobis distances for samples
robust_mahal = robust_cov.mahalanobis(X - robust_cov.location_) ** (0.33)
# Plot boxplots
ax2.boxplot([robust_mahal[:-n_outliers], robust_mahal[-n_outliers:]], widths=0.25)
# Plot individual samples
ax2.plot(
    np.full(n_samples - n_outliers, 1.26),
    robust_mahal[:-n_outliers],
    "+k",
    markeredgewidth=1,
)
ax2.plot(np.full(n_outliers, 2.26), robust_mahal[-n_outliers:], "+k", markeredgewidth=1)
ax2.axes.set_xticklabels(("inliers", "outliers"), size=15)
ax2.set_ylabel(r"$\sqrt[3]{\rm{(Mahal. dist.)}}$", size=16)
ax2.set_title("Using robust estimates\n(Minimum Covariance Determinant)")

plt.show()
