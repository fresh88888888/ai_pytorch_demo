from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
import logging
from sklearn.datasets import fetch_olivetti_faces
from sklearn import cluster, decomposition
from numpy.random import RandomState
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np
from sklearn import datasets, decomposition
from sklearn.decomposition import PCA, KernelPCA

np.random.seed(5)

# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

# fig = plt.figure(1, figsize=(4,3))
# plt.clf()

# ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
# ax.set_position([0, 0, 0.95, 1])

# plt.cla()
# pca = decomposition.PCA(n_components=3)
# pca.fit(X)
# X = pca.transform(X)

# for name, label in [("Setosa", 0), ("Versicolour", 1), ("Virginica", 2)]:
#     ax.text3D(
#         X[y == label, 0].mean(),
#         X[y == label, 1].mean() + 1.5,
#         X[y == label, 2].mean(),
#         name,
#         horizontalalignment="center",
#         bbox=dict(alpha=0.5, edgecolor="w", facecolor="w"),
#     )
# # Reorder the labels to have colors matching the cluster results
# y = np.choose(y, [1, 2, 0]).astype(float)
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral, edgecolor="k")

# ax.xaxis.set_ticklabels([])
# ax.yaxis.set_ticklabels([])
# # ax.zaxis.set_ticklabels([])

# plt.show()

# iris = load_iris()
# X = iris.data
# y = iris.target

# n_components = 2
# ipca = IncrementalPCA(n_components=n_components, batch_size=10)
# X_ipca = ipca.fit_transform(X)

# pca = PCA(n_components=n_components)
# X_pca = pca.fit_transform(X)

# colors = ["navy", "turquoise", "darkorange"]

# for X_transformed, title in [(X_ipca, "Incremental PCA"), (X_pca, "PCA")]:
#     plt.figure(figsize=(8, 8))
#     for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
#         plt.scatter(X_transformed[y == i, 0], X_transformed[y == i, 1], color=color, lw=2, label=target_name,)

#     if "Incremental" in title:
#         err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()
#         plt.title(title + " of iris dataset\nMean absolute unsigned error %.6f" % err)
#     else:
#         plt.title(title + " of iris dataset")
#     plt.legend(loc="best", shadow=False, scatterpoints=1)
#     plt.axis([-4, 4, -1.5, 1.5])

# plt.show()

# rng = RandomState(0)

# # Display progress logs on stdout
# logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# # 加载并预处理Olivetti人脸数据集
# faces, _ = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=rng)
# n_samples, n_features = faces.shape

# # Global centering (focus on one feature, centering all samples)
# faces_centered = faces - faces.mean(axis=0)

# # Local centering (focus on one sample, centering all features)
# faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

# # print("Dataset consists of %d faces" % n_samples)
# n_row, n_col = 2, 3
# n_components = n_row * n_col
# image_shape = (64, 64)

# # 定义一个函数来绘制面部图


# def plot_gallery(title, images, n_col=n_col, n_row=n_row, cmap=plt.cm.gray):
#     fig, axs = plt.subplots(nrows=n_row, ncols=n_col, figsize=(2.0 * n_col, 2.3 * n_row), facecolor="white", constrained_layout=True,)
#     fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, hspace=0, wspace=0)
#     fig.set_edgecolor("black")
#     fig.suptitle(title, size=16)
#     for ax, vec in zip(axs.flat, images):
#         vmax = max(vec.max(), -vec.min())
#         im = ax.imshow(vec.reshape(image_shape), cmap=cmap, interpolation="nearest", vmin=-vmax, vmax=vmax,)
#         ax.axis("off")

#     fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.99, aspect=40, pad=0.01)
#     plt.show()


# # plot_gallery("Faces from dataset", faces_centered[:n_components])

# # 使用奇异值分解(SVD)对数据进行线性降维，将其投影到较低维空间
# pca_estimator = decomposition.PCA(n_components=n_components, svd_solver="randomized", whiten=True)
# pca_estimator.fit(faces_centered)
# plot_gallery("Eigenfaces - PCA using randomized SVD", pca_estimator.components_[:n_components])

# # 使用小批量稀疏主成分分析(SPCA)对数据进行先线性降维，将其投影到较低维空间
# batch_pca_estimator = decomposition.MiniBatchSparsePCA(n_components=n_components, alpha=0.1, max_iter=100, batch_size=3, random_state=rng)
# batch_pca_estimator.fit(faces_centered)
# plot_gallery("Sparse components - MiniBatchSparsePCA", batch_pca_estimator.components_[:n_components],)

X, y = make_circles(n_samples=1_000, factor=0.3, noise=0.05, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

# _, (train_ax, test_ax) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 4))

pca = PCA(n_components=2)
kernel_pca = KernelPCA(n_components=None, kernel="rbf", gamma=10, fit_inverse_transform=True, alpha=0.1)
X_test_pca = pca.fit(X_train).transform(X_test)
X_test_kernel_pca = kernel_pca.fit(X_train).transform(X_test)

# fig, (orig_data_ax, pca_proj_ax, kernel_pca_proj_ax) = plt.subplots(ncols=3, figsize=(14, 4))
# orig_data_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
# orig_data_ax.set_ylabel("Feature #1")
# orig_data_ax.set_xlabel("Feature #0")
# orig_data_ax.set_title("Testing data")

# pca_proj_ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test)
# pca_proj_ax.set_ylabel("Principal component #1")
# pca_proj_ax.set_xlabel("Principal component #0")
# pca_proj_ax.set_title("Projection of testing data\n using PCA")

# kernel_pca_proj_ax.scatter(X_test_kernel_pca[:, 0], X_test_kernel_pca[:, 1], c=y_test)
# kernel_pca_proj_ax.set_ylabel("Principal component #1")
# kernel_pca_proj_ax.set_xlabel("Principal component #0")
# kernel_pca_proj_ax.set_title("Projection of testing data\n using KernelPCA")

# plt.show()

X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test))
X_reconstructed_kernel_pca = kernel_pca.inverse_transform(kernel_pca.transform(X_test))
fig, (orig_data_ax, pca_back_proj_ax, kernel_pca_back_proj_ax) = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(13, 4))

orig_data_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
orig_data_ax.set_ylabel("Feature #1")
orig_data_ax.set_xlabel("Feature #0")
orig_data_ax.set_title("Original test data")

pca_back_proj_ax.scatter(X_reconstructed_pca[:, 0], X_reconstructed_pca[:, 1], c=y_test)
pca_back_proj_ax.set_xlabel("Feature #0")
pca_back_proj_ax.set_title("Reconstruction via PCA")

kernel_pca_back_proj_ax.scatter(X_reconstructed_kernel_pca[:, 0], X_reconstructed_kernel_pca[:, 1], c=y_test)
kernel_pca_back_proj_ax.set_xlabel("Feature #0")
kernel_pca_back_proj_ax.set_title("Reconstruction via KernelPCA")
plt.show()
