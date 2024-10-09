from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from sklearn import metrics
from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load the dataset
customer_data = pd.read_csv("Mall_Customers.csv")

# read the data
print(customer_data.head())
# check for null or missing values

print(customer_data.isna().sum())

# plt.scatter(customer_data['Annual_Income_(k$)'], customer_data['Spending_Score'])
# plt.xlabel('Annual_Income_(k$)')
# plt.ylabel('Spending_Score')
# plt.show()

# plt.scatter(customer_data['Annual_Income_(k$)'], customer_data['Spending_Score'])
# plt.scatter(centroids['Annual_Income_(k$)'], centroids['Spending_Score'], c='black')
# plt.xlabel('Annual_Income_(k$)')
# plt.ylabel('Spending_Score')
# plt.show()

# K = 3
# centroids = customer_data.sample(n=K)
# mask = customer_data['CustomerID'].isin(centroids.CustomerID.tolist())
# X = customer_data[~mask]
# diff = 1
# j = 0
# XD = X
# while (diff != 0):
#     i = 1
#     for index1, row_c in centroids.iterrows():
#         ED = []
#         for index2, row_d in XD.iterrows():
#             d1 = (row_c["Annual_Income_(k$)"] - row_d["Annual_Income_(k$)"])**2
#             d2 = (row_c["Spending_Score"] - row_d["Spending_Score"])**2
#             d = np.sqrt(d1 + d2)
#             ED.append(d)
#         X[i] = ED
#         i = i + 1

#     C = []
#     for index, row in X.iterrows():
#         min_dist = row[1]
#         pos = 1
#         for i in range(K):
#             if row[i + 1] < min_dist:
#                 min_dist = row[i + 1]
#                 pos = i + 1
#         C.append(pos)
#     X["Cluster"] = C
#     centroids_new = X.groupby(["Cluster"]).mean()[["Spending_Score", "Annual_Income_(k$)"]]
#     if j == 0:
#         diff = 1
#         j = j + 1
#     else:
#         diff = (centroids_new['Spending_Score'] - centroids['Spending_Score']).sum() + (centroids_new['Annual_Income_(k$)'] - centroids['Annual_Income_(k$)']).sum()
#     centroids = X.groupby(["Cluster"]).mean()[["Spending_Score", "Annual_Income_(k$)"]]

# color = ['grey', 'blue', 'orange']
# for k in range(K):
#     data = X[X["Cluster"] == (k + 1)]
#     plt.scatter(data["Annual_Income_(k$)"], data["Spending_Score"], c=color[k])
# plt.scatter(centroids["Annual_Income_(k$)"], centroids["Spending_Score"], c='black')
# plt.xlabel('Annual_Income_(k$)')
# plt.ylabel('Spending_Score')
# plt.show()

# km_sample = KMeans(n_clusters=3)
# km_sample.fit(customer_data[['Annual_Income_(k$)', 'Spending_Score']])

# labels_sample = km_sample.labels_
# customer_data['label'] = labels_sample
# sns.scatterplot(x=customer_data['Annual_Income_(k$)'], y=customer_data['Spending_Score'], hue=customer_data['label'], palette='Set1')
# plt.show()

x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])
x2 = np.array([5, 4, 5, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3])

X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)

# # k means determine k
# distortions = []
# K = range(1, 10)
# for k in K:
#     kmeanModel = KMeans(n_clusters=k).fit(X)
#     kmeanModel.fit(X)
#     distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# # Plot the elbow
# plt.plot(K, distortions, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method showing the optimal k')
# plt.show()


sil_avg = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for k in range_n_clusters:
    kmeans = KMeans(n_clusters=k).fit(X)
    labels = kmeans.labels_
    sil_avg.append(silhouette_score(X, labels, metric='euclidean'))

# 绘制每个K值的轮廓分数集合, 选择轮廓得分最大时的聚类数量：
plt.plot(range_n_clusters, sil_avg, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Silhouette score')
plt.title('Silhouette analysis For Optimal k')
plt.show()
