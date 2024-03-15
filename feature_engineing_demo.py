import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA

df = pd.read_csv('concrete.csv')
# print(df.head())

X = df.copy()
y = X.pop('CompressiveStrength')

# Create synthetic features
X["FCRatio"] = X["FineAggregate"] / X["CoarseAggregate"]
X["AggCmtRatio"] = (X["CoarseAggregate"] + X["FineAggregate"]) / X["Cement"]
X["WtrCmtRatio"] = X["Water"] / X["Cement"]

# Train and score baseline model
baseline = RandomForestRegressor(criterion='absolute_error', random_state=0)
baseline_score = cross_val_score(baseline, X=X, y=y, cv=5, scoring="neg_mean_absolute_error")
baseline_score = -1 * baseline_score.mean()

# print(f'MAE baseline score: {baseline_score:.4}')
print(f"MAE Score with Ratio Features: {baseline_score:.4}")


df = pd.read_csv('autos.csv')
X = df.copy()
y = X.pop('price')

# Label encoding for categoricals
for colname in X.select_dtypes('object'):
    X[colname], _ = X[colname].factorize()

# All discrete features should now have integer dtypes (double-check this before using MI!)
discrete_features = X.dtypes == int


from sklearn.feature_selection import mutual_info_regression
import numpy as np

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name='MI Scores', index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    
    return mi_scores

mi_scores = make_mi_scores(X, y, discrete_features=discrete_features)
# print(mi_scores)  # show a few features with their MI scores

def plot_mi_score(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title('Mutual Information Scores')

# plt.figure(dpi=100, figsize=(8, 5))
# plot_mi_score(mi_scores)
# sns.relplot(x="curb_weight", y="price", data=df)
# sns.lmplot(x="horsepower", y="price", hue="fuel_type", data=df)
# plt.show()

customer = pd.read_csv('customer.csv')

customer[['Type', 'Level']] = (
    customer['Policy'].str.split(' ', expand=True)
)
print(customer[['Type', 'Level', 'Policy']].head(10))

# autos = pd.read_csv('autos.csv')
# autos["make_and_style"] = autos["make"] + "_" + autos["body_style"]
# print(autos[["make", "body_style", "make_and_style"]].head())

customer['StateFreq'] = (
    customer.groupby('State')['State'].transform('count') / customer.State.count()
)

print(customer[['State', 'StateFreq']].head(10))

df_train = customer.sample(frac=0.5)
df_valid = customer.drop(df_train.index)

df_train['AverageClaim'] = df_train.groupby('Coverage')['ClaimAmount'].transform('mean')

# Merge the values into the validation set
df_valid = df_valid.merge(
    df_train[['Coverage', 'AverageClaim']].drop_duplicates(),
    on='Coverage',
    how='left',
)

print(df_valid[['Coverage', 'AverageClaim']].head(10))

housing = pd.read_csv('housing.csv')
X = housing.loc[:, ["MedInc", "Latitude", "Longitude"]]

# Create cluster feature
kmeans = KMeans(n_clusters=6)
X['Cluster'] = kmeans.fit_predict(X)
X['Cluster'] = X['Cluster'].astype('category')

# print(X.head())
# sns.relplot(
#     x="Longitude", y="Latitude", hue="Cluster", data=X, height=6,
# )
# X["MedHouseVal"] = housing["MedHouseVal"]
# sns.catplot(x="MedHouseVal", y="Cluster", data=X, kind="boxen", height=6)

# plt.show()


def plot_variance(pca, width=8, dpi=100):
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    return axs


df = pd.read_csv('autos.csv')

features = ["highway_mpg", "engine_size", "horsepower", "curb_weight"]
X = df.copy()
y = X.pop('price')
X = X.loc[:, features]

# Standardize
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

# Create principal components
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Convert to dataframe
component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=component_names)

print(X_pca)

loadings = pd.DataFrame(
    pca.components_.T,  # transpose the matrix of loadings
    columns=component_names,  # so the columns are the principal components
    index=X.columns,  # and the rows are the original features
)
print(loadings)

# Look at explained variance
# plot_variance(pca)
# plt.show()

mi_scores = make_mi_scores(X_pca, y, discrete_features=False)
print(mi_scores)

# Show dataframe sorted by PC3
idx = X_pca["PC3"].sort_values(ascending=False).index
cols = ["make", "body_style", "horsepower", "curb_weight"]
print(df.loc[idx, cols].head(10))

df["sports_or_wagon"] = X.curb_weight / X.horsepower
sns.regplot(x='sports_or_wagon', y='price', data=df, order=2)
plt.show()
