import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

df = pd.read_csv('concrete.csv')
print(df.head())

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
