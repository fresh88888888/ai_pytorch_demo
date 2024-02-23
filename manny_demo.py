import warnings
import seaborn as sns
import xgboost as xgb
import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load diamonds_special of data.
dataset_link = "https://raw.githubusercontent.com/BexTuychiev/medium_stories/master/2024/1_january/4_intro_to_nannyml/diamonds_special.csv"
diamonds_special = pd.read_csv(dataset_link)
print(diamonds_special.head())

# Extract all feature names
all_feature_names = diamonds_special.drop(
    ['price', 'set'], axis=1).columns.tolist()
# Extract the columns and cast into category
cats = diamonds_special.select_dtypes(exclude=np.number).columns
# Define the target column
target = 'price'

for col in cats:
    diamonds_special[col] = diamonds_special[col].astype('category')

tr = diamonds_special[diamonds_special.set == 'train'].drop('set', axis=1)
val = diamonds_special[diamonds_special.set ==
                       'validation'].drop('set', axis=1)
test = diamonds_special[diamonds_special.set == 'test'].drop('set', axis=1)
prod = diamonds_special[diamonds_special.set == 'prod'].drop('set', axis=1)

print(tr.shape)


def split_into_four(df, train_size=0.7):
    """
    A function to split a dataset into four sets:
    - Training
    - Validation
    - Testing
    - Production
    train_size is set by the user.
    The 
    """
    # Do the splits
    training, the_rest = train_test_split(df, train_size=train_size)
    validation, the_rest = train_test_split(the_rest, train_size=1/3)
    testing,production = train_test_split(the_rest, train_size=0.5)
    
    # Reset thr indices
    sets = (training, validation, testing, production)
    for set in sets:
        set.reset_index(inplace= True, drop= True)
    
    return sets


dtrain = xgb.DMatrix(tr[all_feature_names],label=tr[target], enable_categorical=True)
dval = xgb.DMatrix(val[all_feature_names], label=tr[target], enable_categorical=True)
dtest = xgb.DMatrix(test[all_feature_names], label=tr[target], enable_categorical=True)
dprod = xgb.DMatrix(prod[all_feature_names], label=tr[target], enable_categorical=True)

# # Define optimized parameters
# params = {
#     "n_estimators": 10000,
#     "learning_rate": 0.1,
#     "tree_method": "gpu_hist",
#     "max_depth": 6,
#     "min_child_weight": 1,
#     "gamma": 0,
#     "subsample": 0.8,
#     "colsample_bytree": 0.8,
#     "objective": "reg:squarederror",
#     "reg_alpha": 0.01,
#     "reg_lambda": 1,
# }

# # Training with early stopping
# regressor = xgb.train(
#     params,
#     dtrain,
#     num_boost_round=10000,
#     evals=[(dtrain, "train"), (dval, "eval")],
#     early_stopping_rounds=50,
#     verbose_eval=500,
# )

import nannyml

reference = test.copy(deep=True)
analysis = prod.copy(deep=True)

estimator = nannyml.DLE(
    feature_column_names=all_feature_names,
    y_true=target,
    y_pred="y_pred",
    metrics=["rmse"],
    chunk_size=250,
)

# Fit to the reference set
estimator.fit(reference)
# Estimate on the analysis set
extimate_results = estimator.estimate(analysis)
extimate_results.plot().show()

