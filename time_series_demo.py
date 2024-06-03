import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from warnings import simplefilter
from sklearn.linear_model import LinearRegression

simplefilter('ignore')  # ignore warnings to clean up output cells

plt.rc("figure", autolayout=True, figsize=(11, 4))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)

# Load Tunnel Traffic dataset
tunnel = pd.read_csv('tunnel.csv', parse_dates=['Day'])

# Create a time series in Pandas by setting the index to a date column. We parsed 'Day' as a date type
# by using `parse_dates` when loading the date.
tunnel = tunnel.set_index('Day')

# By default, Pandas creates a `DatetimeIndex` with dtype `Timestamp`
# (equivalent to `np.datetime64`, representing a time series as a
# sequence of measurements taken at single moments. A `PeriodIndex`,
# on the other hand, represents a time series as a sequence of
# quantities accumulated over periods of time. Periods are often
# easier to work with, so that's what we'll use in this course.
tunnel = tunnel.to_period()
print(tunnel.head())

# The time step feature
df = tunnel.copy()
df['Time'] = np.arange(len(tunnel.index))
print(df.head())

# Training data
X = df.loc[:, ['Time']]  # features
y = df.loc[:, 'NumVehicles']  # target

# Train the model
model = LinearRegression()
model.fit(X, y)

# Store the fitted values as a time series with the same time index as the training data
y_pred = pd.Series(model.predict(X=X), index=X.index)

ax = y.plot(**plot_params)
ax = y_pred.plot(ax=ax, linewidth=3)
ax.set_title('Time Plot of Tunnel Traffic')

# The lag feature
df['Lag_1'] = df['NumVehicles'].shift(1)
print(df.head())

X = df.loc[:, ['Lag_1']]
X.dropna(inplace=True)  # drop missing values in the feature set
y = df.loc[:, 'NumVehicles']  # create the target
y, X = y.align(X, join='inner')  # drop corresponding values in target

model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)

fig, ax = plt.subplots()
ax.plot(X['Lag_1'], y, '.', color='0.25')
ax.plot(X['Lag_1'], y_pred)
ax.set_aspect('equal')
ax.set_ylabel('NumVehicles')
ax.set_xlabel('Lag_1')
ax.set_title('Lag Plot of Tunnel Traffic')

ax = y.plot(**plot_params)
ax = y_pred.plot()

plt.show()
