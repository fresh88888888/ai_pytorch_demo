
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from warnings import simplefilter
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess


def fourier_features(index, freq, order):
    time = np.arange(len(index), dtype=np.float32)
    k = 2 * np.pi * (1 / freq) * time
    features = {}
    for i in range(1, order + 1):
        features.update({
            f'sin_{freq}_{i}': np.sin(i * k),
            f'cos_{freq}_{i}': np.cos(i * k),
        })
    return pd.DataFrame(features, index=index)

# Compute Fourier features to the 4th order (8 new features) for a
# series y with daily observations and annual seasonality:

# print(fourier_features(y, freq=365.25, order=4).head())


simplefilter('ignore')

plt.rc("figure", autolayout=True, figsize=(11, 5))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)

def seasonal_plot(X, y, period, freq, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    palette = sns.color_palette("husl", n_colors=X[period].nunique(),)
    ax = sns.lineplot(x=freq, y=y, hue=period, data=X, ci=False, ax =ax, palette=palette, legend=False)
    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    for line, name in zip(ax.lines, X[period].unique()):
        y_= line.get_ydata()[-1]
        ax.annotate(name, xy=(1, y_), xytext=(6, 0), color=line.get_color(), 
                    xycoords=ax.get_yaxis_transform(), textcoords="offset points", size=14, va="center")
    return ax

def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = pd.Timedelta("365D") / pd.Timedelta('1D')
    freqencies, spectrum = periodogram(ts, fs=fs, detrend=detrend, window="boxcar", scaling='spectrum',)
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    
    return ax


tunnel = pd.read_csv('tunnel.csv', parse_dates=['Day'])
tunnel = tunnel.set_index('Day').to_period('D')

X = tunnel.copy()

# days within a week
X["day"] = X.index.dayofweek  # the x-axis (freq)
X["week"] = X.index.week  # the seasonal period (period)

# days within a year
X["dayofyear"] = X.index.dayofyear
X["year"] = X.index.year

# fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 6))
# seasonal_plot(X, y='NumVehicles', period='week', freq='day', ax=ax0)
# seasonal_plot(X, y='NumVehicles', period='year', freq='dayofyear', ax=ax1)
# plot_periodogram(tunnel.NumVehicles)

# 10 sin/cos pairs for "A" annual seasonality
fourier = CalendarFourier(freq="A", order=10)
dp = DeterministicProcess(
    index=tunnel.index, 
    # dummy feature for bias (y-intercept)
    constant=True,
    # trend (order 1 means linear)
    order=1,
    # weekly seasonality (indicators)
    seasonal=True,
    # annual seasonality (fourier)
    additional_terms=[fourier],
    # drop terms to avoid collinearity
    drop=True,                   
)
X = dp.in_sample()  # create features for dates in tunnel.index
y = tunnel['NumVehicles']

model = LinearRegression(fit_intercept=False)
_ = model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=y.index)
X_fore = dp.out_of_sample(steps=90)
y_fore = pd.Series(model.predict(X_fore), index = X_fore.index)

ax = y.plot(color='0.25', style='.', title='Tunnel Traffic - Seasonal Forecast')
ax = y_pred.plot(ax=ax, label='Seasonal')
ax = y_fore.plot(ax=ax, label='Seasonal Forecast', color='C3')
_ = ax.legend()

plt.show()
