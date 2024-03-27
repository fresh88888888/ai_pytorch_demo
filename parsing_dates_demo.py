# modules we'll use
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# read in our data
landslides = pd.read_csv("catalog.csv")

# set seed for reproducibility
np.random.seed(0)

print(landslides.head())
print(landslides['date'].head())
print(landslides['date'].dtype)

landslides['date_parsed'] = pd.to_datetime(landslides['date'], format="%m/%d/%y")
print(landslides['date_parsed'].head())

# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
print(day_of_month_landslides.head())

# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.histplot(day_of_month_landslides, kde=False, bins=31)

plt.show()