# First, we'll import pandas, a data processing and CSV file I/O library
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

from pandas.plotting import radviz
from pandas.plotting import andrews_curves
from pandas.plotting import parallel_coordinates

warnings.filterwarnings("ignore")
sns.set_color_codes("deep")

# Next, we'll load the Iris flower dataset.
iris = pd.read_csv("iris.csv")

# Let's see what's in the iris data
print(iris.head())

# Let's see how many examples we have of each species
print(iris["Species"].value_counts())

# The first way we can plot things is using the .plot extension from Pandas dataframes
# We'll use this to make a scatterplot of the Iris features.
# iris.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")

# We can also use the seaborn library to make a similar plot
# A seaborn jointplot shows bivariate scatterplots and univariate histograms in the same figure
# sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris)

# One piece of information missing in the plots above is what species each plant is
# We'll use seaborn's FacetGrid to color the scatterplot by species
# sns.FacetGrid(iris, hue="Species").map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend()

# We can look at an individual feature in Seaborn through a boxplot
# sns.boxplot(x="Species", y="PetalLengthCm", data=iris)

# One way we can extend this plot is adding a layer of individual points on top of
# it through Seaborn's striplot
# ax = sns.boxenplot(x="Species", y="PetalLengthCm", data=iris)
# ax = sns.stripplot(x="Species", y="PetalLengthCm", data=iris, jitter=True, edgecolor="gray")

# A violin plot combines the benefits of the previous two plots and simplifies them
# Denser regions of the data are fatter, and sparser thiner in a violin plot
# sns.violinplot(x="Species", y="PetalLengthCm", data=iris)

# A final seaborn plot useful for looking at univariate relations is the kdeplot,
# which creates and visualizes a kernel density estimate of the underlying feature
# sns.FacetGrid(iris, hue="Species").map(sns.kdeplot, "PetalLengthCm").add_legend()

# Another useful seaborn plot is the pairplot, which shows the bivariate relation
# between each pair of features
#
# From the pairplot, we'll see that the Iris-setosa species is separataed from the other
# two across all feature combinations
# sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3)

# The diagonal elements in a pairplot show the histogram by default
# We can update these elements to show other things, such as a kde
# sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3, diag_kind="auto")

# Now that we've covered seaborn, let's go back to some of the ones we can make with Pandas
# We can quickly make a boxplot with Pandas on each feature split out by species
# iris.drop("Id", axis=1).boxplot(by="Species", figsize=(12, 6))

# One cool more sophisticated technique pandas has available is called Andrews Curves
# Andrews Curves involve using attributes of samples as coefficients for Fourier series
# and then plotting these
# andrews_curves(iris.drop("Id", axis=1), "Species")

# Another multivariate visualization technique pandas has is parallel_coordinates
# Parallel coordinates plots each feature on a separate column & then draws lines
# connecting the features for each data sample
# parallel_coordinates(iris.drop("Id", axis=1), "Species")

# A final multivariate visualization technique pandas has is radviz
# Which puts each feature as a point on a 2D plane, and then simulates
# having each sample attached to those points through a spring weighted
# by the relative value for that feature
radviz(iris.drop("Id", axis=1), "Species")
plt.show()
