import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns


# # Path of the file to read
# fifa_filepath = 'fifa.csv'
# # Read the file into a variablefaia_data
# fifa_data = pd.read_csv(fifa_filepath, index_col='Date', parse_dates=True)
# # print(fifa_data.head())

# # Set the width and weight of the figure
# plt.figure(figsize=(16, 6))
# # Line chart showing how FIFA rankings evoloed over time
# print(list(fifa_data.columns))
# # Add title
# plt.title("FIFA ranking evoloed over time")

# sns.lineplot(data=fifa_data['ARG'])
# sns.lineplot(data=fifa_data['BRA'])
# plt.xlabel('Date')

# flight_filepath = 'flight_delays.csv'
# flight_data = pd.read_csv(flight_filepath, index_col='Month')

# plt.figure(figsize=(10, 6))
# plt.title("Average Arrival Delay for Spirit Airlines Flights, by Month")
# sns.barplot(x=flight_data.index, y=flight_data['NK'])
# plt.ylabel("Arrival delay (in minutes)")

# plt.figure(figsize=(14, 7))
# plt.title("Average Arrival Delay for Each Airline, by Month")
# sns.heatmap(flight_data, annot=True)
# plt.xlabel("Airline")

# insurance_filepath = 'insurance.csv'
# insurance_data = pd.read_csv(insurance_filepath)

# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])
# sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges'])
# sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker'])
# sns.lmplot(x='bmi', y='charges', hue='smoker', data=insurance_data)
# sns.swarmplot(x=insurance_data['smoker'], y=insurance_data['charges'])

iris_filepath = 'iris.csv'
iris_data = pd.read_csv(iris_filepath, index_col='Id')

plt.figure(figsize=(10, 6))
# sns.histplot(data=iris_data['Petal Length (cm)'])
# sns.kdeplot(data=iris_data['Petal Length (cm)'], shade=True)
# sns.jointplot(x=iris_data['Petal Length (cm)'], y=iris_data['Sepal Width (cm)'], kind='kde')

# plt.title('Histogram of Petal Lengths, by Species')
# sns.histplot(data=iris_data, x='Petal Length (cm)', hue='Species')
plt.title('Distribution of Petal Lengths, by Species')
sns.kdeplot(data=iris_data, x='Petal Length (cm)', hue='Species', shade=True)

plt.show()
