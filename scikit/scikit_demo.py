import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


wine_data = load_wine()
# Convert data to pandas dataframe
wine_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)

# Add the target label
wine_df['target'] = wine_data.target

# Take a preview
print(wine_df.head())
wine_df.info()
print(wine_df.describe())
print(wine_df.tail())

print('---------------------------------')

# Split data into features and label
X = wine_df[wine_data.feature_names].copy()
Y = wine_df['target'].copy()

# Instantiate scaler and fit on features
scaler = StandardScaler()
scaler.fit(X)

# Transform features
X_scaled = scaler.transform(X.values)

# View first instance
print(X_scaled[0])

print('---------------------------------')

# Split data into train and test
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
    X_scaled, Y, train_size=.7, random_state=25)
print(f'Train size: {round(len(X_train_scaled) / len(X) * 100)}% Test size: {round(len(X_test_scaled) / len(X) * 100)}%')

print('---------------------------------')

# Instantiating the models
logistic_regression = LogisticRegression()
svm = SVC()
tree = DecisionTreeClassifier()

# Training the models
logistic_regression.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)
tree.fit(X_train_scaled, y_train)

# Making predictions with each model
log_reg_preds = logistic_regression.predict(X_test_scaled)
svm_preds = svm.predict(X_test_scaled)
tree_preds = tree.predict(X_test_scaled)

# Store model predictions in a dictionary, This makes it's easier to interate through each model and print the results.
model_preds = {'Logistic Regression': log_reg_preds,
               'Support Vector Machine': svm_preds, 'Decision Tree': tree_preds}

for model, preds in model_preds.items():
    print(f'{model} Results:\n{classification_report(y_test, preds)}', sep='\n\n')
    
