import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import tensorflow as tf
import keras
from keras import layers, callbacks


red_wine = pd.read_csv('red-wine.csv')

# Create training and validation splits
df_train = red_wine.sample(frac=0.7, random_state=0)
df_valid = red_wine.drop(df_train.index)
display(df_train.head(4))


# Scale to [0, 1]
max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)
df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)

# Split features and target
X_train = df_train.drop('quality', axis=1)
X_valid = df_valid.drop('quality', axis=1)
y_train = df_train['quality']
y_valid = df_valid['quality']

early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,  # minimium amount of change to count as an improvement
    patience=20,      # how many epochs to wait before stopping
    restore_best_weights=True,
)

# model = keras.Sequential([
#     keras.Input(shape=(11,)),
#     layers.Dense(512, activation='relu'),
#     layers.Dense(512, activation='relu'),
#     layers.Dense(512, activation='relu'),
#     layers.Dense(1),
# ])

model = keras.Sequential([
    keras.Input(shape=(11,)),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1),
])

model.compile(optimizer='adam', loss='mae')
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=100,
    callbacks=[early_stopping],  # put your callbacks in a list
    verbose=0,                   # turn off training log
)

# convert the training history to a dataframe
history_df = pd.DataFrame(history.history)
# Use pandas native plot method
history_df.loc[:, ['loss', 'val_loss']].plot()
print(f"Minimum validation loss: {history_df['val_loss'].min()}")

plt.show()