from transformers import TFBertModel
from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer
from tqdm.notebook import tqdm
import transformers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input
import tensorflow as tf
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import re
import string
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from collections import Counter

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import os
import spacy
import random
from spacy.util import compounding
from spacy.util import minibatch
from collections import defaultdict
from collections import Counter
import keras
from keras.models import Sequential
from keras.initializers import Constant
from keras.layers import (LSTM,
                          Embedding,
                          BatchNormalization,
                          Dense,
                          TimeDistributed,
                          Dropout,
                          Bidirectional,
                          Flatten,
                          GlobalMaxPool1D)
from keras_nlp.tokenizers import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    accuracy_score
)

nltk.download('stopwords')

# Defining all our palette colours.
primary_blue = "#496595"
primary_blue2 = "#85a1c1"
primary_blue3 = "#3f4d63"
primary_grey = "#c6ccd8"
primary_black = "#202022"
primary_bgcolor = "#f4f0ea"
primary_green = px.colors.qualitative.Plotly[2]

# 加载数据
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df.dropna(how="any", axis=1)
df.columns = ['target', 'message']
print(df.head())

df['message_len'] = df['message'].apply(lambda x: len(x.split(' ')))
print(df.head())

balance_counts = df.groupby('target')['target'].agg('count').values
print(balance_counts)

# fig = go.Figure()
# fig.add_trace(go.Bar(
#     x=['ham'],
#     y=[balance_counts[0]],
#     name='ham',
#     text=[balance_counts[0]],
#     textposition='auto',
#     marker_color=primary_blue
# ))
# fig.add_trace(go.Bar(
#     x=['spam'],
#     y=[balance_counts[1]],
#     name='spam',
#     text=[balance_counts[1]],
#     textposition='auto',
#     marker_color=primary_grey
# ))
# fig.update_layout(
#     title='<span style="font-size:32px; font-family:Times New Roman">Dataset distribution by target</span>'
# )
# fig.show()

# ham_df = df[df['target'] == 'ham']['message_len'].value_counts().sort_index()
# spam_df = df[df['target'] == 'spam']['message_len'].value_counts().sort_index()

# fig = go.Figure()
# fig.add_trace(go.Scatter(
#     x=ham_df.index,
#     y=ham_df.values,
#     name='ham',
#     fill='tozeroy',
#     marker_color=primary_blue,
# ))
# fig.add_trace(go.Scatter(
#     x=spam_df.index,
#     y=spam_df.values,
#     name='spam',
#     fill='tozeroy',
#     marker_color=primary_grey,
# ))
# fig.update_layout(
#     title='<span style="font-size:32px; font-family:Times New Roman">Data Roles in Different Fields</span>'
# )
# fig.update_xaxes(range=[0, 70])
# fig.show()


def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


df['message_clean'] = df['message'].apply(clean_text)
print(df.head())

stop_words = stopwords.words('english')
more_stopwords = ['u', 'im', 'c']
stop_words = stop_words + more_stopwords


def remove_stopwords(text):
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    return text


df['message_clean'] = df['message_clean'].apply(remove_stopwords)
print(df.head())

stemmer = nltk.SnowballStemmer("english")


def stemm_text(text):
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))
    return text


df['message_clean'] = df['message_clean'].apply(stemm_text)
print(df.head())


def preprocess_data(text):
    # Clean puntuation, urls, and so on
    text = clean_text(text)
    # Remove stopwords
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    # Stemm all the words in the sentence
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))

    return text


df['message_clean'] = df['message_clean'].apply(preprocess_data)
print(df.head())


le = LabelEncoder()
le.fit(df['target'])

df['target_encoded'] = le.transform(df['target'])
print(df.head())

# twitter_mask = np.array(Image.open('twitter_mask3.jpg'))

# wc = WordCloud(
#     background_color='white',
#     max_words=200,
#     mask=twitter_mask,
# )
# wc.generate(' '.join(text for text in df.loc[df['target'] == 'ham', 'message_clean']))
# plt.figure(figsize=(12, 6))
# plt.title('Top words for HAM messages', fontdict={'size': 22,  'verticalalignment': 'bottom'})
# plt.imshow(wc)
# plt.axis("off")
# plt.show()


# twitter_mask = np.array(Image.open('twitter_mask3.jpg'))

# wc = WordCloud(
#     background_color='white',
#     max_words=200,
#     mask=twitter_mask,
# )
# wc.generate(' '.join(text for text in df.loc[df['target'] == 'spam', 'message_clean']))
# plt.figure(figsize=(12, 6))
# plt.title('Top words for SPAM messages',
#           fontdict={'size': 22,  'verticalalignment': 'bottom'})
# plt.imshow(wc)
# plt.axis("off")
# plt.show()

# how to define X and y (from the SMS data) for use with COUNTVECTORIZER
x = df['message_clean']
y = df['target_encoded']

print(len(x), len(y))

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
print(len(x_train), len(y_train))
print(len(x_test), len(y_test))

# instantiate the vectorizer
vect = CountVectorizer()
vect.fit(x_train)

# Use the trained to create a document-term matrix from train and test sets
x_train_dtm = vect.transform(x_train)
x_test_dtm = vect.transform(x_test)

texts = df['message_clean']
target = df['target_encoded']

# # Calculate the length of our vocabulary
# word_tokenizer = Tokenizer()
# word_tokenizer.fit_on_texts(texts)

# vocab_length = len(word_tokenizer.word_index) + 1
# vocab_length


# def embed(corpus):
#     return word_tokenizer.texts_to_sequences(corpus)


# longest_train = max(texts, key=lambda sentence: len(word_tokenize(sentence)))
# length_long_sentence = len(word_tokenize(longest_train))

# train_padded_sentences = pad_sequences(
#     embed(texts),
#     length_long_sentence,
#     padding='post'
# )

# print(train_padded_sentences)

embeddings_dictionary = dict()
embedding_dim = 100

# # Load GloVe 100D embeddings
# with open('/kaggle/input/glove6b100dtxt/glove.6B.100d.txt') as fp:
#     for line in fp.readlines():
#         records = line.split()
#         word = records[0]
#         vector_dimensions = np.asarray(records[1:], dtype='float32')
#         embeddings_dictionary[word] = vector_dimensions

# # Now we will load embedding vectors of those words that appear in the
# # Glove dictionary. Others will be initialized to 0.

# embedding_matrix = np.zeros((vocab_length, embedding_dim))

# for word, index in word_tokenizer.word_index.items():
#     embedding_vector = embeddings_dictionary.get(word)
#     if embedding_vector is not None:
#         embedding_matrix[index] = embedding_vector

# embedding_matrix

x_axes = ['Ham', 'Spam']
y_axes = ['Spam', 'Ham']


def conf_matrix(z, x=x_axes, y=y_axes):
    z = np.flip(z, 0)
    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]
    # set up figure
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')
    # add title
    fig.update_layout(title_text='<b>Confusion matrix</b>',
                      xaxis=dict(title='Predicted value'), yaxis=dict(title='Real value'))
    # add colorbar
    fig['data'][0]['showscale'] = True

    return fig


# Create a Multinomial Naive Bayes model
nb = MultinomialNB()

# # Train the model
# print(nb.fit(x_train_dtm, y_train))

# # Make class anf probability predictions
# y_pred_class = nb.predict(x_test_dtm)
# y_pred_prob = nb.predict_proba(x_test_dtm)[:, 1]

# # calculate accuracy of class predictions
# print(metrics.accuracy_score(y_test, y_pred_class))
# fig = conf_matrix(metrics.confusion_matrix(y_test, y_pred_class))
# fig.show()


# pipe = Pipeline([('bow', CountVectorizer()), ('tfid', TfidfTransformer()), ('model', MultinomialNB())])

# # Fit the pipeline with the data
# pipe.fit(x_train, y_train)
# y_pred_class = pipe.predict(x_test)
# print(metrics.accuracy_score(y_test, y_pred_class))

# fig = conf_matrix(metrics.confusion_matrix(y_test, y_pred_class))
# fig.show()


# pipe = Pipeline([
#     ('bow', CountVectorizer()),
#     ('tfid', TfidfTransformer()),
#     ('model', xgb.XGBClassifier(
#         learning_rate=0.1,
#         max_depth=7,
#         n_estimators=80,
#         use_label_encoder=False,
#         eval_metric='auc',
#         # colsample_bytree=0.8,
#         # subsample=0.7,
#         # min_child_weight=5,
#     ))
# ])

# # Fit the pipeline with the data
# pipe.fit(x_train, y_train)

# y_pred_class = pipe.predict(x_test)
# y_pred_train = pipe.predict(x_train)

# print('Train: {}'.format(metrics.accuracy_score(y_train, y_pred_train)))
# print('Test: {}'.format(metrics.accuracy_score(y_test, y_pred_class)))

# fig = conf_matrix(metrics.confusion_matrix(y_test, y_pred_class))
# fig.show()

# # Split data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(
#     train_padded_sentences,
#     target,
#     test_size=0.25
# )


# def glove_lstm():
#     model = Sequential()

#     model.add(Embedding(
#         input_dim=embedding_matrix.shape[0],
#         output_dim=embedding_matrix.shape[1],
#         weights=[embedding_matrix],
#         input_length=length_long_sentence
#     ))

#     model.add(Bidirectional(LSTM(
#         length_long_sentence,
#         return_sequences=True,
#         recurrent_dropout=0.2
#     )))

#     model.add(GlobalMaxPool1D())
#     model.add(BatchNormalization())
#     model.add(Dropout(0.5))
#     model.add(Dense(length_long_sentence, activation="relu"))
#     model.add(Dropout(0.5))
#     model.add(Dense(length_long_sentence, activation="relu"))
#     model.add(Dropout(0.5))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

#     return model


# model = glove_lstm()
# model.summary()


# try:
#     tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
#     tf.config.experimental_connect_to_cluster(tpu)
#     tf.tpu.experimental.initialize_tpu_system(tpu)
#     strategy = tf.distribute.experimental.TPUStrategy(tpu)

# except:
#     strategy = tf.distribute.get_strategy()

# print('Number of replicas in sync: ', strategy.num_replicas_in_sync)

# tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')


# def bert_encode(data, maximum_length):
#     input_ids = []
#     attention_masks = []

#     for text in data:
#         encoded = tokenizer.encode_plus(
#             text,
#             add_special_tokens=True,
#             max_length=maximum_length,
#             pad_to_max_length=True,

#             return_attention_mask=True,
#         )
#         input_ids.append(encoded['input_ids'])
#         attention_masks.append(encoded['attention_mask'])

#     return np.array(input_ids), np.array(attention_masks)


# texts = df['message_clean']
# target = df['target_encoded']

# train_input_ids, train_attention_masks = bert_encode(texts, 60)


# def create_model(bert_model):

#     input_ids = keras.Input(shape=(60,), dtype='int32')
#     attention_masks = keras.Input(shape=(60,), dtype='int32')

#     output = bert_model([input_ids, attention_masks])
#     output = output[1]
#     output = keras.layers.Dense(32, activation='relu')(output)
#     output = keras.layers.Dropout(0.2)(output)
#     output = keras.layers.Dense(1, activation='sigmoid')(output)

#     model = keras.Model(inputs=[input_ids, attention_masks], outputs=output)
#     model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
#     return model


# bert_model = TFBertModel.from_pretrained('bert-base-uncased')
# model = create_model(bert_model)
# print(model.summary())

df = pd.read_csv("train_nlp.csv", encoding="latin-1")
test_df = pd.read_csv("test_nlp.csv", encoding="latin-1")

df = df.dropna(how="any", axis=1)
df['text_len'] = df['text'].apply(lambda x: len(x.split(' ')))

print(df.head())

balance_counts = df.groupby('target')['target'].agg('count').values
print(balance_counts)

# fig = go.Figure()
# fig.add_trace(go.Bar(
#     x=['Fake'],
#     y=[balance_counts[0]],
#     name='Fake',
#     text=[balance_counts[0]],
#     textposition='auto',
#     marker_color=primary_blue
# ))
# fig.add_trace(go.Bar(
#     x=['Real disaster'],
#     y=[balance_counts[1]],
#     name='Real disaster',
#     text=[balance_counts[1]],
#     textposition='auto',
#     marker_color=primary_grey
# ))
# fig.update_layout(
#     title='<span style="font-size:32px; font-family:Times New Roman">Dataset distribution by target</span>'
# )
# fig.show()

# disaster_df = df[df['target'] == 1]['text_len'].value_counts().sort_index()
# fake_df = df[df['target'] == 0]['text_len'].value_counts().sort_index()

# fig = go.Figure()
# fig.add_trace(go.Scatter(
#     x=disaster_df.index,
#     y=disaster_df.values,
#     name='Real disaster',
#     fill='tozeroy',
#     marker_color=primary_blue,
# ))
# fig.add_trace(go.Scatter(
#     x=fake_df.index,
#     y=fake_df.values,
#     name='Fake',
#     fill='tozeroy',
#     marker_color=primary_grey,
# ))
# fig.update_layout(
#     title='<span style="font-size:32px; font-family:Times New Roman">Data Roles in Different Fields</span>'
# )
# fig.show()


def remove_url(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)


def remove_emoji(text):
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_html(text):
    html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(html, '', text)


def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub(
        'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        '',
        text
    )
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)

    text = remove_url(text)
    text = remove_emoji(text)
    text = remove_html(text)

    return text


# Test emoji removal
print(remove_emoji("Omg another Earthquake 😔😔"))

stop_words = stopwords.words('english')
more_stopwords = ['u', 'im', 'c']
stop_words = stop_words + more_stopwords

stemmer = nltk.SnowballStemmer("english")


def preprocess_data(text):
    # Clean puntuation, urls, and so on
    text = clean_text(text)
    # Remove stopwords and Stemm all the words in the sentence
    text = ' '.join(stemmer.stem(word) for word in text.split(' ') if word not in stop_words)

    return text


test_df['text_clean'] = test_df['text'].apply(preprocess_data)

df['text_clean'] = df['text'].apply(preprocess_data)
print(df.head())


def create_corpus_df(tweet, target):
    corpus = []

    for x in tweet[tweet['target'] == target]['text_clean'].str.split():
        for i in x:
            corpus.append(i)
    return corpus


corpus_disaster_tweets = create_corpus_df(df, 1)

dic = defaultdict(int)
for word in corpus_disaster_tweets:
    dic[word] += 1

top = sorted(dic.items(), key=lambda x: x[1], reverse=True)[:10]
print(top)

twitter_mask = np.array(Image.open('twitter_mask3.jpg'))

# wc = WordCloud(
#     background_color='white',
#     max_words=200,
#     mask=twitter_mask,
# )
# wc.generate(' '.join(text for text in df.loc[df['target'] == 1, 'text_clean']))
# plt.figure(figsize=(12, 6))
# plt.title('Top words for Real Disaster tweets',
#           fontdict={'size': 22,  'verticalalignment': 'bottom'})
# plt.imshow(wc)
# plt.axis("off")
# plt.show()

# corpus_disaster_tweets = create_corpus_df(df, 0)

# dic = defaultdict(int)
# for word in corpus_disaster_tweets:
#     dic[word] += 1

# top = sorted(dic.items(), key=lambda x: x[1], reverse=True)[:10]
# top

# wc = WordCloud(
#     background_color='white',
#     max_words=200,
#     mask=twitter_mask,
# )
# wc.generate(' '.join(text for text in df.loc[df['target'] == 0, 'text_clean']))
# plt.figure(figsize=(12, 6))
# plt.title('Top words for Fake messages',
#           fontdict={'size': 22,  'verticalalignment': 'bottom'})
# plt.imshow(wc)
# plt.axis("off")
# plt.show()

# # how to define X and y (from the SMS data) for use with COUNTVECTORIZER
# x = df['text_clean']
# y = df['target']

# # Split into train and test sets

# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
# print(len(x_train), len(y_train))
# print(len(x_test), len(y_test))

# pipe = Pipeline([
#     ('bow', CountVectorizer()),
#     ('tfid', TfidfTransformer()),
#     ('model', xgb.XGBClassifier(
#         use_label_encoder=False,
#         eval_metric='auc',
#     ))
# ])

# # Fit the pipeline with the data
# pipe.fit(x_train, y_train)

# y_pred_class = pipe.predict(x_test)
# y_pred_train = pipe.predict(x_train)

# print('Train: {}'.format(metrics.accuracy_score(y_train, y_pred_train)))
# print('Test: {}'.format(metrics.accuracy_score(y_test, y_pred_class)))

# fig = conf_matrix(metrics.confusion_matrix(y_test, y_pred_class))
# fig.show()

# train_tweets = df['text_clean'].values
# test_tweets = test_df['text_clean'].values
# train_target = df['target'].values

# # Calculate the length of our vocabulary
# word_tokenizer = Tokenizer()
# word_tokenizer.fit_on_texts(train_tweets)

# vocab_length = len(word_tokenizer.word_index) + 1
# print(vocab_length)


# def show_metrics(pred_tag, y_test):
#     print("F1-score: ", f1_score(pred_tag, y_test))
#     print("Precision: ", precision_score(pred_tag, y_test))
#     print("Recall: ", recall_score(pred_tag, y_test))
#     print("Acuracy: ", accuracy_score(pred_tag, y_test))
#     print("-"*50)
#     print(classification_report(pred_tag, y_test))


# def embed(corpus):
#     return word_tokenizer.texts_to_sequences(corpus)


# longest_train = max(train_tweets, key=lambda sentence: len(word_tokenize(sentence)))
# length_long_sentence = len(word_tokenize(longest_train))

# train_padded_sentences = pad_sequences(
#     embed(train_tweets),
#     length_long_sentence,
#     padding='post'
# )
# test_padded_sentences = pad_sequences(
#     embed(test_tweets),
#     length_long_sentence,
#     padding='post'
# )

# print(train_padded_sentences)

# Now we will load embedding vectors of those words that appear in the
# Glove dictionary. Others will be initialized to 0.
embedding_matrix = np.zeros((vocab_length, embedding_dim))

for word, index in word_tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

print(embedding_matrix)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    train_padded_sentences,
    train_target,
    test_size=0.25
)

# Load the model and train!!

model = glove_lstm()

checkpoint = ModelCheckpoint(
    'model.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    verbose=1,
    patience=5,
    min_lr=0.001
)
history = model.fit(
    X_train,
    y_train,
    epochs=7,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1,
    callbacks=[reduce_lr, checkpoint]
)

plot_learning_curves(history, [['loss', 'val_loss'], ['accuracy', 'val_accuracy']])

preds = model.predict_classes(X_test)
show_metrics(preds, y_test)
