import re
def preprocess_text(text):
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL', text)
    text = re.sub('@[^\s]+','', text)

    # Ignore the preprocessing from below
    # text = text.lower().replace("ё", "е")
    # text = re.sub('[^a-zA-Zа-яА-Я]+',' ', text)
    # text = re.sub(' +',' ', text)
    return text.strip()

import numpy as np
import pandas as pd
# Tweet ID,text,SentiStrength,Vader,Textblob,Vote
labels = []
texts = []
data = pd.read_csv('vote_all_no_conflict.csv')
# data = pd.read_csv('vote_json_all_no_conflict_small.csv')
reviews = np.array(data['text'])
l = np.array(data['Vote'])
for label in l:
    if label == 'Neutral':
        labels.append(0)
    elif label == 'Positive':
        labels.append(1)
    else:
        labels.append(2)
for i in range(len(reviews)):
    reviews[i] = preprocess_text(reviews[i])


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
maxlen = 150
#Cuts off reviews after 150 words
training_samples = 2000000
#Trains on 2,000,000 samples
validation_samples = 475775
#Validates (Test) on 475,775 samples
max_words = 1000000
#Considers only the top 1,000,000 words in the dataset

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)



indices = np.arange(data.shape[0])
#Splits the data into a training set and a validation set, but first shuffles
#the data, because you're starting with data in which samples are ordered
#(all negative first, then all positive)
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]


import os
twt_dir = 'twitter_embeds'
embeddings_index = {}
f = open(os.path.join(twt_dir, 'glove.twitter.27B.25d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))


embedding_dim = 25
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            #Words not found in the embedding index will be all zeros.


from tensorflow.keras.preprocessing import sequence
max_features = 10000
#Number of words to consider as features
#Cuts off texts after this many words (among the max_features most common words)
print('Loading data...')
(input_train, y_train, input_test, y_test) = x_train, y_train, x_val, y_val
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')
print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)


from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
import tensorflow as tf
model = Sequential()
model.add(Embedding(max_words, embedding_dim))
model.add(SimpleRNN(128))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='rmsprop', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['acc'])
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

############### Comment if you want to not train
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

BATCH_SIZE = 20
STEPS_PER_EPOCH = labels.size / BATCH_SIZE
SAVE_PERIOD = 1
checkpoint_path = 'pre_trained_twitter_model.h5'
# Create a callback that saves the model's weights every 10 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=int(SAVE_PERIOD * STEPS_PER_EPOCH))

with tf.device('CPU:0'):
    history = model.fit(x=input_train,
                        y=y_train,
                        epochs=10,
                        batch_size=1024,
                        validation_split=0.1,
                        verbose=1,
                        callbacks=[cp_callback],
                        validation_data=None,
                        shuffle=True,
                        class_weight=None,
                        sample_weight=None,
                        initial_epoch=0,
                        steps_per_epoch=None,
                        validation_steps=None,
                        validation_batch_size=None,
                        validation_freq=1,
                        max_queue_size=20,
                        workers=16,
                        use_multiprocessing=True)
################## End comment

# Evaluate the model on the test data using `evaluate`

model.load_weights('pre_trained_twitter_model.h5')
print("Loaded model from disk")

print("Evaluate on test data")
results = model.evaluate(input_test, y_test, batch_size=1024)
print("test loss, test acc:", results)
from sklearn.metrics import classification_report

y_pred = model.predict(input_test)
print(classification_report(y_test, np.argmax(y_pred, axis=-1)))
# 465/465 [==============================] - 6s 13ms/step - loss: 0.2218 - sparse_categorical_accuracy: 0.9299
# test loss, test acc: [0.22180010378360748, 0.9298744201660156]



# model.save_weights('pre_trained_twitter_model.h5')


# import matplotlib.pyplot as plt
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(acc) + 1)
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()