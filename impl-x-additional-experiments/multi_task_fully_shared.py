#!/usr/bin/env python
"""
Multi task system on trained/tested on enISEAR
"""

import argparse
import sys
parser = argparse.ArgumentParser()
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument('--dataset', '-d',
            type=str,
            help='specify a dataset',
            choices=['enISEAR_V1', 'enISEAR_V2', 'enISEAR_V3'],
            required=True)
optional.add_argument('--folds', '-f',
            default=10, type=int,
            help='set the number of folds (default 10)')
optional.add_argument('--runs', '-r',
            default=10, type=int,
            help='set the number of runs (default 10)')
optional.add_argument('--quiet',
            action='store_true', help='reduce keras outputs')

args = parser.parse_args()
MODEL = 'CNN Multi-task'
_DATASET = args.dataset
KFOLDS = args.folds
ROUNDS = args.runs

if (args.quiet):
    VERBOSITY = 0
else: VERBOSITY = 1

if (_DATASET == 'enISEAR_V1'):
    DATASET = '../corpora/enISEAR-appraisal-V1/enISEAR_appraisal_majority.tsv'
    DIMENSIONS = ['Attention', 'Certainty', 'Effort', 'Pleasant', 'Responsibility', 'Control', 'Circumstance']
elif (_DATASET == 'enISEAR_V2'):
    DIMENSIONS = ['Attention', 'Certainty', 'Effort', 'Pleasant', 'Responsibility', 'Control', 'Circumstance']
    print('\nError: V2 dataset (annotations with visible emotions) not done yet.')
    print('Exiting')
    sys.exit()
    DATASET = '../corpora/enISEAR-appraisal-V2/enISEAR_appraisal.tsv'
elif (_DATASET == 'enISEAR_V3'):
    DATASET = '../corpora/enISEAR-appraisal-V3/enISEAR_appraisal_automated_binary.tsv'
    DIMENSIONS = ['Attention', 'Certainty', 'Effort', 'Pleasant', 'Resp./Control', 'Sit. Control']


print('----------------------------------')
print('  Starting multi-task experiment  ')
print('----------------------------------')
print('   Model:    \t' , MODEL)
print('   Dataset:  \t' , _DATASET)
print('   Folds:    \t', KFOLDS)
print('   Runs:     \t', ROUNDS)
print('----------------------------------\n')


import pandas as pd
import csv
import numpy as np
import tensorflow as tf

from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences, sequence

from keras.models import Sequential
from keras.layers.core import Dense, Activation

import sys
sys.path.append('..')
import util.metrics as metrics
import util.embedding

import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Reshape, Concatenate, Dropout
from keras.layers import GlobalMaxPooling1D, Flatten, Activation
from keras.layers import Conv1D, MaxPooling1D, Embedding, Reshape
from keras.layers import Dense, Input, Reshape, Concatenate, Dropout
from keras.layers import Conv2D, MaxPool2D, Embedding
from keras import regularizers
from keras.models import Model
from keras.initializers import Constant
from keras.utils import plot_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Hide tensorflow infos
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '6'

"""
Phase 3: Multi-Task System

Learn to predict dimensions and emotions jointly (CNN)

"""

EXPERIMENTNAME = 'CNN Multi-task ' + _DATASET
SAVEFILE = 'results/multitask_fully_shared_CNN_enISEAR_' + _DATASET + '.txt'
EMBEDDING_FILE = '../embeddings/enISEAR_glove_embedding.npy'
LABELS = ['Anger', 'Disgust', 'Fear', 'Guilt', 'Joy', 'Sadness', 'Shame']

# Parameters
EPOCHS = 25
BATCH_SIZE = 32
OPTIMIZER = 'adam'

EMBEDDING_DIMS = 300
MAX_SEQUENCE_LENGTH = 500
MAX_NUM_WORDS = 10000

# Load dimension vectors
vectors = []
with open(DATASET) as tsvfile:
  reader = csv.reader(tsvfile, delimiter='\t')
  firstLine = True
  for row in reader:
    if firstLine:# Skip first line
        firstLine = False
        continue
    if (_DATASET == 'enISEAR_V1'):
        vector_row = [int(row[3]),int(row[4]),int(row[5]),int(row[6]),int(row[7]),int(row[8]),int(row[9])]
    elif (_DATASET == 'enISEAR_V2'):
        vector_row = [int(row[3]),int(row[4]),int(row[5]),int(row[6]),int(row[7]),int(row[8]),int(row[9])]
    elif (_DATASET == 'enISEAR_V3'):
        vector_row = [int(row[3]),int(row[4]),int(row[5]),int(row[6]),int(row[7]),int(row[8])]
    vectors.append(vector_row)

vectors = np.array(vectors)

# Load sentences and emotions
data_raw = pd.read_csv(DATASET, sep='\t')
classes_enISEAR = data_raw['Prior_Emotion']
data = data_raw['Sentence']

sentence_enISEAR = data_raw['Sentence']
vectorizer = CountVectorizer()
sentence_enISEAR = vectorizer.fit_transform(sentence_enISEAR)

# Preprocessing dataset sentences
# To lower case, remove , and .
data = data.str.lower().str.replace(".","").str.replace(",","")

# Tokenize sentences
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, lower=True)
tokenizer.fit_on_texts(data)
vocab_size = len(tokenizer.word_index) + 1
word_index = tokenizer.word_index
num_words = min(MAX_NUM_WORDS, vocab_size)
data = tokenizer.texts_to_sequences(data)
data_padded = sequence.pad_sequences(data, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

# Load embedding matrix
embedding_matrix = util.embedding.loadEmbedding(EMBEDDING_FILE)


def create_CNN_Model():
    filter_sizes = [2,3,4]
    embedding_size = 300
    num_filters = 128
    sequence_length = MAX_SEQUENCE_LENGTH

    # Text input
    input_for_dimensions = Input(shape=(MAX_SEQUENCE_LENGTH,), name='input_text1')
    input_for_emotions   = Input(shape=(MAX_SEQUENCE_LENGTH,), name='input_text2')

    embedding1 = Embedding(vocab_size, embedding_size, embeddings_initializer=Constant(embedding_matrix), name='glove300_embedding1', trainable=False)(input_for_dimensions)
    # embedding2 = Embedding(vocab_size, embedding_size, embeddings_initializer=Constant(embedding_matrix), name='glove300_embedding2', trainable=False)(input_for_emotions)

    expend_shape = [embedding1.get_shape().as_list()[1], embedding1.get_shape().as_list()[2], 1]
    embedding_chars1 = Reshape(expend_shape)(embedding1)

    # expend_shape = [embedding2.get_shape().as_list()[1], embedding2.get_shape().as_list()[2], 1]
    # embedding_chars2 = Reshape(expend_shape)(embedding2)

    # Shared layers
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        conv = Conv2D(filters=num_filters,
                    kernel_size=[filter_size, embedding_size],
                    activation='relu',
                    name=('conv_filtersize%d' % filter_size))
        encoded_a = conv(embedding_chars1)
        encoded_b = conv(embedding_chars1)
        merged_conv = keras.layers.concatenate([encoded_b, encoded_a], axis=-1)
        max_pool = MaxPool2D(pool_size=[sequence_length - filter_size + 1, 1],
                       strides=(2, 2),
                       padding='valid',
                       name=('max_pool_%d' % filter_size))(merged_conv)
        max_pool = Dropout (0.2)(max_pool)
        pooled_outputs.append(max_pool)

    num_filters_total = num_filters * len(filter_sizes) * 2
    h_pool = Concatenate(axis=3)(pooled_outputs)
    merged = Reshape([num_filters_total])(h_pool)
    dropout = Dropout (0.5, name='0.5')(merged)
    fullyconnected = Dense(128, name='fullyconnected', activation='relu')(dropout)

    emo_output = Dense(len(LABELS), activation='softmax', name='emo_out', kernel_initializer='glorot_normal',
    bias_initializer=keras.initializers.constant(0.1))(fullyconnected)
    seq_output = Dense(len(DIMENSIONS), activation='sigmoid', name='dim_out', kernel_initializer='glorot_normal',
    bias_initializer=keras.initializers.constant(0.1))(fullyconnected)

    multitask_model = Model(inputs=[input_for_emotions, input_for_dimensions], outputs=[emo_output, seq_output])
    multitask_model.compile(loss='categorical_crossentropy',
                  optimizer=OPTIMIZER, metrics=['accuracy'], loss_weights=[0.5,1])

    return multitask_model

def performCrossValidation(x_data, y_data):
    metrics_final = metrics.metrics(None, None, LABELS, 2)

    for seed in range(ROUNDS):
        np.random.seed(seed)
        kfold = KFold(n_splits=KFOLDS, shuffle=True, random_state=seed)
        for train, test in kfold.split(x_data, y_data):
            from keras import backend as K
            K.clear_session()
            classes_train = pd.concat([y_data[train], pd.get_dummies(y_data[train])],axis=1).drop(['Prior_Emotion'],axis=1)
            classes_test = pd.concat([y_data[test], pd.get_dummies(y_data[test])],axis=1).drop(['Prior_Emotion'],axis=1)

            model = create_CNN_Model()
            # vectors_shaped = np.expand_dims(vectors[train], axis=2)
            # classes_train  = np.expand_dims(classes_train, axis=1)
            history = model.fit([data_padded[train], data_padded[train]], [classes_train, vectors[train]], batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSITY)

            predicted_emotions = []
            predictions = model.predict([data_padded[test], data_padded[test]])
            print(predictions[0][0])
            for i in range(len(predictions[0])):
                index = np.argmax(predictions[0][i])
                predicted_emotions.append(LABELS[index])

            metrics_fold = metrics.metrics(y_data[test], predicted_emotions, LABELS, 2)
            metrics_fold.showResults()
            metrics_final.addIntermediateResults(y_data[test], predicted_emotions)
    print('\nFinal Result:')
    metrics_final.writeResults(EXPERIMENTNAME, SAVEFILE)
    return


performCrossValidation(vectors, classes_enISEAR)
