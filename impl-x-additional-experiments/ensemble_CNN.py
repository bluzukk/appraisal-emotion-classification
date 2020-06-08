#!/usr/bin/python3.5
import csv
import pandas as pd
import numpy as np
from numpy import asarray, zeros
import tensorflow as tf

from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer

from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences, sequence
from keras.models import load_model

import sys
sys.path.append('..')
import util.embedding
from util.keras_models import cnn_model, MaxEnt
from util.keras_models import text_cnn_model_baseline, text_cnn_model_appraisals, shallowNN_emotions_from_dimensions, text_cnn_model_selector
import util.metrics as metrics


# Hide tensorflow infos
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '6'

LABELS = ['Anger', 'Disgust', 'Fear', 'Guilt', 'Joy', 'Sadness', 'Shame']
DIMENSIONS = ['Attention', 'Certainty', 'Effort', 'Pleasant', 'Responsibility', 'Control', 'Circumstance']

# Parameters
EPOCHS_p1 = 15 # 15
EPOCHS_p2 = 25 # 25
EPOCHS_text_to_emotion = 25 # 25
EPOCHS_SELECTION_MODEL = 10

BATCH_SIZE = 32
DROPOUT = 0.5
LAYER_DIM = 128
FILTER_SIZE = [2,3,4]

KFOLDS = 10
ROUNDS = 10
VERBOSITY = 1
OPTIMIZER = 'adam'

EMBEDDING_DIMS = 300
MAX_SEQUENCE_LENGTH = 300
MAX_NUM_WORDS = 50000

SAVEFILE = 'results/baseline_CNN_enISEAR.txt'
EMBEDDING_FILE = '../embeddings/enISEAR_glove_embedding.npy'
DATASET = '../corpora/enISEAR-appraisal-V1/enISEAR_appraisal_majority.tsv'

vectors_raw = []
with open(DATASET) as tsvfile:
  reader = csv.reader(tsvfile, delimiter='\t')
  firstLine = True
  for row in reader:
    if firstLine:# Skip first line
        firstLine = False
        continue
    vector_row = [int(row[3]),int(row[4]),int(row[5]),int(row[6]),int(row[7]),int(row[8]),int(row[9])]
    vectors_raw.append(vector_row)

vectors = np.array(vectors_raw)

data_raw = pd.read_csv(DATASET, sep='\t')
classes_enISEAR = data_raw['Prior_Emotion']
data = data_raw['Sentence']

sentence_enISEAR = data_raw['Sentence']
sentence_enISEAR_raw = data_raw['Sentence']
vectorizer = CountVectorizer()
sentence_enISEAR = vectorizer.fit_transform(sentence_enISEAR)

# Preprocessing dataset
# To lower case, remove , and .
data = data.str.lower().str.replace(".","").str.replace(",","")

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, lower=True)
tokenizer.fit_on_texts(data)
vocab_size = len(tokenizer.word_index) + 1
word_index = tokenizer.word_index
num_words = min(MAX_NUM_WORDS, vocab_size)

data_enISEAR = tokenizer.texts_to_sequences(data)
data_padded = sequence.pad_sequences(data_enISEAR, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

embedding_path = '../../../../../Downloads/glove.6B.300d.txt'
# embedding_matrix = util.embedding.prepareEmbeddings(word_index, MAX_NUM_WORDS, embedding_path)
# util.embedding.saveEmbedding(embedding_matrix, EMBEDDING_FILEe)
print('Loading embedding' + EMBEDDING_FILE)
embedding_matrix = util.embedding.loadEmbedding(EMBEDDING_FILE)


class_weight = {0: 1.131,
                1: 1.000,
                2: 1.903,
                3: 5.107,
                4: 2.019,
                5: 3.338,
                6: 3.171}

class_weight_selector_model = {0: 1.0,
                               1: 10.0}

from keras import backend as K
from keras.models import load_model

def performCrossValidation(x_data, y_data):
    percentage_done = 0
    metrics_final = metrics.metrics(None, None, LABELS, 2)
    TP_total = 0
    size_total = 0
    TP_Baseline = 0

    for seed in range(ROUNDS):
        np.random.seed(seed)

        kfold = KFold(n_splits=KFOLDS, shuffle=True, random_state=seed)
        for train, test in kfold.split(x_data, y_data):
            K.clear_session()

            classes_train = pd.concat([y_data[train], pd.get_dummies(y_data[train])],axis=1).drop(['Prior_Emotion'],axis=1)
            classes_test = pd.concat([y_data[test], pd.get_dummies(y_data[test])],axis=1).drop(['Prior_Emotion'],axis=1)

            # Learn to predict emotins based on text (on enISEAR)
            text_to_emotion_model = text_cnn_model_baseline(MAX_SEQUENCE_LENGTH, 3116, EMBEDDING_DIMS, FILTER_SIZE, CONV_FILTERS, embedding_matrix, DROPOUT, LABELS, 'softmax')
            text_to_emotion_model.compile(OPTIMIZER, 'categorical_crossentropy', metrics=['accuracy'])
            text_to_emotion_model.fit(data_padded[train], classes_train, batch_size=BATCH_SIZE, epochs=EPOCHS_text_to_emotion, verbose=VERBOSITY)

            # text_to_emotion_model.save('saved_models/baseline_' + str(percentage_done))
            # # del text_to_emotion_model
            # # K.clear_session()
            # text_to_emotion_model = load_model('saved_models/baseline_' + str(percentage_done))

            predicted_classes_text_to_emotion = []
            predictions = text_to_emotion_model.predict(x_data[test])
            for i in range(len(predictions)):
                index = np.argmax(predictions[i])
                predicted_classes_text_to_emotion.append(LABELS[index])

            metrics_fold1 = metrics.metrics(y_data[test], predicted_classes_text_to_emotion, LABELS, 2)
            metrics_fold1.showResults()

            # # Learn to predict emotions from dimensions
            model1 = shallowNN_emotions_from_dimensions(len(DIMENSIONS), LAYER_DIM, DROPOUT, LABELS, 'softmax')
            model1.compile(OPTIMIZER, 'categorical_crossentropy', metrics=['accuracy'])
            # vectors_shaped = np.expand_dims(vectors[train], axis=2)
            model1.fit(vectors[train], classes_train, batch_size=BATCH_SIZE, epochs=EPOCHS_p1, verbose=VERBOSITY)
            # model1.save('saved_models/dim_to_emotion_' + str(percentage_done))
            # model1 = load_model('saved_models/dim_to_emotion_' + str(percentage_done))

            model = text_cnn_model_appraisals(MAX_SEQUENCE_LENGTH, vocab_size, EMBEDDING_DIMS, FILTER_SIZE, CONV_FILTERS, embedding_matrix, DROPOUT, len(LABELS), 'sigmoid')

            model.compile(OPTIMIZER, 'binary_crossentropy', metrics=['accuracy'])
            # model.fit(data_padded[train], vectors[train], batch_size=BATCH_SIZE, epochs=EPOCHS_p2, verbose=VERBOSITY, class_weight=class_weight)
            model.fit(data_padded[train], vectors[train], batch_size=BATCH_SIZE, epochs=EPOCHS_p2, verbose=VERBOSITY)
            # model.save('saved_models/text_to_dim_' + str(percentage_done))
            # model = load_model('saved_models/text_to_dim_' + str(percentage_done))

            # predict dimensions on enISEAR
            preds = model.predict(data_padded[test])
            results = []
            for row in range(len(preds)):
                res = []
                for dim in range(len(DIMENSIONS)):
                    value = preds[row][dim]
                    if (value >= 0.5):
                        value = 1
                    else:
                        value = 0

                    res.append(value)
                results.append(res)

            results = np.array(results)

            predicted_classes = []
            predictions = model1.predict(results)
            for i in range(len(predictions)):
                index = np.argmax(predictions[i])
                predicted_classes.append(LABELS[index])


            preds = model.predict(data_padded)
            results = []
            for row in range(len(preds)):
                res = []
                for dim in range(len(DIMENSIONS)):
                    value = preds[row][dim]
                    if (value >= 0.5):
                        value = 1
                    else:
                        value = 0

                    res.append(value)
                results.append(res)
            results = np.array(results)

            # Predict emotions from predicted appraisals on enISEAR
            predicted_classes_train = []
            predictions = model1.predict(results)
            for i in range(len(predictions)):
                index = np.argmax(predictions[i])
                predicted_classes_train.append(LABELS[index])

            print('enISEAR: T->A->E')
            metrics_fold = metrics.metrics(y_data[test], predicted_classes, LABELS, 2)
            metrics_fold.showResults()

            # Predict emotions from text on enISEAR
            predicted_classes_text_to_emotion_train = []
            predictions = text_to_emotion_model.predict(data_padded)
            for i in range(len(predictions)):
                index = np.argmax(predictions[i])
                predicted_classes_text_to_emotion_train.append(LABELS[index])


            sentences = data_enISEAR
            classes = classes_enISEAR

            TP_text_to_emotion = 0
            TP_text_to_appraisal_to_emotion = 0
            FN = 0

            train_sentences = []
            pred_combined = []
            labels_selector = []
            for i in range(len(data_padded)):
                label_gold = classes_enISEAR.iloc[i]
                if (label_gold == predicted_classes_text_to_emotion_train[i]
                and label_gold == predicted_classes_train[i]):
                    # Both models correct
                    labels_selector.append([1,1])
                elif (label_gold == predicted_classes_train[i]):
                    # Only appraisal model correct
                    labels_selector.append([1,0])
                elif (label_gold == predicted_classes_text_to_emotion_train[i]):
                    # Only text-to-emotion model correct
                    labels_selector.append([0,1])
                else:
                    # Both models predicted the wrong emotion
                    labels_selector.append([0,0])

            # print(len(labels_selector))

            labels_selector = np.array(labels_selector)
            selector_model = text_cnn_model_selector(MAX_SEQUENCE_LENGTH, vocab_size, EMBEDDING_DIMS, FILTER_SIZE, CONV_FILTERS, embedding_matrix, DROPOUT, LABELS, 'sigmoid')
            selector_model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=OPTIMIZER)
            selector_model.fit(data_padded, labels_selector, batch_size=32, epochs=EPOCHS_SELECTION_MODEL, verbose=VERBOSITY, class_weight=class_weight_selector_model)


            TP_text_to_emotion = 0
            TP_text_to_appraisal_to_emotion = 0
            FN = 0

            selection_ = []
            selections = selector_model.predict(x_data[test])
            for i in range(len(selections)):
                selection_.append(np.argmax(selections[i]))
                # selection_.append(selection[i])
                # print(selection_[i])
                # print(selection[i])

            sentences = sentence_enISEAR_raw[test]
            classes   = classes_enISEAR[test]

            pred_combined = []
            labels_selector = []
            for i in range(len(x_data[test])):
                label_gold = classes.iloc[i]
                if (selection_[i] == 1):
                    TP_text_to_emotion += 1
                    pred_combined.append(predicted_classes_text_to_emotion[i])
                elif (selection_[i] == 0):
                    # Appraisal system is correct
                    TP_text_to_appraisal_to_emotion += 1 # = TN
                    pred_combined.append(predicted_classes[i])
                else:
                    FN += 1 #
                    pred_combined.append(predicted_classes_text_to_emotion[i])

            percentage_done += 1
            print('\nPerforming CV... (' + str(percentage_done) + "%)")
            print('Selected from Baseline : ' + str(TP_text_to_emotion))
            print('Selected from Pipeline : ' + str(TP_text_to_appraisal_to_emotion))


            metrics_fold = metrics.metrics(y_data[test], pred_combined, LABELS, 2)
            metrics_fold.showResults()
            metrics_final.addIntermediateResults(y_data[test], pred_combined)
            # TP_total += TP
            # size_total += size


    print('\nFinal Result:')
    metrics_final.writeResults(EXPERIMENTNAME, SAVEFILE)
    return


performCrossValidation(data_padded, classes_enISEAR)
