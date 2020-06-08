#!/usr/bin/env python

"""
Combining predictions based on text and appraisals using oracle
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
MODEL = 'CNN'
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
print('   Starting pipeline experiment   ')
print('----------------------------------')
print('   Model:    \t' , MODEL)
print('   Dataset:  \t' , _DATASET)
print('   Folds:    \t', KFOLDS)
print('   Runs:     \t', ROUNDS)
print('----------------------------------\n')


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

import sys
sys.path.append('..')
import util.embedding
from util.keras_models import text_cnn_model_baseline, text_cnn_model_appraisals, shallowNN_emotions_from_dimensions, cnn_model
import util.metrics as metrics
from keras.models import load_model


# Hide tensorflow infos
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '6'

LABELS = ['Anger', 'Disgust', 'Fear', 'Guilt', 'Joy', 'Sadness', 'Shame']
EXPERIMENTNAME = 'Combined Model upper Bound testing'
SAVEFILE = 'results/oracle-Combined-model.txt'
embedding_file = '../embeddings/enISEAR_glove_embedding.npy'

# Parameters
EPOCHS_p1 = 10 # 15
EPOCHS_p2 = 15 # 25
EPOCHS_text_to_emotion = 15 # 25

BATCH_SIZE = 32
DROPOUT = 0.5
LAYER_DIM = 128
FILTER_SIZE = [2,3,4]
OPTIMIZER = 'adam'

EMBEDDING_DIMS = 300
MAX_SEQUENCE_LENGTH = 300
MAX_NUM_WORDS = 50000


# Retrieve appraisal dimension data
vectors_raw = []
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
data = data.str.lower().str.replace(".","").str.replace(",","")
# Tokenize dataset
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, lower=True)
tokenizer.fit_on_texts(data)
vocab_size = len(tokenizer.word_index) + 1
word_index = tokenizer.word_index
num_words = min(MAX_NUM_WORDS, vocab_size)

data = tokenizer.texts_to_sequences(data)
data_padded = sequence.pad_sequences(data, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

embedding_matrix = util.embedding.loadEmbedding(embedding_file)

if (_DATASET == 'enISEAR_V1'):
    class_weight = {0: 1.131,
                    1: 1.000,
                    2: 1.903,
                    3: 5.107,
                    4: 2.019,
                    5: 3.338,
                    6: 3.171}
elif (_DATASET == 'enISEAR_V2'):
    print('not done yet')
    class_weight = {0: 999,
                    1: 999,
                    2: 999,
                    3: 999,
                    4: 999,
                    5: 999,
                    6: 999}
elif (_DATASET == 'enISEAR_V3'):
    class_weight = {0: 1.667,
                    1: 1.000,
                    2: 1.000,
                    3: 5.000,
                    4: 1.667,
                    5: 2.500}

def performCrossValidation(x_data, y_data):
    percentage_done = 0
    metrics_final = metrics.metrics(None, None, LABELS, 2)
    TP_total = 0
    size_total = 0

    for seed in range(ROUNDS):
        np.random.seed(seed)

        kfold = KFold(n_splits=KFOLDS, shuffle=True, random_state=seed)
        for train, test in kfold.split(x_data, y_data):
            from keras import backend as K
            K.clear_session()
            tf.reset_default_graph()

            classes_train = pd.concat([y_data[train], pd.get_dummies(y_data[train])],axis=1).drop(['Prior_Emotion'],axis=1)
            classes_test = pd.concat([y_data[test], pd.get_dummies(y_data[test])],axis=1).drop(['Prior_Emotion'],axis=1)

            ####################################################################
            # The problem with this orcale setup is that the models somehow
            # influence each other if they are running in the same process
            # on the GPU. A workaround is to train the baseline model on the
            # folds and save all model weights locally. Then do the same for
            # the pipeline. Afterwards load the saved weights and use them to
            # predict the emotions.
            # This means 10*10 models will be saved (10 folds times 10 runs)
            ####################################################################

            ####################################################################
            # Uncomment this to create and save the baseline model weights
            ####################################################################
            text_to_emotion_model = text_cnn_model_baseline(MAX_SEQUENCE_LENGTH, vocab_size, EMBEDDING_DIMS, FILTER_SIZE, CONV_FILTERS, embedding_matrix, DROPOUT, LABELS, 'softmax')
            text_to_emotion_model.compile(OPTIMIZER, 'categorical_crossentropy', metrics=['accuracy'])
            text_to_emotion_model.fit(x_data[train], classes_train, batch_size=BATCH_SIZE, epochs=EPOCHS_text_to_emotion, verbose=VERBOSITY)

            text_to_emotion_model.save('baseline_' + str(percentage_done))

            ####################################################################
            # This will load the baseline model weights
            ####################################################################
            text_to_emotion_model = load_model('baseline_' + str(percentage_done))

            # Evaluate baseline model
            predicted_classes_text_to_emotion = []
            predictions = text_to_emotion_model.predict(x_data[test])
            for i in range(len(predictions)):
                index = np.argmax(predictions[i])
                predicted_classes_text_to_emotion.append(LABELS[index])

            metrics_fold = metrics.metrics(y_data[test], predicted_classes_text_to_emotion, LABELS, 2)
            metrics_fold.showResults()

            ####################################################################
            # Uncomment this to create and save the pipeline model weights
            ####################################################################
            appraisal_emotion_predictor = shallowNN_emotions_from_dimensions(
                        len(DIMENSIONS), LAYER_DIM, DROPOUT, LABELS, 'softmax')
            appraisal_emotion_predictor.compile(OPTIMIZER,
                        'categorical_crossentropy',
                        metrics=['accuracy'])
            # vectors_shaped = np.expand_dims(vectors[train], axis=2)
            appraisal_emotion_predictor.fit(
                        vectors[train], classes_train, batch_size=BATCH_SIZE,
                        epochs=EPOCHS_p1, verbose=VERBOSITY)
            # Save weights
            appraisal_emotion_predictor.save('dim_to_emotion_' +
                        str(percentage_done))

            input_shape  = sentence_enISEAR.shape[1] # feature count
            model = text_cnn_model_appraisals(MAX_SEQUENCE_LENGTH, vocab_size,
                        EMBEDDING_DIMS, FILTER_SIZE, CONV_FILTERS,
                        embedding_matrix, DROPOUT, LABELS, 'sigmoid')

            model.compile(OPTIMIZER, 'binary_crossentropy', metrics=['accuracy'])
            # model.fit(data_padded[train], vectors[train], batch_size=BATCH_SIZE, epochs=EPOCHS_p2, verbose=VERBOSITY)
            model.fit(x_data[train], vectors[train], batch_size=BATCH_SIZE,
                        epochs=EPOCHS_p2, verbose=VERBOSITY,
                        class_weight=class_weight)
            model.save('text_to_dim_' + str(percentage_done))


            # Load models
            appraisal_emotion_predictor = load_model('dim_to_emotion_' + str(percentage_done))
            model = load_model('text_to_dim_' + str(percentage_done))


            # predict dimensions from ISEAR
            weights = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
            preds = model.predict(data_padded[test])
            results = []
            for row in range(len(preds)):
                res = []
                for dim in range(len(DIMENSIONS)):
                    value = preds[row][dim]
                    if (value >= weights[dim]):
                        value = 1
                    else:
                        value = 0

                    res.append(value)
                results.append(res)

            results = np.array(results)

            predicted_classes = []
            predictions = appraisal_emotion_predictor.predict(results)
            for i in range(len(predictions)):
                index = np.argmax(predictions[i])
                predicted_classes.append(LABELS[index])

            metrics_fold = metrics.metrics(y_data[test], predicted_classes, LABELS, 2)
            metrics_fold.showResults()

            sentences   = sentence_enISEAR_raw[test]
            classes    = classes_enISEAR[test]

            TP_text_to_emotion = 0
            TP_text_to_appraisal_to_emotion = 0
            FN = 0

            pred_combined = []
            for i in range(len(x_data[test])):
                label_gold = classes.iloc[i]
                if (label_gold == predicted_classes_text_to_emotion[i]):
                    # Text based system is correct
                    TP_text_to_emotion += 1
                    # green('\n')
                    # green(i)
                    # green('Gold-Emotion    : ' + label_gold)
                    # green('Prediction T->E   : ' + str(predicted_classes_text_to_emotion[i]))
                    # green('Prediction T->A->E: ' + predicted_classes[i])
                    pred_combined.append(predicted_classes_text_to_emotion[i])
                elif (label_gold == predicted_classes[i]):
                    # Appraisal system is correct
                    TP_text_to_appraisal_to_emotion += 1 # = TN
                    # yellow('\n')
                    # yellow(i)
                    # yellow('Gold-Emotion    : ' + label_gold)
                    # yellow('Prediction T->E   : ' + str(predicted_classes_text_to_emotion[i]))
                    # yellow('Prediction T->A->E: ' + predicted_classes[i])
                    pred_combined.append(predicted_classes[i])
                else:
                    FN += 1 #
                    # print('\nGold-Emotion      : ' + label_gold)
                    # print('Prediction T->E   : ' + str(predicted_classes_text_to_emotion[i]))
                    # print('Prediction T->A->E: ' + predicted_classes[i])
                    pred_combined.append(predicted_classes_text_to_emotion[i])

            percentage_done += 1
            print('\nPerforming CV... (%2d%%)' % percentage_done)

            size = len(predicted_classes)
            TP = TP_text_to_emotion + TP_text_to_appraisal_to_emotion
            accuracy = (TP/size)*100
            print('Current fold:')
            print('Accuracy: %2.2f' % accuracy)
            print('Text-to-emotion TP         : %2d' % TP_text_to_emotion)
            print('Text-to-appr-to-emotion TP : %2d' % TP_text_to_appraisal_to_emotion)


            metrics_fold = metrics.metrics(y_data[test], pred_combined, LABELS, 2)
            # metrics_fold.showResults()
            metrics_final.addIntermediateResults(y_data[test], pred_combined)
            TP_total += TP
            size_total += size


    print('\n\nFinal Result:')
    accuracy = ((TP_total)/size_total)*100
    print('Accuracy: %2.2f' % accuracy)
    print(TP_total)
    print(size_total)
    metrics_final.writeResults(EXPERIMENTNAME, SAVEFILE)
    return


performCrossValidation(data_padded, classes_enISEAR)
