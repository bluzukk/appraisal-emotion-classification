#!/usr/bin/env python
"""
Predicting emotions based on annotated appraisal dimensions (MaxEnt)
"""

import argparse
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
MODEL = 'MaxEnt'
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

print('------------------------------------------------------------')
print(' Starting emotion prediction from gold appraisal experiment ')
print('------------------------------------------------------------')
print('   Model:  \t' , MODEL)
print('   Folds:  \t', KFOLDS)
print('   Runs:   \t', ROUNDS)
print('------------------------------------------------------------\n')

import pandas as pd
import csv
import numpy as np
import tensorflow as tf

from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import Sequential
from keras.layers import Dense, Input, Activation

import sys
sys.path.append('..')
import util.metrics as metrics

# Hide tensorflow infos
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

# Parameters
EPOCHS = 10
BATCH_SIZE = 1
OPTIMIZER = 'adam'

EXPERIMENTNAME = 'MaxEnt annotated-Appr. -> Emotion ' + DATASET
SAVEFILE = 'results/gold_appraisal_based_emotion_predictionsMaxEnt_' + DATASET + '.txt'
LABELS = ['Anger', 'Disgust', 'Fear', 'Guilt', 'Joy', 'Sadness', 'Shame']

# Load appraisal annotation
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

data_raw = pd.read_csv(DATASET, sep='\t')
classes_enISEAR = data_raw['Prior_Emotion']


def performCrossValidation(x_data, y_data, LABELS, KFOLDS, EPOCHS, ROUNDS, BATCH_SIZE, VERBOSITY, RESULTSFILE, EXPERIMENTNAME):
    metrics_final = metrics.metrics(None, None, LABELS, 2)

    for seed in range(ROUNDS):
        np.random.seed(seed)
        kfold = KFold(n_splits=KFOLDS, shuffle=True, random_state=seed)
        for train, test in kfold.split(x_data, y_data):
            from keras import backend as K
            K.clear_session()
            classes_train = pd.concat([y_data[train], pd.get_dummies(y_data[train])],axis=1).drop(['Prior_Emotion'],axis=1)
            classes_test = pd.concat([y_data[test], pd.get_dummies(y_data[test])],axis=1).drop(['Prior_Emotion'],axis=1)

            model = Sequential()
            model.add(Dense(len(LABELS), input_shape=(len(DIMENSIONS),)))
            model.add(Activation('softmax')) # Softmax regression
            model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=OPTIMIZER)
            model.fit(x_data[train], classes_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSITY)

            predicted_classes = []
            loss, accuracy = model.evaluate(x_data[test], classes_test, verbose=0)
            print(accuracy)
            predictions = model.predict(x_data[test])
            for i in range(len(predictions)):
                index = np.argmax(predictions[i])
                predicted_classes.append(LABELS[index])

            rounding_decimals = 2
            metrics_fold = metrics.metrics(y_data[test], predicted_classes, LABELS, rounding_decimals)
            metrics_fold.showResults()
            metrics_final.addIntermediateResults(y_data[test], predicted_classes)
    print('\nFinal Result:')
    metrics_final.writeResults(EXPERIMENTNAME, SAVEFILE)


performCrossValidation(vectors, classes_enISEAR, LABELS, KFOLDS, EPOCHS, ROUNDS, BATCH_SIZE, VERBOSITY, SAVEFILE, EXPERIMENTNAME)
