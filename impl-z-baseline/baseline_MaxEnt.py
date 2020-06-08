#!/usr/bin/env python
"""
MaxEnt baseline system
"""

import argparse
parser = argparse.ArgumentParser()
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument('--dataset', '-d',
            type=str,
            choices=['enISEAR', 'TEC', 'ISEAR'],
            help='specify a dataset', required=True)
optional.add_argument('--folds', '-f',
            default=10, type=int,
            help='set the number of folds (default 10)')
optional.add_argument('--runs', '-r',
            default=10, type=int,
            help='set the number of runs (default 10)')
optional.add_argument('--gpu',
            action='store_true', help='force to run experiment on GPU')
optional.add_argument('--cpu',
            action='store_true', help='force to run experiment on CPU')
optional.add_argument('--quiet',
            action='store_true', help='reduce keras outputs')

args = parser.parse_args()
KFOLDS = args.folds
ROUNDS = args.runs
MODEL = 'MaxEnt'
_DATASET = args.dataset
if (args.quiet):
    VERBOSITY = 0
else: VERBOSITY = 1

print('----------------------------------')
print('   Starting baseline experiment   ')
print('----------------------------------')
print('   Model:  \t' , MODEL)
print('   Dataset:\t', _DATASET)
print('   Folds:  \t', KFOLDS)
print('   Runs:   \t', ROUNDS)
print('----------------------------------\n')

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
# Testing with sklearn
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report

import sys
sys.path.append('..')
from util.keras_models import MaxEnt
import util.metrics as metrics

# Hide tensorflow infos
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

EXPERIMENTNAME = 'Baseline MaxEnt'
OPTIMIZER = 'adam'

## enISEAR config
if (args.dataset == 'enISEAR'):
    LABELS = ['Anger', 'Disgust', 'Fear', 'Guilt', 'Joy', 'Sadness', 'Shame']
    SAVEFILE = 'results_baseline_MaxEnt_enISEAR.txt'
    DATASET = '../corpora/enISEAR-appraisal-V1/enISEAR_appraisal_majority.tsv'
    EPOCHS = 15
    BATCH_SIZE = 1
    dataframe = pd.read_csv(DATASET, sep='\t')

## ISEAR config
if (args.dataset == 'ISEAR'):
    LABELS = ['Anger', 'Disgust', 'Fear', 'Guilt', 'Joy', 'Sadness', 'Shame']
    DATASET = '../corpora/ISEAR.csv'
    SAVEFILE = 'results/baseline_MaxEnt_ISEAR.txt'
    EPOCHS = 5
    BATCH_SIZE = 1
    dataframe = pd.read_csv(DATASET, sep=',')

## TEC config
if (args.dataset == 'TEC'):
    LABELS = [':: anger', ':: disgust', ':: fear', ':: joy', ':: sadness', ':: surprise']
    DATASET = '../corpora/TEC_dataset.csv'
    SAVEFILE = 'results/baseline_MaxEnt_TEC.txt'
    EPOCHS = 5
    BATCH_SIZE = 16
    dataframe = pd.read_csv(DATASET, sep=',')


# Load dataset
instances = dataframe['Sentence']
labels = dataframe['Prior_Emotion']

# Preprocessing dataset
instances = instances.str.lower().str.replace(".","").str.replace(",","")
# Vectorize data
vectorizer = CountVectorizer()
instances_vects = vectorizer.fit_transform(instances)
input_shape  = instances_vects.shape[1] # feature count
output_shape = len(LABELS) # number of classes (7)


def performCrossValidation(x_data, y_data):
    metrics_final = metrics.metrics(None, None, LABELS, 2)

    for seed in range(ROUNDS):
        np.random.seed(seed)
        kfold = KFold(n_splits=KFOLDS, shuffle=True, random_state=seed)

        for train, test in kfold.split(x_data, y_data):
            classes_train = pd.concat([y_data[train], pd.get_dummies(y_data[train])],axis=1).drop(['Prior_Emotion'],axis=1)
            classes_test = pd.concat([y_data[test], pd.get_dummies(y_data[test])],axis=1).drop(['Prior_Emotion'],axis=1)

            model = MaxEnt(input_shape, output_shape, 'softmax')
            model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=OPTIMIZER)
            model.fit(x_data[train], classes_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSITY)

            predicted_classes = []
            loss, accuracy = model.evaluate(x_data[test], classes_test, verbose=0)
            print(accuracy)
            predictions = model.predict(x_data[test])
            for i in range(len(predictions)):
                index = np.argmax(predictions[i])
                predicted_classes.append(LABELS[index])

            # Show fold result
            metrics_fold = metrics.metrics(y_data[test], predicted_classes, LABELS, 2)
            metrics_fold.showResults()
            metrics_final.addIntermediateResults(y_data[test], predicted_classes)

            # Testing with sklearn
            # logisticRegression = LogisticRegression(n_jobs=1, C=1e5, solver='lbfgs', multi_class='multinomial', max_iter=1000, penalty='l2').fit(x_data[train], y_data[train])
            # predicted = logisticRegression.predict(x_data[test])
            # print(classification_report(y_data[test], predicted, target_names=LABELS))

            from keras import backend as K
            K.clear_session()


    print('\nFinal Result:')
    metrics_final.writeResults(EXPERIMENTNAME, SAVEFILE)
    return

if (args.gpu):
    # Force on GPU
    tf.debugging.set_log_device_placement(True)
    with tf.device('/device:GPU:0'):
        performCrossValidation(instances_vects, labels)
elif (args.cpu):
    # Force on GPU
    tf.debugging.set_log_device_placement(True)
    with tf.device('/device:CPU:0'):
        performCrossValidation(instances_vects, labels)
else:
    # Let tensorflow choose device
    performCrossValidation(instances_vects, labels)
