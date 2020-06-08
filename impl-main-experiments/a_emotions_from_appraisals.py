#!/usr/bin/env python
"""
Predicting emotions based on annotated appraisal dimensions
with a shallow neural network.

"""

import sys
import pandas as pd
import csv
import numpy as np
import datetime
import argparse
from argparse import RawTextHelpFormatter
import tensorflow as tf

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences, sequence

sys.path.append('..')
import util.metrics as metrics
import util.embedding
from util.keras_models import shallowNN_emotions_from_dimensions, text_cnn_model_baseline

parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
parameter_config = parser.add_argument_group('parameter configuration')
backend_setup = parser.add_argument_group('optional tensorflow setup')
required.add_argument('--dataset', '-d',
            type=str,
            help='specify the input dataset.\n'
            'Corresponds to the training set if you also specify a testset',
            required=True)
optional.add_argument('--testset', '-t',
            type=str,
            help='if you have a test set and you don\'t want to use cross-validation\n'
            'you can specify a test set here.\n Then the --dataset argument must be your training set')
optional.add_argument('--savemodel', '-s',
            type=str,
            help='If you do not want to run a cross-validation you can save\n'
            'the created emotion prediction model weights and use them in other experiments')
parameter_config.add_argument('--epochs', '-e',
            default=20, type=int,
            help='set the number training epochs (default: 20)')
parameter_config.add_argument('--batchsize', '-b',
            default=32, type=int,
            help='set the training batchsize (default: 32)')
parameter_config.add_argument('--folds', '-f',
            default=10, type=int,
            help='set the number of CV folds (default: 10)')
parameter_config.add_argument('--runs', '-r',
            default=10, type=int,
            help='set the number of CV runs (default: 10)')
backend_setup.add_argument('--gpu',
            action='store_true', help='force to run experiment on GPU')
backend_setup.add_argument('--cpu',
            action='store_true', help='force to run experiment on CPU')
backend_setup.add_argument('--quiet',
            action='store_true', help='reduce keras outputs')
backend_setup.add_argument('--debug',
            action='store_true', help='show tensorflow backend informations')
args = parser.parse_args()

if (not args.debug):
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

### Parameters #################################################################
DROPOUT = 0.5
LAYER_DIM = 128
OPTIMIZER = 'adam'
EPOCHS = args.epochs
BATCH_SIZE = args.batchsize

KFOLDS = args.folds
ROUNDS = args.runs
EMBEDDING_DIMS = 300         # Embedding dimension
MAX_SEQUENCE_LENGTH = 500    # Maximum length of an instance (in words)
MAX_NUM_WORDS = 50000        # Maximum number of words in the dataset
################################################################################

if (args.quiet):
    VERBOSITY = 0
else: VERBOSITY = 1

def getAppraisals(dataset):
    # Craw possible dimensions - hacky way
    APPRAISALS = []
    try:
        with open(dataset) as f:
            first_line = f.readline()
            if ('Attention' in first_line):
                APPRAISALS.append('Attention')
            if ('Certainty' in first_line):
                APPRAISALS.append('Certainty')
            if ('Effort' in first_line):
                APPRAISALS.append('Effort')
            if ('Pleasant' in first_line):
                APPRAISALS.append('Pleasantness')
            if ('Responsibility/Control' in first_line):
                APPRAISALS.append('Responsibility/Control')
            if ('Situational Control' in first_line):
                APPRAISALS.append('Situational Control')
        if (not APPRAISALS):
            print('ERROR: Could not find appraisals in dataset "%s"' % dataset)
            print('\nExiting.')
            sys.exit(1)
    except FileNotFoundError:
        print('ERROR: Could not find dataset "%s"' % dataset)
        print('\nExiting.')
        sys.exit(1)
    return APPRAISALS

if (args.dataset == 'enISEAR_V1'):
    DATASET = '../corpora/enISEAR-appraisal-V1/enISEAR_appraisal_majority.tsv'
    APPRAISALS = ['Attention', 'Certainty', 'Effort', 'Pleasant', 'Responsibility', 'Control', 'Circumstance']
elif (args.dataset == 'enISEAR_V2'):
    APPRAISALS = ['Attention', 'Certainty', 'Effort', 'Pleasant', 'Resp./Control', 'Situational Control']
    DATASET = '../corpora/enISEAR-appraisal-V2/enISEAR_appraisals_automated_continous.tsv'
elif (args.dataset == 'enISEAR_V3'):
    DATASET = '../corpora/enISEAR-appraisal-V3/enISEAR_appraisal_automated_binary.tsv'
    APPRAISALS = ['Attention', 'Certainty', 'Effort', 'Pleasant', 'Resp./Control', 'Situational Control']
else:
    DATASET = args.dataset
    APPRAISALS = getAppraisals(DATASET)

if (args.testset):
    APPRAISALS_test = getAppraisals(args.testset)
    if (len(APPRAISALS) != len(APPRAISALS_test)):
        print('ERROR: Train and test set do not contain the same appraisals')
        print('\nExiting.')
        sys.exit(1)

# Load dataset
try:
    pandas_frame = pd.read_csv(DATASET, sep='\t')
    class_labels = pandas_frame['Prior_Emotion']
    LABELS = sorted(list(class_labels.unique()))
    # Load appraisal data
    appraisals = []
    with open(DATASET) as tsvfile:
      reader = csv.reader(tsvfile, delimiter='\t')
      firstLine = True
      for row in reader:
        if firstLine:# Skip first line
            firstLine = False
            continue
        if (args.dataset == 'enISEAR_V1'):
            vector_row = [int(row[3]),int(row[4]),int(row[5]),int(row[6]),int(row[7]),int(row[8]),int(row[9])]
        elif (args.dataset == 'enISEAR_V2'):
            vector_row = [row[2],row[3],row[4],row[5],row[6],row[7]]
        elif (args.dataset == 'enISEAR_V3'):
            vector_row = [int(row[3]),int(row[4]),int(row[5]),int(row[6]),int(row[7]),int(row[8])]
        else:
            vector_row = []
            for i in range(len(APPRAISALS)):
                vector_row.append(row[i+2])
                # First index is emotion_label, second is instance_text, third is first appraisal, ...
        appraisals.append(vector_row)
    appraisals = np.array(appraisals)
except FileNotFoundError:
    print('\nERROR: File "%s" was not found.' % DATASET)
    print('       Make sure you enter the correct path and filename with extension.')
    print('\nExiting.')
    sys.exit(1)
except IndexError:
    print('\nERROR: The dataset "%s" seems to have a wrong format.' % DATASET)
    print('       Make sure your dataset has the correct format, i.e.')
    print('       Prior_Emotion,Sentence,Attention,Certainty,Effort,Pleasant,Resp./Control,Situational Control')
    print('\nExiting.')
    sys.exit(1)
except:
    print("Unexpected error:", sys.exc_info()[1])
    sys.exit(1)

if args.testset:
    try:
        pandas_frame = pd.read_csv(args.testset, sep='\t')
        class_labels_test = pandas_frame['Prior_Emotion']
        appraisals_test = []
        with open(args.testset) as tsvfile:
          reader = csv.reader(tsvfile, delimiter='\t')
          firstLine = True
          for row in reader:
            if firstLine:# Skip first line
                firstLine = False
                continue
            vector_row = []
            for i in range(len(APPRAISALS)):
                vector_row.append(int(row[i+2]))
            appraisals_test.append(vector_row)
        appraisals_test = np.array(appraisals_test)
    except FileNotFoundError:
        print('\nERROR: The test set "%s" was not found.' % args.testset)
        print('       Make sure you enter the correct path and filename with extension.')
        print('\nExiting.')
        sys.exit(1)
    except IndexError:
        print('\nERROR: The test set "%s" seems to have a wrong format.' % DATASET)
        print('       Make sure your dataset has the correct format, i.e.')
        print('       Prior_Emotion,Sentence,Attention,Certainty,Effort,Pleasant,Resp./Control,Situational Control')
        print('\nExiting.')
        sys.exit(1)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        sys.exit(1)


print('------------------------------------------------------------')
print(' Starting emotion prediction from gold appraisal experiment ')
print('------------------------------------------------------------')
if (args.testset):
    print('   Train Set    :', DATASET)
    print('   Instances    :', len(appraisals))
    print('')
    print('   Train Set    :', args.testset)
    print('   Instances    :', len(appraisals_test))
    print('')
else:
    print('   Dataset      :', DATASET)
    print('   Instances    :', len(appraisals))
    print('')
print('   Class Labels :', LABELS)
print('   Appraisals   :', APPRAISALS)
print('   Epochs       :', EPOCHS)
print('   Batch Size   :', BATCH_SIZE)
if (not args.testset):
    print('   Folds        :', KFOLDS)
    print('   Runs         :', ROUNDS)
print('------------------------------------------------------------\n')


EXPERIMENTNAME = 'ANN annotated-Appr. -> Emotion' + DATASET
SAVEFILE = 'results_gold_appraisal_based_emotion_predictionsANN.txt'

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

            model = shallowNN_emotions_from_dimensions(len(APPRAISALS),
                                    LAYER_DIM, DROPOUT, LABELS, 'softmax')

            model.compile(OPTIMIZER, 'categorical_crossentropy', metrics=['accuracy'])
            # appraisals_shaped = np.expand_dims(appraisals[train], axis=2)
            print('\n\nTraining on %d instances...' % len(classes_train))
            if (args.quiet):
                for _ in range(EPOCHS):
                    model.fit(appraisals[train], classes_train, batch_size=BATCH_SIZE, epochs=1, verbose=VERBOSITY)
                    print('.', end='', flush=True)
            else:
                model.fit(appraisals[train], classes_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSITY)

            predicted_classes = []
            print('\nEvaluating fold (%d instances)...' % len(y_data[test]))
            # appraisals_shaped = np.expand_dims(appraisals[test], axis=2)
            predictions = model.predict(appraisals[test])
            for i in range(len(predictions)):
                index = np.argmax(predictions[i])
                predicted_classes.append(LABELS[index])

            metrics_fold = metrics.metrics(y_data[test], predicted_classes, LABELS, 2)
            metrics_fold.showResults()
            metrics_final.addIntermediateResults(y_data[test],predicted_classes)

    print('\nFinal Result:')
    metrics_final.writeResults(EXPERIMENTNAME, SAVEFILE)
    # metrics_final.createMarkdownResults()

def evalTrainAndTestSet(x_train, x_test, y_train, y_test):
    classes_train = pd.concat([y_train, pd.get_dummies(y_train)],axis=1).drop(['Prior_Emotion'],axis=1)
    classes_test = pd.concat([y_test, pd.get_dummies(y_test)],axis=1).drop(['Prior_Emotion'],axis=1)

    model = shallowNN_emotions_from_dimensions(len(APPRAISALS), LAYER_DIM, DROPOUT, LABELS, 'softmax')
    model.compile(OPTIMIZER, 'categorical_crossentropy', metrics=['accuracy'])

    # appraisals_shaped = np.expand_dims(appraisals, axis=2)
    print('\nTraining on %d instances...' % len(x_train))
    if (args.quiet):
        for _ in range(EPOCHS):
            model.fit(x_train, classes_train, batch_size=BATCH_SIZE, epochs=1, verbose=VERBOSITY)
            print('.', end='', flush=True)
    else:
        model.fit(x_train, classes_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSITY)


    predicted_classes = []
    print('\nTesting on %d instances...' % len(x_test))
    # appraisals_shaped = np.expand_dims(appraisals, axis=2)
    predictions = model.predict(x_test)
    for i in range(len(predictions)):
        index = np.argmax(predictions[i])
        predicted_classes.append(LABELS[index])

    metrics_fold = metrics.metrics(y_test, predicted_classes, LABELS, 2)
    metrics_fold.showResults()

def trainAndSaveModel(x_train, y_train):
    classes_train = pd.concat([y_train, pd.get_dummies(y_train)],axis=1).drop(['Prior_Emotion'],axis=1)

    model = shallowNN_emotions_from_dimensions(len(APPRAISALS), LAYER_DIM, DROPOUT, LABELS, 'softmax')
    model.compile(OPTIMIZER, 'categorical_crossentropy', metrics=['accuracy'])

    # appraisals_shaped = np.expand_dims(appraisals, axis=2)
    print('\nTraining on %d instances...' % len(x_train))
    if (args.quiet):
        for _ in range(EPOCHS):
            model.fit(x_train, classes_train, batch_size=BATCH_SIZE, epochs=1, verbose=VERBOSITY)
            print('.', end='', flush=True)
    else:
        model.fit(x_train, classes_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSITY)

    if (not args.savemodel.endswith('.h5')):
        args.savemodel = args.savemodel + '.h5'
    model.save(args.savemodel)
    print('Saved model to %s' % args.savemodel)


# Force to run experiment on cpu or gpu if desired
# otherwise let backend choose (prefers gpu if available)
if (args.gpu or args.cpu):
    if (args.gpu):
        print('INFO: Forcing experiment to run on GPU (GPU:0)')
        DEVICE = '/device:GPU:0'
    elif (args.cpu):
        print('INFO: Forcing experiment to run on CPU (CPU:0)')
        DEVICE = '/device:CPU:0'
    tf.debugging.set_log_device_placement(True)
    with tf.device(DEVICE):
        if args.testset:
            evalTrainAndTestSet(appraisals, appraisals_test, class_labels, class_labels_test)
        elif args.savemodel:
            trainAndSaveModel(appraisals, class_labels)
        else:
            performCrossValidation(appraisals, class_labels)
else:
    if args.testset:
        evalTrainAndTestSet(appraisals, appraisals_test, class_labels, class_labels_test)
    elif args.savemodel:
        trainAndSaveModel(appraisals, class_labels)
    else:
        performCrossValidation(appraisals, class_labels)
