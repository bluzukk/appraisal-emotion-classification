#!/usr/bin/env python
"""
CNN appraisal prediction on textual input

"""

import pandas as pd
import csv
import numpy as np
import tensorflow as tf
import argparse
from argparse import RawTextHelpFormatter

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences, sequence
from keras.models import load_model
from keras import backend as K

import sys
sys.path.append('..')
import util.multilabel_metrics as metrics
import util.regression_metrics as reg_metrics
import util.embedding
from util.keras_models import text_cnn_model_appraisals


parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
embdedding_config = parser.add_argument_group('embedding configuration')
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
optional.add_argument('--annotate', '-a',
            type=str,
            help='annotate a dataset using the predicted appraisals')
optional.add_argument('--savemodel', '-s',
            type=str,
            help='if you do not want to run a cross-validation you can save\n'
            'the created appraisal prediction model weights and use them in other experiments')
optional.add_argument('--loadmodel', '-l',
            type=str,
            help='test your saved models\n'
            'The dataset specified with the --dataset command will be your test set')
embdedding_config.add_argument('--loadembedding', '-le',
            type=str,
            help='path to your embedding file created for your dataset')
embdedding_config.add_argument('--createembedding', '-ce',
            type=str,
            help='path to the embedding file for your dataset, which will be created\n'
                 'with all words in your dataset')
embdedding_config.add_argument('--embeddingpath', '-ep',
            type=str,
            default='glove.840B.300d.txt',
            help='path to your pre-trained embedding download (e.g. glove300)\n'
                 'note that only 300 dimensional embeddings are supported')
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
parameter_config.add_argument('--continous', '-c',
            action='store_true',
            help='use this if you are using continous-valued appraisal annotation\n'
                 'instead of binary')
backend_setup.add_argument('--format',
            type=str,
            choices=['text', 'latex', 'markdown'],
            default='text',
            help='final result output format (default: text)')
backend_setup.add_argument('--gpu',
            action='store_true',
            help='force to run experiment on GPU')
backend_setup.add_argument('--cpu',
            action='store_true',
            help='force to run experiment on CPU')
backend_setup.add_argument('--quiet',
            action='store_true',
            help='reduce keras and info outputs')
backend_setup.add_argument('--debug',
            action='store_true',
            help='show tensorflow backend informations')
args = parser.parse_args()


### Parameters #################################################################
DROPOUT = 0.5
CONV_FILTERS = 128
FILTER_SIZE = [2,3,4]
OPTIMIZER = 'adam'

EMBEDDING_DIMS = 300       # Embedding dimension
MAX_SEQUENCE_LENGTH = 500  # Maximum length of an instance (in words)
MAX_NUM_WORDS = 50000      # Maximum number of words in the dataset

EPOCHS = args.epochs
BATCH_SIZE = args.batchsize
KFOLDS = args.folds
ROUNDS = args.runs

if (args.dataset=='enISEAR_V1' or args.dataset=='enISEAR_V2' or args.dataset=='enISEAR_V3'):
    EMBEDDING_FILE = '../embeddings/enISEAR_glove_embedding.npy'
else:
    if (args.loadembedding):
        EMBEDDING_FILE = args.loadembedding
    elif (not args.createembedding and not args.loadmodel):
        print('\nERROR: You need to specify a prepared embdedding file if you want ')
        print('       to run the experiment on other datasets other than enISEAR')
        print('       use --createembedding (-ce) to create an embedding for your dataset or')
        print('       use --loadembedding to load an embedding.')
        print('\n Exiting.')
        sys.exit(1)

if (args.embeddingpath):
    EMBEDDING_PATH = args.embeddingpath
################################################################################

# Hide keras output
if (args.quiet):
    VERBOSITY = 0
else: VERBOSITY = 1

# Hide tensorflow warnings
if (not args.debug):
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

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

# Manage preconfigured enISEAR datasets
if (args.dataset == 'enISEAR_V1'):
    DATASET = '../corpora/enISEAR-appraisal-V1/enISEAR_appraisal_majority.tsv'
    APPRAISALS = ['Attention', 'Certainty', 'Effort', 'Pleasant', 'Responsibility', 'Control', 'Circumstance']
elif (args.dataset == 'enISEAR_V2'):
    APPRAISALS = ['Attention', 'Certainty', 'Effort', 'Pleasant', 'Resp./Control', 'Situational Control']
    DATASET = '../corpora/enISEAR-appraisal-V2/enISEAR_appraisals_automated_continous.tsv'
    args.continous = True
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

# Load dataset with class_labels, text instances and appraisals
try:
    pandas_frame = pd.read_csv(DATASET, sep='\t')
    text_instances = pandas_frame['Sentence']
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
            # continous valued
            vector_row = [float(row[2]),float(row[3]),float(row[4]),float(row[5]),float(row[6]),float(row[7])]
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
    print("Unexpected error: %s in line %d" % (sys.exc_info()[1], sys.exc_info()[-1].tb_lineno))
    sys.exit(1)

# Load test set if desired
if args.testset:
    try:
        pandas_frame = pd.read_csv(args.testset, sep='\t')
        text_instances_test = pandas_frame['Sentence']
        appraisals_test = []
        with open(args.testset) as tsvfile:
          reader = csv.reader(tsvfile, delimiter='\t')
          firstLine = True
          for row in reader:
            if firstLine:# Skip first line
                firstLine = False
                continue
            vector_row = [row[2],row[3],row[4],row[5],row[6],row[7]]
            appraisals_test.append(vector_row)
        appraisals_test = np.array(appraisals_test)
        if (args.continous == True):
            appraisals_test = appraisals_test.astype(float)
        else:
            appraisals_test = appraisals_test.astype(int)
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
        print("Unexpected error: %s in line %d" % (sys.exc_info()[1], sys.exc_info()[-1].tb_lineno))
        sys.exit(1)

# Load dataset which will be annotated
if args.annotate:
    try:
        pandas_frame = pd.read_csv(args.annotate, sep='\t')
        text_instances_annotate = pandas_frame['Sentence']
    except FileNotFoundError:
        print('\nERROR: The test set "%s" was not found.' % args.annotate)
        print('       Make sure you enter the correct path and filename with extension.')
        print('\nExiting.')
        sys.exit(1)
    except IndexError:
        print('\nERROR: The test set "%s" seems to have a wrong format.' % args.annotate)
        print('       Make sure your dataset has the correct format, i.e.')
        print('       "Prior_Emotion,Sentence" or just "Sentence"')
        print('\nExiting.')
        sys.exit(1)
    except:
        print("Unexpected error: %s in line %d" % (sys.exc_info()[1], sys.exc_info()[-1].tb_lineno))
        sys.exit(1)

print('----------------------------------------------------------')
print('    Starting appraisal prediction from text experiment    ')
print('----------------------------------------------------------')
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
print('   Epochs       :', EPOCHS)
print('   Batch Size   :', BATCH_SIZE)
if (not args.testset):
    print('   Folds        :', KFOLDS)
    print('   Runs         :', ROUNDS)
print('---------------------------------------------------------- \n')


# Preprocessing dataset
# To lower case, remove , and .
text_instances = text_instances.str.lower().str.replace(".","").str.replace(",","")
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, lower=True)
tokenizer.fit_on_texts(text_instances)
vocab_size = len(tokenizer.word_index) + 1
word_index = tokenizer.word_index
num_words = min(MAX_NUM_WORDS, vocab_size)

text_instances = tokenizer.texts_to_sequences(text_instances)
text_instances_padded = sequence.pad_sequences(
            text_instances, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

if (args.testset):
    text_instances_test = text_instances_test.str.lower().str.replace(".","").str.replace(",","")
    text_instances_test = tokenizer.texts_to_sequences(text_instances_test)
    text_instances_padded_test = sequence.pad_sequences(
        text_instances_test, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

if (args.annotate):
    text_instances_annotate = text_instances_annotate.str.lower().str.replace(".","").str.replace(",","")
    text_instances_annotate = tokenizer.texts_to_sequences(text_instances_annotate)
    text_instances_padded_annotate = sequence.pad_sequences(
        text_instances_annotate, maxlen=MAX_SEQUENCE_LENGTH, padding='post')


# Prepare embedding
if (args.createembedding):
    if (not args.createembedding.endswith('.npy')):
        args.createembedding = args.createembedding + '.npy'
    embedding_matrix = util.embedding.prepareEmbeddings(word_index,
                        MAX_NUM_WORDS, args.embeddingpath, EMBEDDING_DIMS)
    util.embedding.saveEmbedding(embedding_matrix, args.createembedding)
elif (not args.loadmodel):
    embedding_matrix = util.embedding.loadEmbedding(EMBEDDING_FILE)


if (args.continous):
    class_weight = {}
    for i in range(len(APPRAISALS)):
        class_weight[i] = 1
else:
    # Predicting appraisals is an unbalanced classification problem
    # Calculate appraisal weights based on occurences in dataset (train set)
    appraisals = appraisals.astype(int)
    appraisal_counts = list(np.sum(appraisals, axis=0))
    _max = max(appraisal_counts)
    if (args.dataset == 'enISEAR_V1'):
        class_weight = {0: _max / appraisal_counts[0], # 1.131
                        1: _max / appraisal_counts[1], # 1.000
                        2: _max / appraisal_counts[2], # 1.903
                        3: _max / appraisal_counts[3], # 5.107
                        4: _max / appraisal_counts[4], # 2.019
                        5: _max / appraisal_counts[5], # 3.338
                        6: _max / appraisal_counts[6]} # 3.171
    else:
        class_weight = {}
        for i in range(len(APPRAISALS)):
            class_weight[i] = _max / appraisal_counts[i]


if (args.continous):
    cs = MinMaxScaler()
    appraisals = cs.fit_transform(appraisals)
    if (args.testset):
        appraisals_test = cs.transform(appraisals_test)

if (not args.quiet and not args.continous):
    print('\nINFO: Using the following weights for appraisal prediction')
    print('      based on the appraisal annotation counts in your dataset:')
    _class_weight = list(class_weight.values())
    maxlen = len(max(APPRAISALS, key=len))
    _APPRAISALS = [(x + (maxlen - len(x))* ' ') for x in APPRAISALS]
    for i in range(len(APPRAISALS)):
        print('\t%s %dx \t(weight %0.4f)' % (_APPRAISALS[i], appraisal_counts[i], _class_weight[i]))
    print('')


def performCrossValidation(x_data, y_data):
    if (args.continous):
        _reg_metrics = reg_metrics.metrics_regression(APPRAISALS, 2)
    else:
        _metrics = metrics.metrics(APPRAISALS, 2)

    for seed in range(ROUNDS):
        np.random.seed(seed)

        kfold = KFold(n_splits=KFOLDS, shuffle=True, random_state=seed)
        for train, test in kfold.split(x_data, y_data):
            K.clear_session()

            if (args.continous):
                activation = 'linear'  # Use linear activation
                # activation = 'sigmoid' # Use linear activation
                loss = 'mse'           # Use mean squared error loss
                metric = ['mse', 'mae']
            else:
                activation = 'sigmoid'
                loss = 'binary_crossentropy'
                metric = ['accuracy']

            appraisal_predictor = text_cnn_model_appraisals(MAX_SEQUENCE_LENGTH, vocab_size,
                        EMBEDDING_DIMS, FILTER_SIZE, CONV_FILTERS,
                        embedding_matrix, DROPOUT, len(APPRAISALS), activation)
            appraisal_predictor.compile(OPTIMIZER, loss, metrics=metric)

            if (args.quiet):
                for _ in range(EPOCHS):
                    appraisal_predictor.fit(text_instances_padded[train], appraisals[train],
                                    batch_size=BATCH_SIZE, epochs=1,
                                    verbose=VERBOSITY, class_weight=class_weight)
                    print('.', end='', flush=True)
            else:
                appraisal_predictor.fit(text_instances_padded[train], appraisals[train],
                                batch_size=BATCH_SIZE, epochs=EPOCHS,
                                verbose=VERBOSITY, class_weight=class_weight)

            if (args.continous==False):
                # weights = [0.55, 0.65, 0.48, 0.3, 0.425, 0.4, 0.45] # Some experimental settings
                weights = [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50]
                test_instances = text_instances_padded[test]
                preds = appraisal_predictor.predict(test_instances)
                results = []
                for row in range(len(preds)):
                    res = []
                    for dim in range(len(APPRAISALS)):
                        value = preds[row][dim]
                        if (value >= weights[dim]):
                            value = 1
                        else:
                            value = 0
                        res.append(value)
                    results.append(res)
                appraisal_predictions = np.array(results)
                _metrics.evaluateFold(appraisal_predictions, appraisals[test])
            else:
                test_instances = text_instances_padded[test]
                appraisal_test = appraisals[test]
                preds = appraisal_predictor.predict(test_instances)
                # for i in range(len(preds)):
                    # print('\n Predicted:', preds[i])
                    # print('  Correct: ', appraisal_test[i])
                appraisal_predictions = np.array(preds)
                _reg_metrics.evaluateFold(appraisal_predictions, appraisals[test])

    if (args.continous == False):
        if (args.format):
            _metrics.showFinalResults(args.format)
        else: _metrics.showFinalResults(format='text')
    else:
        _reg_metrics.showResults()

def evalTrainAndTestSet(x_train, x_test, y_train, y_test):
    if (args.continous):
        _reg_metrics = reg_metrics.metrics_regression(APPRAISALS, 2)
        activation = 'linear'  # Use linear activation
        # activation = 'sigmoid' # Use linear activation
        loss = 'mse'           # Use mean squared error loss
        metric = ['mse', 'mae']
    else:
        _metrics = metrics.metrics(APPRAISALS, 2)
        activation = 'sigmoid'
        loss = 'binary_crossentropy'
        metric = ['accuracy']

    appraisal_predictor = text_cnn_model_appraisals(MAX_SEQUENCE_LENGTH, vocab_size,
                EMBEDDING_DIMS, FILTER_SIZE, CONV_FILTERS,
                embedding_matrix, DROPOUT, len(APPRAISALS), activation)
    appraisal_predictor.compile(OPTIMIZER, loss, metrics=metric)

    # appraisals_shaped = np.expand_dims(appraisals, axis=2)
    print('\nINFO: Training on %d instances...' % len(x_train))
    if (args.quiet):
        for _ in range(EPOCHS):
            appraisal_predictor.fit(x_train,  y_train, batch_size=BATCH_SIZE, epochs=1, verbose=VERBOSITY)
            print('.', end='', flush=True)
    else:
        appraisal_predictor.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSITY)


    print('\nINFO: Testing on %d instances...' % len(x_test))
    if (args.continous==False):
        # weights = [0.55, 0.65, 0.48, 0.3, 0.425, 0.4, 0.45] # Some experimental settings
        weights = [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50]
        preds = appraisal_predictor.predict(x_test)
        results = []
        for row in range(len(preds)):
            res = []
            for dim in range(len(APPRAISALS)):
                value = preds[row][dim]
                if (value >= weights[dim]):
                    value = 1
                else:
                    value = 0
                res.append(value)
            results.append(res)
        appraisal_predictions = np.array(results)
        _metrics.evaluateFold(appraisal_predictions, y_test)
    else:
        preds = appraisal_predictor.predict(x_test)
        # for i in range(len(preds)):
            # print('\n Predicted:', preds[i])
            # print('  Correct: ', appraisal_test[i])
        appraisal_predictions = np.array(preds)
        _reg_metrics.evaluateFold(appraisal_predictions, y_test)

    if (args.savemodel):
        if (not args.savemodel.endswith('.h5')):
            args.savemodel += '.h5'
        model.save(args.savemodel)
        print('\nINFO: Saved model to %s' % args.savemodel)

def trainAndSaveModel(x_train, y_train):
    if (args.continous):
        activation = 'linear'  # Use linear activation
        # activation = 'sigmoid' # Use linear activation
        loss = 'mse'           # Use mean squared error loss
        metric = ['mse', 'mae']
    else:
        activation = 'sigmoid'
        loss = 'binary_crossentropy'
        metric = ['accuracy']

    appraisal_predictor = text_cnn_model_appraisals(MAX_SEQUENCE_LENGTH, vocab_size,
                EMBEDDING_DIMS, FILTER_SIZE, CONV_FILTERS,
                embedding_matrix, DROPOUT, len(APPRAISALS), activation)
    appraisal_predictor.compile(OPTIMIZER, loss, metrics=metric)

    # appraisals_shaped = np.expand_dims(appraisals, axis=2)
    print('\nINFO: Training on %d instances...' % len(x_train))
    if (args.quiet):
        for _ in range(EPOCHS):
            appraisal_predictor.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1, verbose=VERBOSITY)
            print('.', end='', flush=True)
    else:
        appraisal_predictor.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSITY)

    if (not args.savemodel.endswith('.h5')):
        print('\nINFO: Your model does not end with ".h5".')
        print('      Automatically appending file extension.')
        args.savemodel += '.h5'
    appraisal_predictor.save(args.savemodel)
    print('\nSUCCESS: Saved model to %s' % args.savemodel)

def testModel(x_test, y_test):
    _metrics = metrics.metrics(APPRAISALS, 2)
    if (not args.loadmodel.endswith('.h5')):
        args.loadmodel += '.h5'
    try:
        model = load_model(args.loadmodel)
        print('INFO: Loaded model weights from %s' % args.loadmodel)
    except:
        print('\nUnexpected error:', sys.exc_info()[1])
        sys.exit(1)

    print('\nINFO: Testing on %d instances...' % len(x_test))
    if (args.continous==False):
        # weights = [0.55, 0.65, 0.48, 0.3, 0.425, 0.4, 0.45] # Some experimental settings
        weights = [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50]
        preds = appraisal_predictor.predict(x_test)
        results = []
        for row in range(len(preds)):
            res = []
            for dim in range(len(APPRAISALS)):
                value = preds[row][dim]
                if (value >= weights[dim]):
                    value = 1
                else:
                    value = 0
                res.append(value)
            results.append(res)
        appraisal_predictions = np.array(results)
        _metrics.evaluateFold(appraisal_predictions, y_test)
    else:
        preds = appraisal_predictor.predict(x_test)
        # for i in range(len(preds)):
            # print('\n Predicted:', preds[i])
            # print('  Correct: ', appraisal_test[i])
        appraisal_predictions = np.array(preds)
        _reg_metrics.evaluateFold(appraisal_predictions, y_test)

def annotatePredictedAppraisals(text_instances_padded,
                                text_instances_padded_annotate,
                                appraisals):
    print('INFO: Annotating Dataset')
    if (args.continous):
        _reg_metrics = reg_metrics.metrics_regression(APPRAISALS, 2)
        activation = 'linear'  # Use linear activation
        # activation = 'sigmoid' # Use linear activation
        loss = 'mse'           # Use mean squared error loss
        metric = ['mse', 'mae']
    else:
        _metrics = metrics.metrics(APPRAISALS, 2)
        activation = 'sigmoid'
        loss = 'binary_crossentropy'
        metric = ['accuracy']

    appraisal_predictor = text_cnn_model_appraisals(MAX_SEQUENCE_LENGTH, vocab_size,
                EMBEDDING_DIMS, FILTER_SIZE, CONV_FILTERS,
                embedding_matrix, DROPOUT, len(APPRAISALS), activation)
    appraisal_predictor.compile(OPTIMIZER, loss, metrics=metric)
    appraisal_predictor.fit(text_instances_padded, appraisals,
                            batch_size=BATCH_SIZE, epochs=EPOCHS,
                            verbose=VERBOSITY)

    if (args.continous==False):
        weights = [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50]
        preds = appraisal_predictor.predict(text_instances_padded_annotate)
        results = []
        for row in range(len(preds)):
            res = []
            for dim in range(len(APPRAISALS)):
                value = preds[row][dim]
                if (value >= weights[dim]):
                    value = 1
                else:
                    value = 0
                res.append(value)
            results.append(res)
        appraisal_predictions = list(results)
    else:
        preds = appraisal_predictor.predict(text_instances_padded_annotate)
        appraisal_predictions = list(preds)

    out_file_name = args.annotate[:len(args.annotate)-4] + '_appraisals.tsv'
    first_line = True
    with open(out_file_name, 'w') as out_file:
        with open(args.annotate, 'r') as in_file:
            for i, line in enumerate(in_file):
                if (first_line):
                    annotation = ''
                    for dimension in APPRAISALS:
                        annotation += '\t' + dimension
                    out_file.write(line.rstrip('\n') + str(annotation) + '\n')
                    first_line = False
                else:
                    annotation = ''
                    for p in range(len(APPRAISALS)):
                        annotation += '\t' + str(appraisal_predictions[i-1][p])
                    out_file.write(line.rstrip('\n') + annotation + '\n')

# Force to run experiment on cpu or gpu
# otherwise let backend choose (prefers gpu if available)
if (args.gpu or args.cpu):
    if (args.gpu):
        print('\nINFO: Forcing experiment to run on GPU (GPU:0)')
        DEVICE = 'GPU:0'
    elif (args.cpu):
        print('\nINFO: Forcing experiment to run on CPU (CPU:0)')
        DEVICE = 'CPU:0'
    tf.debugging.set_log_device_placement(True)
    with tf.device(DEVICE):
        if args.testset:
            evalTrainAndTestSet(text_instances_padded, text_instances_padded_test,
                                appraisals, appraisals_test)
        elif args.annotate:
            annotatePredictedAppraisals(text_instances_padded,
                                text_instances_padded_annotate,
                                appraisals)
        elif args.savemodel:
            trainAndSaveModel(text_instances_padded, appraisals)
        elif args.loadmodel:
            testModel(x_train, y_train)
        else:
            print('\nINFO: Starting Cross-Validation')
            performCrossValidation(text_instances_padded, appraisals)
else:
    if args.testset:
        evalTrainAndTestSet(text_instances_padded, text_instances_padded_test,
                            appraisals, appraisals_test)
    elif args.annotate:
        annotatePredictedAppraisals(text_instances_padded,
                            text_instances_padded_annotate,
                            appraisals)
    elif args.savemodel:
        trainAndSaveModel(text_instances_padded, appraisals)
    elif args.loadmodel:
        testModel(text_instances_padded, appraisals)
    else:
        print('\nINFO: Starting Cross-Validation')
        performCrossValidation(text_instances_padded, appraisals)
