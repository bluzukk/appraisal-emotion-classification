#!/usr/bin/env python
"""
Pipeline first predicting appraisals from text and then emotions
from predicted appraisals

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
import util.metrics as metrics
import util.embedding
from util.keras_models import text_cnn_model_appraisals, shallowNN_emotions_from_dimensions


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
optional.add_argument('--savemodel', '-s',
            type=str,
            nargs='+',
            help='If you do not want to run a cross-validation you can save\n'
            'the created emotion prediction model weights and use them in other experiments\n'
            'usage: --savemodel <APPRAISAL_MODEL.h5> <EMOTION_MODEL.h5>')
optional.add_argument('--loadmodel', '-l',
            type=str,
            nargs='+',
            help='test your saved models\n'
            'The dataset specified with the --dataset command will be your test set\n'
            'usage: --loadmodel <APPRAISAL_MODEL.h5> <EMOTION_MODEL.h5>')
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
parameter_config.add_argument('--epochs_appraisal', '-ea',
            default=20, type=int,
            help='set the number training epochs (default: 20)')
parameter_config.add_argument('--epochs_emotion', '-ee',
            default=10, type=int,
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

EMBEDDING_DIMS = 300         # Embedding dimension
MAX_SEQUENCE_LENGTH = 500    # Maximum length of an instance (in words)
MAX_NUM_WORDS = 50000        # Maximum number of words in the dataset
MAX_NUM_WORDS = 200000

EPOCHS_A = args.epochs_appraisal
EPOCHS_E = args.epochs_emotion
BATCH_SIZE = args.batchsize
KFOLDS = args.folds
ROUNDS = args.runs

if (args.dataset=='enISEAR_V1' or args.dataset=='enISEAR_V2' or args.dataset=='enISEAR_V3'):
    EMBEDDING_FILE = '../embeddings/enISEAR_glove_embedding.npy'
else:
    if (args.loadembedding):
        EMBEDDING_FILE = args.loadembedding
    elif (not args.createembedding):
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
    APPRAISALS = ['Attention', 'Certainty', 'Effort', 'Pleasant', 'Resp./Control', 'Situational Control']


# Load dataset with class_labels, text instances and appraisals
try:
    pandas_frame = pd.read_csv(DATASET, sep='\t')
    text_instances = pandas_frame['Sentence']
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
        # Manage additional appraisal in enISEAR_V1 dataset
        if (args.dataset == 'enISEAR_V1'):
            vector_row = [int(row[3]),int(row[4]),int(row[5]),int(row[6]),int(row[7]),int(row[8]),int(row[9])]
        elif (args.dataset == 'enISEAR_V2'):
            # continous valued
            vector_row = [float(row[2]),float(row[3]),float(row[4]),float(row[5]),float(row[6]),float(row[7])]
        elif (args.dataset == 'enISEAR_V3'):
            vector_row = [int(row[3]),int(row[4]),int(row[5]),int(row[6]),int(row[7]),int(row[8])]
        else:
            if (args.continous):
                vector_row = [float(row[2]),float(row[3]),float(row[4]),float(row[5]),float(row[6]),float(row[7])]
            else:
                vector_row = [int(row[2]),int(row[3]),int(row[4]),int(row[5]),int(row[6]),int(row[7])]

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
except ValueError:
    print('\nERROR: The dataset "%s" seems to have a wrong format.' % DATASET)
    print('       Make sure you use the the --continous (-c) flag')
    print('       if you are using continous valued annotations.')
    print('\nExiting.')
    sys.exit(1)
except:
    print("Unexpected error: %s in line %d" % (sys.exc_info()[1], sys.exc_info()[-1].tb_lineno))
    print('\nExiting.')
    sys.exit(1)

# Load test set if desired
if args.testset:
    try:
        pandas_frame = pd.read_csv(args.testset, sep='\t')
        text_instances_test = pandas_frame['Sentence']
        class_labels_test = pandas_frame['Prior_Emotion']
        LABELS_test = sorted(list(class_labels.unique()))
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
print('   Epochs Appr. :', EPOCHS_A)
print('   Epochs Emo.  :', EPOCHS_E)
print('   Batch Size   :', BATCH_SIZE)
if (not args.testset):
    print('   Folds        :', KFOLDS)
    print('   Runs         :', ROUNDS)
print('------------------------------------------------------------\n')


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


# Prepare embedding
if (args.createembedding):
    if (not args.createembedding.endswith('.npy')):
        args.createembedding = args.createembedding + '.npy'
    embedding_matrix = util.embedding.prepareEmbeddings(word_index,
                        MAX_NUM_WORDS, args.embeddingpath, EMBEDDING_DIMS)
    util.embedding.saveEmbedding(embedding_matrix, args.createembedding)
else: embedding_matrix = util.embedding.loadEmbedding(EMBEDDING_FILE)


if (args.continous):
    class_weight = {}
    for i in range(len(APPRAISALS)):
        class_weight[i] = 1
else:
    # Predicting appraisals is an unbalanced classification problem
    # Calculate appraisal weights based on occurences in dataset (train set)
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

if (not args.quiet and not args.continous):
    print('\nINFO: Using the following weights for appraisal prediction')
    print('      based on the appraisal annotation counts in your dataset:')
    _class_weight = list(class_weight.values())
    maxlen = len(max(APPRAISALS, key=len))
    _APPRAISALS = [(x + (maxlen - len(x))* ' ') for x in APPRAISALS]
    for i in range(len(APPRAISALS)):
        print('\t%s %dx \t(weight %0.4f)' % (_APPRAISALS[i], appraisal_counts[i], _class_weight[i]))
    print('')

def performCrossValidation(x_data, y_data_appraisal, y_data_emotion):
    metrics_final = metrics.metrics(None, None, LABELS, 2)
    percentage_done = 1
    for seed in range(ROUNDS):
        np.random.seed(seed)

        kfold = KFold(n_splits=KFOLDS, shuffle=True, random_state=seed)
        for train, test in kfold.split(x_data, y_data_emotion):
            K.clear_session()

            emotions_train = pd.concat([y_data_emotion[train], pd.get_dummies(y_data_emotion[train])],axis=1).drop(['Prior_Emotion'],axis=1)
            emotions_test = pd.concat([y_data_emotion[test], pd.get_dummies(y_data_emotion[test])],axis=1).drop(['Prior_Emotion'],axis=1)

            ####################################################################
            # Task 1 : Learn to predict appraisals from text
            ####################################################################
            if (args.continous):
                activation = 'linear'  # Use linear activation
                # activation = 'sigmoid' # Use linear activation
                loss = 'mse'           # Use mean squared error loss
                metric = ['mse', 'mae']
            else:
                activation = 'sigmoid'
                loss = 'binary_crossentropy'
                metric = ['accuracy']


            print('\nINFO: Learning to predict appraisals from text...')
            appraisal_predictor = text_cnn_model_appraisals(MAX_SEQUENCE_LENGTH, vocab_size,
                        EMBEDDING_DIMS, FILTER_SIZE, CONV_FILTERS,
                        embedding_matrix, DROPOUT, len(APPRAISALS), activation)
            appraisal_predictor.compile(OPTIMIZER, loss, metrics=metric)

            if (args.quiet):
                for _ in range(EPOCHS_A):
                    appraisal_predictor.fit(text_instances_padded[train], appraisals[train],
                                    batch_size=BATCH_SIZE, epochs=1,
                                    verbose=VERBOSITY, class_weight=class_weight)
                    print('.', end='', flush=True)
            else:
                appraisal_predictor.fit(text_instances_padded[train], appraisals[train],
                                batch_size=BATCH_SIZE, epochs=EPOCHS_A,
                                verbose=VERBOSITY, class_weight=class_weight)

            if (args.continous == False):
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
            else:
                test_instances = text_instances_padded[test]
                appraisal_test = appraisals[test]
                preds = appraisal_predictor.predict(test_instances)
                # for i in range(len(preds)):
                    # print('\n Predicted:', preds[i])
                    # print('  Correct: ', appraisal_test[i])
                appraisal_predictions = np.array(preds)

            ####################################################################
            # Task 2 : Learn to predict emotions from appraisals
            ####################################################################
            print('\nINFO: Learning to predict emotions from appraisals...')
            emotion_predictor = shallowNN_emotions_from_dimensions(
                    len(APPRAISALS), CONV_FILTERS, DROPOUT, LABELS, 'softmax')

            emotion_predictor.compile(OPTIMIZER, 'categorical_crossentropy', metrics=['accuracy'])
            # vectors_shaped = np.expand_dims(vectors[train], axis=2)
            if (args.quiet):
                for _ in range(EPOCHS_E):
                    emotion_predictor.fit(appraisals[train], emotions_train,
                                    batch_size=BATCH_SIZE, epochs=1,
                                    verbose=VERBOSITY)
                    print('.', end='', flush=True)
            else:
                emotion_predictor.fit(appraisals[train], emotions_train,
                                batch_size=BATCH_SIZE, epochs=EPOCHS_E,
                                verbose=VERBOSITY)

            # Predict emotions based on appraisal predictions
            predicted_emotions = []
            # results = np.expand_dims(results, axis=2)
            emotions_predictions = emotion_predictor.predict(appraisal_predictions)
            for i in range(len(emotions_predictions)):
                index = np.argmax(emotions_predictions[i])
                predicted_emotions.append(LABELS[index])

            # Show results
            print('\n\nINFO: Evaluating CV-fold...')
            metrics_fold = metrics.metrics(y_data_emotion[test], predicted_emotions, LABELS, 2)
            metrics_fold.showResults()
            metrics_final.addIntermediateResults(y_data_emotion[test], predicted_emotions)
    metrics_final.showResults()
    metrics_final.showConfusionMatrix(False)


def evalTrainAndTestSet(text_instances_padded, text_instances_padded_test,
            appraisals_train, appraisals_test, class_labels_train, class_labels_test):
    emotions_train = pd.concat([class_labels_train, pd.get_dummies(class_labels_train)],axis=1).drop(['Prior_Emotion'],axis=1)
    emotions_test = pd.concat([class_labels_test, pd.get_dummies(class_labels_test)],axis=1).drop(['Prior_Emotion'],axis=1)

    print('\nINFO: Learning to predict emototions on %d instances...' % len(text_instances_padded))
    emotion_predictor = shallowNN_emotions_from_dimensions(
        len(APPRAISALS), CONV_FILTERS, DROPOUT, LABELS, 'softmax')

    emotion_predictor.compile(OPTIMIZER, 'categorical_crossentropy', metrics=['accuracy'])

    # vectors_shaped = np.expand_dims(vectors[train], axis=2)
    if (args.quiet):
        for _ in range(EPOCHS_E):
            emotion_predictor.fit(appraisals_train, emotions_train,
                            batch_size=BATCH_SIZE, epochs=1,
                            verbose=VERBOSITY)
            print('.', end='', flush=True)
    else:
        emotion_predictor.fit(appraisals_train, emotions_train,
                        batch_size=BATCH_SIZE, epochs=EPOCHS_E,
                        verbose=VERBOSITY)

    if (args.continous):
        activation = 'linear'  # Use linear activation
        activation = 'sigmoid' # Use sigmoid activation
        loss = 'mse'           # Use mean squared error loss
        metric = ['mse', 'mae']
    else:
        activation = 'sigmoid'
        loss = 'binary_crossentropy'
        metric = ['accuracy']

    appraisal_predictor = text_cnn_model_appraisals(MAX_SEQUENCE_LENGTH,
                            vocab_size, EMBEDDING_DIMS,
                            FILTER_SIZE, CONV_FILTERS,
                            embedding_matrix, DROPOUT,
                            len(APPRAISALS), activation)
    appraisal_predictor.compile(OPTIMIZER, loss, metrics=metric)

    # appraisals_shaped = np.expand_dims(appraisals, axis=2)
    print('\nINFO: Learning to predict appraisals on %d instances...' % len(text_instances_padded))
    if (args.quiet):
        for _ in range(EPOCHS_A):
            appraisal_predictor.fit(text_instances_padded, appraisals_train, batch_size=BATCH_SIZE, epochs=1, verbose=VERBOSITY)
            print('.', end='', flush=True)
    else:
        appraisal_predictor.fit(text_instances_padded,appraisals_train, batch_size=BATCH_SIZE, epochs=EPOCHS_A, verbose=VERBOSITY)

    print('\nINFO: Testing on %d instances...' % len(class_labels_test))

    if (args.continous == False):
        # weights = [0.55, 0.65, 0.48, 0.3, 0.425, 0.4, 0.45] # Some experimental settings
        weights = [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50]
        preds = appraisal_predictor.predict(text_instances_padded_test)
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
    else:
        preds = appraisal_predictor.predict(text_instances_padded_test)
        appraisal_predictions = np.array(preds)

    # Predict emotions based on appraisal predictions
    predicted_emotions = []
    # results = np.expand_dims(results, axis=2)
    emotions_predictions = emotion_predictor.predict(appraisal_predictions)
    for i in range(len(emotions_predictions)):
        index = np.argmax(emotions_predictions[i])
        predicted_emotions.append(LABELS[index])

    _metrics = metrics.metrics(class_labels_test, predicted_emotions, LABELS, 2)
    _metrics.showResults()

def trainAndSaveModel(x_train, y_appraisal, y_emotion):
    if (args.continous):
        activation = 'linear'  # Use linear activation
        activation = 'sigmoid' # Use sigmoid activation
        loss = 'mse'           # Use mean squared error loss
        metric = ['mse', 'mae']
    else:
        activation = 'sigmoid'
        loss = 'binary_crossentropy'
        metric = ['accuracy']

    model = text_cnn_model_appraisals(MAX_SEQUENCE_LENGTH,
                            vocab_size, EMBEDDING_DIMS,
                            FILTER_SIZE, CONV_FILTERS,
                            embedding_matrix, DROPOUT,
                            len(APPRAISALS), activation)
    model.compile(OPTIMIZER, loss, metrics=metric)


    # appraisals_shaped = np.expand_dims(appraisals, axis=2)
    print('\nINFO: Learning to predict appraisals on %d instances...' % len(x_train))
    if (args.quiet):
        for _ in range(EPOCHS_A):
            model.fit(x_train, y_appraisal, batch_size=BATCH_SIZE, epochs=1, verbose=VERBOSITY)
            print('.', end='', flush=True)
    else:
        model.fit(x_train, y_appraisal, batch_size=BATCH_SIZE, epochs=EPOCHS_A, verbose=VERBOSITY)

    if (not args.savemodel[0].endswith('.h5')):
        print('\nINFO: Your appraisal model does not end with ".h5".')
        print('      Automatically appending file extension.')
        args.savemodel[0] += '.h5'
    model.save(args.savemodel[0])
    print('\nINFO: Saved appraisal prediction model to %s' % args.savemodel[0])

    emotion_predictor = shallowNN_emotions_from_dimensions(
            len(APPRAISALS), CONV_FILTERS, DROPOUT, LABELS, 'softmax')

    emotion_predictor.compile(OPTIMIZER, 'categorical_crossentropy', metrics=['accuracy'])
    emotions_train = pd.concat([y_emotion, pd.get_dummies(y_emotion)],axis=1).drop(['Prior_Emotion'],axis=1)

    print('\nINFO: Learning to predict emototions on %d instances...' % len(text_instances_padded))
    if (args.quiet):
        for _ in range(EPOCHS_E):
            emotion_predictor.fit(y_appraisal, emotions_train,
                            batch_size=BATCH_SIZE, epochs=1,
                            verbose=VERBOSITY)
            print('.', end='', flush=True)
    else:
        emotion_predictor.fit(y_appraisal, emotions_train,
                        batch_size=BATCH_SIZE, epochs=EPOCHS_E,
                        verbose=VERBOSITY)

    if (not args.savemodel[1].endswith('.h5')):
        print('\nINFO: Your emotion model does not end with ".h5".')
        print('      Automatically appending file extension.')
        args.savemodel[1] += '.h5'
    emotion_predictor.save(args.savemodel[1])
    print('\nINFO: Saved emotion prediction model to %s' % args.savemodel[1])

def testModel(x_test, y_test):
    _metrics = metrics.metrics(None, None, APPRAISALS, 2)
    if (not args.loadmodel[0].endswith('.h5')):
        args.loadmodel[0] += '.h5'
    try:
        appraisal_predictor = load_model(args.loadmodel[0])
        print('INFO: Loaded appraisal prediction model weights from %s' % args.loadmodel[0])
    except:
        print('\nUnexpected error:', sys.exc_info()[1])
        sys.exit(1)

    if (not args.loadmodel[1].endswith('.h5')):
        args.loadmodel[1] += '.h5'
    try:
        emotion_predictor = load_model(args.loadmodel[1])
        print('INFO: Loaded emotion prediction model weights from %s' % args.loadmodel[1])
    except:
        print('\nUnexpected error:', sys.exc_info()[1])
        sys.exit(1)

    print('\nINFO: Testing on %d instances...' % len(x_test))
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

    if (args.continous == False):
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
    else:
        preds = appraisal_predictor.predict(text_instances_padded_test)

    appraisal_predictions = np.array(results)
    emotions_predictions = emotion_predictor.predict(x_test)
    predicted_emotions = []
    for i in range(len(emotions_predictions)):
        index = np.argmax(emotions_predictions[i])
        predicted_emotions.append(LABELS[index])

    _metrics = metrics.metrics(y_test, predicted_emotions, LABELS, 2)
    _metrics.showResults()


# Force to run experiment on cpu or gpu if desired
# otherwise let backend choose (prefers gpu if available)
if (args.gpu or args.cpu):
    if (args.gpu):
        print('\nINFO: Forcing experiment to run on GPU (GPU:0)')
        DEVICE = '/device:GPU:0'
    elif (args.cpu):
        print('\nINFO: Forcing experiment to run on CPU (CPU:0)')
        DEVICE = '/device:CPU:0'
    tf.debugging.set_log_device_placement(True)
    with tf.device(DEVICE):
        if args.testset:
            evalTrainAndTestSet(text_instances_padded, text_instances_padded_test,
                        appraisals, appraisals_test, class_labels, class_labels_test)
        elif args.savemodel:
            trainAndSaveModel(text_instances_padded, appraisals, class_labels)
        elif args.loadmodel:
            testModel(text_instances_padded, class_labels)
        else:
            performCrossValidation(text_instances_padded, appraisals, class_labels)
else:
    if args.testset:
        evalTrainAndTestSet(text_instances_padded, text_instances_padded_test,
                    appraisals, appraisals_test, class_labels, class_labels_test)
    elif args.savemodel:
        trainAndSaveModel(text_instances_padded, appraisals, class_labels)
    elif args.loadmodel:
        testModel(text_instances_padded, class_labels)
    else:
        performCrossValidation(text_instances_padded, appraisals, class_labels)
