#!/usr/bin/env python
"""
CNN baseline system
"""

import pandas as pd
import numpy as np
from numpy import asarray, zeros
import tensorflow as tf

from sklearn.model_selection import KFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences, sequence

import sys
sys.path.append('..')
from util.keras_models import text_cnn_model_baseline
import util.metrics as metrics
import util.embedding

import argparse
parser = argparse.ArgumentParser()
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
embdedding_config = parser.add_argument_group('embedding configuration')
parameter_config = parser.add_argument_group('parameter configuration')
backend_setup = parser.add_argument_group('optional tensorflow setup')
required.add_argument('--dataset', '-d',
            type=str,
            help='specify a dataset', required=True)
optional.add_argument('--testset', '-t',
            type=str,
            help='if you have a test set and you don\'t want to use cross-validation\n'
            'you can specify a test set here.\n Then the --dataset argument must be your training set')
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
            default=15, type=int,
            help='set the number training epochs (default: 15)')
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
KFOLDS = args.folds
ROUNDS = args.runs

_DATASET = args.dataset
if (args.quiet):
    VERBOSITY = 0
else: VERBOSITY = 1

if (not args.debug):
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

EXPERIMENTNAME = 'Baseline TextCNN ' + _DATASET

################################################################################
## enISEAR config
################################################################################
if (args.dataset == 'enISEAR'):
    LABELS = ['Anger', 'Disgust', 'Fear', 'Guilt', 'Joy', 'Sadness', 'Shame']
    SAVEFILE = 'results_baseline_CNN_enISEAR.txt'
    EMBEDDING_FILE = '../embeddings/enISEAR_glove_embedding.npy'
    DATASET = '../corpora/enISEAR-appraisal-V1/enISEAR_appraisal_majority.tsv'
    dataframe = pd.read_csv(DATASET, sep='\t')

    DROPOUT = 0.5
    CONV_FILTERS = 128
    FILTER_SIZE = [2,3,4]
    OPTIMIZER = 'adam'
    MAX_SEQUENCE_LENGTH = 300
    MAX_NUM_WORDS = 50000
    EPOCHS = args.epochs         # 15
    BATCH_SIZE = args.batchsize  # 32

################################################################################
## ISEAR CONFIG
################################################################################
elif (args.dataset == 'ISEAR'):
    LABELS = ['Anger', 'Disgust', 'Fear', 'Guilt', 'Joy', 'Sadness', 'Shame']
    SAVEFILE = 'results_baseline_CNN_ISEAR.txt'
    EMBEDDING_FILE = '../embeddings/ISEAR_glove_embedding.npy'
    DATASET = '../corpora/ISEAR.csv'
    dataframe = pd.read_csv(DATASET, sep=',')

    DROPOUT = 0.5
    CONV_FILTERS = 128
    FILTER_SIZE = [2,3,4]
    OPTIMIZER = 'adam'
    MAX_SEQUENCE_LENGTH = 300
    MAX_NUM_WORDS = 50000
    EPOCHS = args.epochs         # 25
    BATCH_SIZE = args.batchsize  # 32

################################################################################
## TEC config
################################################################################
elif (args.dataset == 'TEC'):
    LABELS = [':: anger', ':: disgust', ':: fear', ':: joy', ':: sadness', ':: surprise']
    SAVEFILE = 'results_baseline_CNN_TEC.txt'
    embedding_file = '../embeddings/TEC_glove840B_complete_embedding.npy'
    DATASET = '../corpora/TEC_dataset.csv'
    dataframe = pd.read_csv(DATASET, sep=',')

    DROPOUT = 0.5
    CONV_FILTERS = 128
    FILTER_SIZE = [2,3,4]
    OPTIMIZER = 'adam'
    MAX_SEQUENCE_LENGTH = 300
    MAX_NUM_WORDS = 50000
    EPOCHS = args.epochs         # 25
    BATCH_SIZE = args.batchsize  # 128

################################################################################
## Other datasets
################################################################################
else:
    # print('INFO: Using dataset %s' % args.dataset)
    SAVEFILE = 'results_baseline_'+ args.dataset + '.txt'
    DATASET = args.dataset
    if (DATASET.endswith('.csv')):
        dataframe = pd.read_csv(DATASET, sep=',')
    else:
        dataframe = pd.read_csv(DATASET, sep='\t')

    if (args.testset):
        try:
            if (DATASET.endswith('.csv')):
                dataframe_test = pd.read_csv(args.testset, sep=',')
            else:
                dataframe_test = pd.read_csv(args.testset, sep='\t')
        except:
            print('ERROR: Could not read "%s", make sure you enter the correct path.' % args.testset)
            print('\nExiting.')
            sys.exit(1)
        instances_test = dataframe_test['Sentence']
        labels_test = dataframe_test['Prior_Emotion']

    EPOCHS = args.epochs
    BATCH_SIZE = args.batchsize
    DROPOUT = 0.5
    CONV_FILTERS = 128
    FILTER_SIZE = [2,3,4]
    OPTIMIZER = 'adam'
    MAX_SEQUENCE_LENGTH = 300
    MAX_NUM_WORDS = 100000

    if (args.loadembedding or (args.createembedding and args.embeddingpath)):
        pass
    else:
        print('\nERROR: You need to specify a prepared embdedding file if you want ')
        print('       to run the experiment on other datasets other than enISEAR')
        print('       use --createembedding (-ce) to create an embedding for your dataset or')
        print('       use --loadembedding to load an embedding.')
        print('\n Exiting.')
        sys.exit(1)

# Load dataset
instances = dataframe['Sentence']
labels = dataframe['Prior_Emotion']
LABELS = sorted(list(labels.unique()))
# print('\nINFO: Detected labels: %s' % LABELS)

print('-------------------------------')
print(' Starting baseline experiment ')
print('-------------------------------')
print('   Dataset      :', DATASET)
print('   Instances    :', len(instances))
print('')
print('   Class Labels :', LABELS)
print('   Epochs       :', EPOCHS)
print('   Batch Size   :', BATCH_SIZE)
if (not args.testset):
    print('   Folds        :', KFOLDS)
    print('   Runs         :', ROUNDS)
print('-------------------------------\n')

# Tokenize and create word index
print('INFO: Loading Dataset')
instances = instances.str.lower().str.replace('.','').str.replace(',','')
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, lower=True)
tokenizer.fit_on_texts(instances)
vocab_size = len(tokenizer.word_index) + 1
word_index = tokenizer.word_index
num_words = min(MAX_NUM_WORDS, vocab_size)

instances_sequences = tokenizer.texts_to_sequences(instances)
instances_padded = sequence.pad_sequences(instances_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

if (args.testset):
    instances_test = instances_test.str.lower().str.replace('.','').str.replace(',','')
    instances_sequences_test = tokenizer.texts_to_sequences(instances_test)
    instances_padded_test = sequence.pad_sequences(instances_sequences_test, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

# Prepare embedding if not present yet
EMBEDDING_DIMS = 300
if (args.createembedding):
    if (not args.createembedding.endswith('.npy')):
        args.createembedding += '.npy'
    embedding_matrix = util.embedding.prepareEmbeddings(word_index, MAX_NUM_WORDS, args.embeddingpath, EMBEDDING_DIMS)
    util.embedding.saveEmbedding(embedding_matrix, args.createembedding)
else:
    if (args.loadembedding):
        if (not args.loadembedding.endswith('.npy')):
            args.loadembedding += '.npy'
        EMBEDDING_FILE = args.loadembedding

    print('\nINFO: Loading embedding' + EMBEDDING_FILE)
    embedding_matrix = util.embedding.loadEmbedding(EMBEDDING_FILE)


def performCrossValidation(x_data, y_data):
    metrics_final = metrics.metrics(None, None, LABELS, 2)

    # i.e seed in {0..9}
    for seed in range(ROUNDS):
        np.random.seed(seed)
        kfold = KFold(n_splits=KFOLDS, shuffle=True, random_state=seed)

        for train, test in kfold.split(x_data, y_data):
            # Create ohe-encoding
            classes_train = pd.concat([y_data[train], pd.get_dummies(y_data[train])],axis=1).drop(['Prior_Emotion'],axis=1)
            classes_test = pd.concat([y_data[test], pd.get_dummies(y_data[test])],axis=1).drop(['Prior_Emotion'],axis=1)

            model = text_cnn_model_baseline(MAX_SEQUENCE_LENGTH, vocab_size, EMBEDDING_DIMS, FILTER_SIZE, CONV_FILTERS, embedding_matrix, DROPOUT, LABELS, 'softmax')
            model.compile(OPTIMIZER, 'categorical_crossentropy', metrics=['accuracy'])
            print('\nINFO: Training...')
            model.fit(x_data[train], classes_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSITY)

            predicted_classes = []
            print('\nINFO: Evaluating fold...')
            predictions = model.predict(x_data[test])
            for i in range(len(predictions)):
                index = np.argmax(predictions[i])
                predicted_classes.append(LABELS[index])

            # Show fold result
            metrics_fold = metrics.metrics(y_data[test], predicted_classes, LABELS, 2)
            metrics_fold.showResults()
            metrics_final.addIntermediateResults(y_data[test], predicted_classes)

            # Free memory
            from keras import backend as K
            K.clear_session()

    print('\nINFO: Final Result:')
    metrics_final.writeResults(EXPERIMENTNAME, SAVEFILE)
    return

def evalTrainAndTestSet(instances_padded, labels, instances_padded_test, labels_test):
    classes_train = pd.concat([labels, pd.get_dummies(labels)],axis=1).drop(['Prior_Emotion'],axis=1)
    classes_test = pd.concat([labels_test, pd.get_dummies(labels_test)],axis=1).drop(['Prior_Emotion'],axis=1)

    model = text_cnn_model_baseline(MAX_SEQUENCE_LENGTH, vocab_size, EMBEDDING_DIMS, FILTER_SIZE, CONV_FILTERS, embedding_matrix, DROPOUT, LABELS, 'softmax')
    model.compile(OPTIMIZER, 'categorical_crossentropy', metrics=['accuracy'])
    print('\nINFO: Training...')
    model.fit(instances_padded, classes_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSITY)

    # model.save('models/epoch_' + str(i) + '.h5')
    # from keras.models import load_model
    # model = load_model('models/epoch_' + str(i+9) + '.h5')
    # print('models/epoch_' + str(i+9) + '.h5')

    predicted_classes = []
    print('\nINFO: Evaluating fold...')
    predictions = model.predict(instances_padded_test)
    for i in range(len(predictions)):
        index = np.argmax(predictions[i])
        predicted_classes.append(LABELS[index])

    # Show fold result
    metrics_fold = metrics.metrics(labels_test, predicted_classes, LABELS, 2)
    metrics_fold.showResults()


if (args.gpu or args.cpu):
    if (args.gpu):
        print('INFO: Forcing experiment to run on GPU (GPU:0)')
        DEVICE = '/device:GPU:0'
    elif (args.cpu):
        print('INFO: Forcing experiment to run on CPU (CPU:0)')
        DEVICE = '/device:CPU:0'
    with tf.device(DEVICE):
        if (args.testset):
            evalTrainAndTestSet(instances_padded, labels, instances_padded_test, labels_test)
        else:
            performCrossValidation(instances_padded, labels)
else:
    # Let tensorflow choose device
    if (args.testset):
        evalTrainAndTestSet(instances_padded, labels, instances_padded_test, labels_test)
    else:
        performCrossValidation(instances_padded, labels)
