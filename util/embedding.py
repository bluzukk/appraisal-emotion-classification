#!/usr/bin/env python
"""
Helper for preparing, saving and loading word embeddings

"""

import numpy as np
import gensim
from numpy import asarray, zeros
import sys

def prepareEmbeddings(word_index, MAX_NUM_WORDS, EMBEDDING_PATH, EMBEDDING_DIMS):
    # Load the pre trained embedding into memory
    embeddings_index = dict()
    try:
        f = open(EMBEDDING_PATH)
        print('\nINFO: Found embedding %s' % EMBEDDING_PATH)
        print('INFO: Preparing embedding for your dataset... (might take a few minutes)')
    except FileNotFoundError:
        print('\nERROR: Path "%s" does not contain a valid embedding' % EMBEDDING_PATH)
        print('       Make sure you enter the correct path with the --embeddingpath option.')
        print('\nExiting.')
        sys.exit(1)
    except:
        print("Unexpected error:", sys.exc_info()[1])
        sys.exit(1)

    for line in f:
        values = line.split()
        cnt = 0
        # 'handle' words with spaces, special letters or meta information
        try:
            word = values[0]
            coefs = asarray(values[1:], dtype='float32')
        except ValueError:
            try:
                word = values[0]+values[1]
                coefs = asarray(values[2:], dtype='float32')
            except ValueError:
                try:
                    word = values[0]+values[1]+values[2]
                    coefs = asarray(values[3:], dtype='float32')
                except ValueError:
                    pass
        # word = values[0]
        # coefs = asarray(values[1:], dtype='float32')
        # print(coefs)
        embeddings_index[word] = coefs
    f.close()

    print('INFO: Loaded %s word vectors.' % len(embeddings_index))
    # create weight matrix for words in training docs
    # prepare embedding matrix
    num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
    # First entry must be all-zeros for the padding values
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIMS))
    found = 0
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            found += 1
    # print(embedding_matrix)
    print('INFO: Words found in embedding: ' + str(found) + '/' + str(num_words-1))
    return embedding_matrix

def saveEmbedding(embedding, file):
    np.save(file, embedding, allow_pickle=True)
    print('\nINFO: Successfully saved embedding saved as ' + file)

def loadEmbedding(file):
    embedding = np.load(file, allow_pickle=True)
    print('\nINFO: Successfully loaded embedding ' + file)
    return embedding
