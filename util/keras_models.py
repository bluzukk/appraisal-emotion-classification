#!/usr/bin/env python
"""
Keras models for different tasks

- Maximum Entropy model
- CNN for predicting emotions from text (TextCNN)
- CNN for predicting appraisals from text (TextCNN)
- Shallow neural network for predicting emotions from appraisals

+ Experimental models
"""

import sys
import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Reshape, Concatenate, Dropout
from keras.layers import GlobalMaxPooling1D, Flatten, Activation
from keras.layers import Conv1D, MaxPooling1D, Embedding, Reshape
from keras.layers import Conv2D, MaxPool2D, Embedding, Reshape
from keras.initializers import Constant
from keras.models import Model, Sequential
from keras import regularizers

"""
MaxEnt model
"""
def MaxEnt(input_shape, output_shape, activation):
    model = Sequential()
    model.add(Dense(output_shape, input_shape=(input_shape ,), activation=activation, activity_regularizer=regularizers.l2(0.1)))
    return model

"""
TextCNN model (Text->Emotion)
"""
def text_cnn_model_baseline(sequence_length, vocab_size, embedding_size,
    filter_sizes, num_filters, embedding_matrix, drop_out, LABELS, activation):

    input = Input(shape=(sequence_length,), name='input_text')

    embedding = Embedding(vocab_size, embedding_size, embeddings_initializer=Constant(embedding_matrix), name='glove300_embedding', trainable=False)(input)
    expend_shape = [embedding.get_shape().as_list()[1], embedding.get_shape().as_list()[2], 1]
    embedding = Reshape(expend_shape)(embedding)

    # conv->max pool
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
    # embedding = Dropout(0.2)(embedding)
        conv = Conv2D(filters=num_filters,
            kernel_size=[filter_size, embedding_size],
            activation='relu',
            name=('conv_filtersize%d' % filter_size))(embedding)
        # conv = Dropout(0.5)(conv)
        max_pool = MaxPool2D(pool_size=[sequence_length - filter_size + 1, 1],
            strides=(2, 2),
            padding='valid',
            name=('max_pool_%d' % filter_size))(conv)
        # max_pool = Dropout(drop_out)(max_pool)
        pooled_outputs.append(max_pool)

    # combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = Concatenate(axis=3)(pooled_outputs)
    # h_pool = Dropout(0.5, name='0.5')(h_pool)
    h_pool_flat = Reshape([num_filters_total])(h_pool)
    fullyconnected = Dense(128, name='fullyconnected2', activation='relu')(h_pool_flat)
    dropout = Dropout(drop_out, name='0.5-')(fullyconnected)
    # output layer
    output = Dense(len(LABELS),
        kernel_initializer='glorot_normal',
        bias_initializer=keras.initializers.constant(0.1),
        activation=activation,
        name='output_emotion')(dropout)

    model = Model(input, output)
    return model

"""
TextCNN model (Text->Appraisal)
"""
def text_cnn_model_appraisals(sequence_length, vocab_size, embedding_size,
    filter_sizes, num_filters, embedding_matrix, drop_out, output_count, activation):

    input = Input(shape=(sequence_length,), name='input_text')
    try:
        embedding = Embedding(vocab_size, embedding_size, embeddings_initializer=Constant(embedding_matrix), name='glove300_embedding', trainable=False)(input)
    except TypeError:
        print('\nERROR: Embedding does not fit dataset:')
        print('       Datasets contains %d terms and embedding contains %d terms...' % (vocab_size, embedding_matrix.shape[0]))
        print('\nExiting.')
        sys.exit(1)
    except:
        print('\nUnexpected error:', sys.exc_info()[1])
        sys.exit(1)

    expend_shape = [embedding.get_shape().as_list()[1], embedding.get_shape().as_list()[2], 1]
    embedding = Reshape(expend_shape)(embedding)

    # conv->max pool
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
    # embedding = Dropout(0.2)(embedding)
        conv = Conv2D(filters=num_filters,
            kernel_size=[filter_size, embedding_size],
            activation='relu',
            name=('conv_filtersize%d' % filter_size))(embedding)
        # conv = Dropout(0.5)(conv)
        max_pool = MaxPool2D(pool_size=[sequence_length - filter_size + 1, 1],
            strides=(2, 2),
            padding='valid',
            name=('max_pool_%d' % filter_size))(conv)
        # max_pool = Dropout(drop_out)(max_pool)
        pooled_outputs.append(max_pool)

    # combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = Concatenate(axis=3)(pooled_outputs)
    # h_pool = Dropout(drop_out, name='0.5')(h_pool)
    h_pool_flat = Reshape([num_filters_total])(h_pool)
    fullyconnected = Dense(128, name='fullyconnected2', activation='relu')(h_pool_flat)
    dropout = Dropout(drop_out, name='d_out')(fullyconnected)
    # output layer
    output = Dense(output_count,
        kernel_initializer='glorot_normal',
        bias_initializer=keras.initializers.constant(0.1),
        activation=activation,
        name='output_emotion')(dropout)

    model = Model(input, output)
    return model

"""
Shallow neural network for predicting emotions from dimensions
"""
def shallowNN_emotions_from_dimensions(input_length, layer_dim, drop_out, LABELS, activation):

    model = Sequential()
    model.add(Dense(layer_dim, input_shape=(input_length ,), activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(layer_dim, activation='relu'))
    model.add(Dropout(drop_out))
    model.add(Dense(len(LABELS),
        kernel_initializer='glorot_normal',
        bias_initializer=keras.initializers.constant(0.1),
        activation=activation,
        name='output_emotion'))
    return model

"""
CNN model experimental
"""
def cnn_model(sequence_length, vocab_size, embedding_size,
    filter_sizes, num_filters, embedding_matrix, drop_out, LABELS, activation):
    # print('CNN model:')

    embedding_layer = Embedding(
        vocab_size,
        300,
        weights=[embedding_matrix],
        input_length=sequence_length,
        trainable=False)

    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(num_filters, 2, activation='relu'))
    model.add(MaxPooling1D(1))
    model.add(Dropout(drop_out))
    model.add(Conv1D(num_filters, 3, activation='relu'))
    model.add(MaxPooling1D(1))
    model.add(Dropout(drop_out))
    model.add(Conv1D(num_filters, 4, activation='relu'))
    model.add(MaxPooling1D(1))
    model.add(Dropout(drop_out))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(len(LABELS),
        kernel_initializer='glorot_normal',
        bias_initializer=keras.initializers.constant(0.1),
        activation=activation,
        name='output'))
    return model

"""
TextCNN model (Ensemble Selection)
"""
def text_cnn_model_selector(sequence_length, vocab_size, embedding_size,
    filter_sizes, num_filters, embedding_matrix, drop_out, LABELS, activation):

    input = Input(shape=(sequence_length,), name='input_text')

    embedding = Embedding(vocab_size, embedding_size, embeddings_initializer=Constant(embedding_matrix), name='glove300_embedding', trainable=False)(input)
    expend_shape = [embedding.get_shape().as_list()[1], embedding.get_shape().as_list()[2], 1]
    embedding = Reshape(expend_shape)(embedding)

    # conv->max pool
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
    # embedding = Dropout(0.2)(embedding)
        conv = Conv2D(filters=num_filters,
            kernel_size=[filter_size, embedding_size],
            activation='relu',
            name=('conv_filtersize%d' % filter_size))(embedding)
        # conv = Dropout(0.5)(conv)
        max_pool = MaxPool2D(pool_size=[sequence_length - filter_size + 1, 1],
            strides=(2, 2),
            padding='valid',
            name=('max_pool_%d' % filter_size))(conv)
        max_pool = Dropout(0.25)(max_pool)
        pooled_outputs.append(max_pool)

    # combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = Concatenate(axis=3)(pooled_outputs)
    # h_pool = Dropout(0.5, name='0.5')(h_pool)
    h_pool_flat = Reshape([num_filters_total])(h_pool)
    # fullyconnected = Dense(128, name='fullyconnected2', activation='relu')(h_pool_flat)
    dropout = Dropout(0.5, name='0.5-')(h_pool_flat)
    # output layer
    output = Dense(2,
        kernel_initializer='glorot_normal',
        bias_initializer=keras.initializers.constant(0.1),
        activation=activation,
        name='output_emotion')(dropout)

    model = Model(input, output)
    return model


def multi_task_CNN_only_weights():
    # Text input
    input_for_dimensions = Input(shape=(MAX_SEQUENCE_LENGTH,), name='input_dim')
    input_for_emotions   = Input(shape=(MAX_SEQUENCE_LENGTH,), name='input_demo')

    embedding1 = Embedding(vocab_size, embedding_size, embeddings_initializer=Constant(embedding_matrix), name='embedding1', trainable=False)(input_for_dimensions)
    # embedding2 = Embedding(vocab_size, embedding_size, embeddings_initializer=Constant(embedding_matrix), name='embedding2', trainable=False)(input_for_emotions)

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
                    name=('conv_%d' % filter_size))(embedding_chars1)
        # encoded_a = conv(embedding_chars1)
        # encoded_b = conv(embedding_chars1)
        # merged_conv = keras.layers.concatenate([encoded_b, encoded_a], axis=-1)
        max_pool = MaxPool2D(pool_size=[sequence_length - filter_size + 1, 1],
                       strides=(2, 2),
                       padding='valid',
                       name=('max_pool_%d' % filter_size))(conv)
        # max_pool = Dropout (0.5)(max_pool)
        pooled_outputs.append(max_pool)

    num_filters_total = num_filters * len(filter_sizes)
    h_pool = Concatenate(axis=3)(pooled_outputs)
    merged = Reshape([num_filters_total])(h_pool)
    merged1 = Dropout (0.5)(merged)

    emo_output = Dense(len(LABELS), activation='softmax', name='emo')(merged1)
    seq_output = Dense(len(DIMENSIONS), activation='sigmoid', name='dim')(merged1)

    multitask_model = Model(inputs=[input_for_emotions, input_for_dimensions], outputs=[emo_output, seq_output])
    multitask_model.compile(loss='categorical_crossentropy',
                  optimizer=OPTIMIZER, metrics=['accuracy'], loss_weights=[1,0.25])

    return multitask_model
