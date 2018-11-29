#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class of Auto Encoders.

Author: W.M.Kouw
Date: 28-11-2018
"""
import numpy as np
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape
from keras.models import Model, Sequential, model_from_json
from keras.regularizers import l1, l2
from keras.callbacks import TensorBoard

from keras import backend as K


class autoEncoder(object):
    """
    Superclass of auto-encoders.

    Contains common methods and attributes.
    """

    def __init__(self, ):
        """Set common initializations."""
        self.is_trained = False


class simpleAutoEncoder(autoEncoder):
    """
    Simple implementation of an auto-encoder.

    Contains purely an encoder-decoder part.
    """

    def __init__(self,
                 encoding_dim=2,
                 batch_size=32,
                 num_epochs=8,
                 input_shape=(10, 10),
                 opt='adadelta',
                 loss='binary_crossentropy'):
        """
        Initialize model parameters.

        Parameters
        ----------
        encoding_dim : int
            Dimensionality of the encoding space. (def: 2)
        batch_size : int
            Size of the training batch. (def: 32)
        num_epochs : int
            Number of epochs to train for. (def: 8)
        input_shape : tuple
            Input shape of image. (def: (10, 10))
        opt : str
            Name of the optimizer to use. (def: 'adadelta')
        loss : str
            Name of the loss function to use. (def: 'binary_crossentropy')

        Returns
        -------
        None

        """
        # Check for minimum dim of 1
        if encoding_dim >= 1:
            self.encoding_dim = encoding_dim
        else:
            raise ValueError('Encoding dim needs to be larger than 0.')

        # Check for minimum batch size of 1
        if batch_size >= 1:
            self.batch_size = batch_size
        else:
            raise ValueError('Batch size needs to be larger than 0.')

        # Check for minimum dim of 1
        if num_epochs >= 1:
            self.num_epochs = num_epochs
        else:
            raise ValueError('Number of epochs needs to be larger than 0.')

        self.input_shape = input_shape
        self.opt = opt
        self.loss = loss

        # Compile network
        self.compile()

    def compile(self):
        """
        Compile network architecture.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # Data input shape
        inp = Input(shape=(np.prod(self.input_shape), ))

        # Encoding
        enc = Dense(self.encoding_dim,
                    activation='relu',
                    activity_regularizer=l1(10e-8))(inp)

        # Decoding
        dec = Dense(np.prod(self.input_shape),
                    activation='sigmoid')(enc)

        # Call model
        self.model = Model(inp, dec)

        ''' Encoder model '''

        # Store the encoder
        self.encoder = Model(inp, enc)

        ''' Decoder model '''

        # Placeholder for encoded input
        enc_input = Input(shape=(self.encoding_dim,))

        # Retrieve last layer of model
        dec_layer = self.model.layers[-1]

        # Store the decoder
        self.decoder = Model(enc_input, dec_layer(enc_input))

        ''' Compile model '''

        # Compile model using optimizer and loss function.
        self.model.compile(optimizer=self.opt,
                           loss=self.loss)

    def fit(self, X, Y, val=()):
        """
        Train network on given data.

        Parameters
        ----------
        X : array
            Training data
        Y : array
            Training labels
        val : tuple
            Validation data and labels.

        Returns
        -------
        None

        """
        # Fit model using training parameters
        self.model.fit(X, Y,
                       epochs=self.num_epochs,
                       batch_size=self.batch_size,
                       shuffle=True,
                       validation_data=val,
                       verbose=2)

    def encode(self, Z):
        """
        Propagate image through encoder.

        Parameters
        ----------
        Z : array
            Image that should be encoded.

        Returns
        -------
        H : array
            Embedding of image in encoded space.

        """
        # Reshape image to feature vector
        Z = Z.reshape((-1,))

        # Propagate image through encoder
        return self.encoder.predict(Z)

    def reconstruct(self, Z):
        """
        Propagate image through encoder and decoder.

        Parameters
        ----------
        Z : array
            Image that should be encoded.

        Returns
        -------
        Z_ : array
            Same size image reconstructed through the network.

        """
        # Reshape image to feature vector
        Z = Z.reshape((-1,))

        # Propagate through encoder and decoder
        return self.decoder.predict(Z)


class convolutionalAutoEncoder(autoEncoder):
    """
    Convolutional auto-encoder.

    Encoder and decoder are purely convolutional. The encoding space remains an 
    image (color depends on whether the intermediate kernels have 3 channels).
    """

    def __init__(self,
                 encoding_dim=2,
                 input_shape=(10, 10),
                 kernel_shape=(3, 3),
                 pool_shape=(2, 2),
                 batch_size=32,
                 num_epochs=8,
                 padding='same',
                 opt='adadelta',
                 loss='binary_crossentropy'):
        """
        Initialize model parameters.

        Parameters
        ----------
        encoding_dim : int
            Dimensionality of the encoding space. (def: 2)
        batch_size : int
            Size of the training batch. (def: 32)
        num_epochs : int
            Number of epochs to train for. (def: 8)
        input_shape : tuple
            Input shape of image. (def: (10, 10))
        kernel_shape : tuple
            Shape of kernels to use. (def: (3, 3))
        opt : str
            Name of the optimizer to use. (def: 'adadelta')
        loss : str
            Name of the loss function to use. (def: 'binary_crossentropy')

        Returns
        -------
        None

        """
        # Check for minimum dim of 1
        if encoding_dim >= 1:
            self.encoding_dim = encoding_dim
        else:
            raise ValueError('Encoding dim needs to be larger than 0.')

        # Check for minimum batch size of 1
        if batch_size >= 1:
            self.batch_size = batch_size
        else:
            raise ValueError('Batch size needs to be larger than 0.')

        # Check for minimum dim of 1
        if num_epochs >= 1:
            self.num_epochs = num_epochs
        else:
            raise ValueError('Number of epochs needs to be larger than 0.')

        self.input_shape = input_shape
        self.kernel_shape = kernel_shape
        self.pool_shape = pool_shape
        self.padding = padding
        self.opt = opt
        self.loss = loss

        # Compile network
        self.compile()

    def compile(self):
        """
        Compile network architecture.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # Data input shape
        inp = Input(shape=(*self.input_shape, ))

        # Encoding
        encoder = Sequential()
        encoder.add(Conv2D(16,
                    self.kernel_shape,
                    input_shape=self.input_shape,
                    activation='relu',
                    padding=self.padding))
        encoder.add(MaxPooling2D(self.pool_shape, padding=self.padding))
        encoder.add(Conv2D(3,
                    self.kernel_shape,
                    activation='relu',
                    padding=self.padding))
        encoder.add(MaxPooling2D(self.pool_shape, padding=self.padding))

        # Decoder
        decoder = Sequential()
        decoder.add(Conv2D(3,
                    self.kernel_shape,
                    input_shape=encoder.layers[-1].output_shape[1:],
                    activation='relu',
                    padding=self.padding))
        decoder.add(UpSampling2D(self.pool_shape))
        decoder.add(Conv2D(16,
                    self.kernel_shape,
                    activation='relu',
                    padding=self.padding))
        decoder.add(UpSampling2D(self.pool_shape))
        decoder.add(Conv2D(3,
                    self.kernel_shape,
                    padding=self.padding))

        # Connect sequential models
        enc = encoder(inp)
        dec = decoder(enc)
        self.model = Model(inputs=inp, outputs=dec)

        ''' Encoder model '''

        # Store the encoder separately
        self.encoder = Model(inp, encoder(inp))

        ''' Decoder model '''

        # Intermediate output layer
        enc_inp = Input(shape=self.encoder.layers[-1].output_shape[1:])

        # Store the decoder
        self.decoder = Model(enc_inp, decoder(enc_inp))

        ''' Compile model '''

        # Compile model using optimizer and loss function.
        self.model.compile(optimizer=self.opt, loss=self.loss)

        # Print summary
        self.model.summary()

    def train(self, X, val=(), callback=False):
        """
        Train network on given data.

        Parameters
        ----------
        X : array
            Training data
        Y : array
            Training labels
        val : tuple
            Validation data and labels.
        callback : bool
            Whether to call a Tensorboard instance.

        Returns
        -------
        None

        """
        if callback:

            # Fit model using training parameters
            self.model.fit(X, X,
                           epochs=self.num_epochs,
                           batch_size=self.batch_size,
                           validation_data=val,
                           callbacks=[TensorBoard(log_dir='logs')],
                           shuffle=True,
                           verbose=2)

        else:
            # Fit model using training parameters
            self.model.fit(X, X,
                           epochs=self.num_epochs,
                           batch_size=self.batch_size,
                           validation_data=val,
                           shuffle=True,
                           verbose=2)

    def encode(self, Z):
        """
        Propagate image through encoder.

        Parameters
        ----------
        Z : array
            Image that should be encoded.

        Returns
        -------
        H : array
            Embedding of image in encoded space.

        """
        # Propagate image through encoder
        return self.encoder.predict(Z)

    def decode(self, H):
        """
        Propagate embedding through decoder.

        Parameters
        ----------
        H : array
            Embedding of an image that needs to be decoded.

        Returns
        -------
        Z : array
            Decoded image.

        """
        # Propagate image through encoder
        return self.decoder.predict(H)

    def reconstruct(self, Z):
        """
        Propagate image through encoder and decoder.

        Parameters
        ----------
        Z : array
            Image that should be encoded.

        Returns
        -------
        Z_ : array
            Same size image reconstructed through the network.

        """
        # Reshape image to feature vector
        Z = Z.reshape((-1,))

        # Propagate through encoder and decoder
        return self.model.predict(Z)

    def save(self, filename):
        """
        Write model to json file.

        Parameters
        ----------
        filename : str
            Filename to write model to.

        Returns
        -------
        None

        """
        # TODO: if file extension given, strip extension

        # Write model file
        with open(filename + '_model.json', "w") as json_file:
            json_file.write(self.model.to_json())

        # serialize weights to HDF5
        self.model.save_weights(filename + '_weights.h5')

        # Report
        print("Saved model to disk.")

    def load(self, filename):
        """
        Load model from file.

        Parameters
        ----------
        filename : str
            Filename of model.

        Returns
        -------
        None

        """
        # Load json model config
        with open(filename + '_model.json', 'r') as file:
            model_file = file.read()
        self_model = model_from_json(model_file)

        # Load weights
        self_model.load_weights(filename + '_weights.h5')

        # Report
        print("Loaded model from disk.")


class contractiveAutoEncoder(autoEncoder):
    """
    Contractive auto-encoder.

    Network has convolutional step followed by a dense low-dimensional 
    projection. The weights of this dense part should be invertible for the 
    decoder.
    """

    def __init__(self,
                 encoding_dim=2,
                 input_shape=(10, 10),
                 kernel_shape=(3, 3),
                 pool_shape=(2, 2),
                 batch_size=32,
                 num_epochs=8,
                 padding='same',
                 opt='adadelta',
                 loss='binary_crossentropy'):
        """
        Initialize model parameters.

        Parameters
        ----------
        encoding_dim : int
            Dimensionality of the encoding space. (def: 2)
        batch_size : int
            Size of the training batch. (def: 32)
        num_epochs : int
            Number of epochs to train for. (def: 8)
        input_shape : tuple
            Input shape of image. (def: (10, 10))
        kernel_shape : tuple
            Shape of kernels to use. (def: (3, 3))
        opt : str
            Name of the optimizer to use. (def: 'adadelta')
        loss : str
            Name of the loss function to use. (def: 'binary_crossentropy')

        Returns
        -------
        None

        """
        # Check for minimum dim of 1
        if encoding_dim >= 1:
            self.encoding_dim = encoding_dim
        else:
            raise ValueError('Encoding dim needs to be larger than 0.')

        # Check for minimum batch size of 1
        if batch_size >= 1:
            self.batch_size = batch_size
        else:
            raise ValueError('Batch size needs to be larger than 0.')

        # Check for minimum dim of 1
        if num_epochs >= 1:
            self.num_epochs = num_epochs
        else:
            raise ValueError('Number of epochs needs to be larger than 0.')

        self.input_shape = input_shape
        self.kernel_shape = kernel_shape
        self.pool_shape = pool_shape
        self.padding = padding
        self.opt = opt
        self.loss = loss

        # Compile network
        self.compile()

    def compile(self):
        """
        Compile network architecture.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # Data input shape
        inp = Input(shape=(*self.input_shape, ))

        # Encoding
        encoder = Sequential()
        encoder.add(Conv2D(16,
                    self.kernel_shape,
                    input_shape=self.input_shape,
                    activation='relu',
                    padding=self.padding))
        encoder.add(MaxPooling2D(self.pool_shape, padding=self.padding))
        encoder.add(Conv2D(16,
                    self.kernel_shape,
                    activation='relu',
                    padding=self.padding))
        encoder.add(MaxPooling2D(self.pool_shape, padding=self.padding))
        encoder.add(Dense(self.encoding_dim))

        # Decoder
        decoder = Sequential()
        decoder.add(Dense(np.prod(encoder.layers[-2].output_shape[1:])))
        decoder.add(Reshape(encoder.layers[-2].output_shape[1:]))
        decoder.add(Conv2D(16,
                    self.kernel_shape,
                    activation='relu',
                    padding=self.padding))
        decoder.add(UpSampling2D(self.pool_shape))
        decoder.add(Conv2D(16,
                    self.kernel_shape,
                    activation='relu',
                    padding=self.padding))
        decoder.add(UpSampling2D(self.pool_shape))
        decoder.add(Conv2D(3,
                    self.kernel_shape,
                    padding=self.padding))

        # Connect sequential models
        enc = encoder(inp)
        dec = decoder(enc)
        self.model = Model(inputs=inp, outputs=dec)

        ''' Encoder model '''

        # Store the encoder separately
        self.encoder = Model(inp, encoder(inp))

        ''' Decoder model '''

        # Intermediate output layer
        enc_inp = Input(shape=self.encoder.layers[-1].output_shape[1:])

        # Store the decoder
        self.decoder = Model(enc_inp, decoder(enc_inp))

        ''' Compile model '''

        # Compile model using optimizer and loss function.
        self.model.compile(optimizer=self.opt, loss=self.loss)

        # Print summary
        self.model.summary()

    def train(self, X, val=(), callback=False):
        """
        Train network on given data.

        Parameters
        ----------
        X : array
            Training data
        Y : array
            Training labels
        val : tuple
            Validation data and labels.
        callback : bool
            Whether to call a Tensorboard instance.

        Returns
        -------
        None

        """
        if callback:

            # Fit model using training parameters
            self.model.fit(X, X,
                           epochs=self.num_epochs,
                           batch_size=self.batch_size,
                           validation_data=val,
                           callbacks=[TensorBoard(log_dir='logs')],
                           shuffle=True,
                           verbose=2)

        else:
            # Fit model using training parameters
            self.model.fit(X, X,
                           epochs=self.num_epochs,
                           batch_size=self.batch_size,
                           validation_data=val,
                           shuffle=True,
                           verbose=2)

    def encode(self, Z):
        """
        Propagate image through encoder.

        Parameters
        ----------
        Z : array
            Image that should be encoded.

        Returns
        -------
        H : array
            Embedding of image in encoded space.

        """
        # Propagate image through encoder
        return self.encoder.predict(Z)

    def decode(self, H):
        """
        Propagate embedding through decoder.

        Parameters
        ----------
        H : array
            Embedding of an image that needs to be decoded.

        Returns
        -------
        Z : array
            Decoded image.

        """
        # Propagate image through encoder
        return self.decoder.predict(H)

    def reconstruct(self, Z):
        """
        Propagate image through encoder and decoder.

        Parameters
        ----------
        Z : array
            Image that should be encoded.

        Returns
        -------
        Z_ : array
            Same size image reconstructed through the network.

        """
        # Reshape image to feature vector
        Z = Z.reshape((-1,))

        # Propagate through encoder and decoder
        return self.model.predict(Z)

    def save(self, filename):
        """
        Write model to json file.

        Parameters
        ----------
        filename : str
            Filename to write model to.

        Returns
        -------
        None

        """
        # TODO: if file extension given, strip extension

        # Write model file
        with open(filename + '_model.json', "w") as json_file:
            json_file.write(self.model.to_json())

        # serialize weights to HDF5
        self.model.save_weights(filename + '_weights.h5')

        # Report
        print("Saved model to disk.")

    def load(self, filename):
        """
        Load model from file.

        Parameters
        ----------
        filename : str
            Filename of model.

        Returns
        -------
        None

        """
        # Load json model config
        with open(filename + '_model.json', 'r') as file:
            model_file = file.read()
        self_model = model_from_json(model_file)

        # Load weights
        self_model.load_weights(filename + '_weights.h5')

        # Report
        print("Loaded model from disk.")