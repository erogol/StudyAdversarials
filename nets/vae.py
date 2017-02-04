'''This script demonstrates how to build a variational autoencoder with Keras.
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras import optimizers
from keras.datasets import mnist

class VAE(object):
    def __init__(self,batch_size=100, original_dim=784, latent_dim=2,
                 intermediate_dim=256, nb_epoch=50, epsilon_std=1.0,
                 learning_rate=0.01):
        # set arguments
        self.batch_size = batch_size
        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.nb_epoch = nb_epoch
        self.epsilon_std = epsilon_std
        self.learning_rate = learning_rate

        # set layers
        x = Input(batch_shape=(self.batch_size, self.original_dim))
        h = Dense(self.intermediate_dim, activation='relu')(x)
        self.z_mean = Dense(self.latent_dim)(h)
        self.z_log_var = Dense(self.latent_dim)(h)

        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(self.sampling, output_shape=(self.latent_dim,))([self.z_mean, self.z_log_var])

        # we instantiate these layers separately so as to reuse them later
        decoder_h = Dense(self.intermediate_dim, activation='relu')
        decoder_mean = Dense(self.original_dim, activation='sigmoid')
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)

        # create optimizer
        # optimizer = optimizers.Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
        optimizer = optimizers.RMSprop(lr=learning_rate)

        # create whole VAE
        self.vae = Model(x, x_decoded_mean)
        self.vae.compile(optimizer=optimizer, loss=self.vae_loss)

        # create model for projection
        self.encoder = Model(x, self.z_mean)

        # build a digit generator that can sample from the learned distribution
        decoder_input = Input(shape=(latent_dim,))
        _h_decoded = decoder_h(decoder_input)
        _x_decoded_mean = decoder_mean(_h_decoded)
        self.generator = Model(decoder_input, _x_decoded_mean)
        self.feat_extractor = Model(x, h_decoded)

    def train(self, x_train, y_train, x_test=None, y_test=None, validation_split=None):
        if x_test is not None and y_test is not None:
            self.vae.fit(x_train, y_train,
                        shuffle=True,
                        nb_epoch=self.nb_epoch,
                        batch_size=self.batch_size,
                        validation_data=(x_test, x_test))
        elif validation_split is not None:
            self.vae.fit(x_train, y_train,
                        shuffle=True,
                        nb_epoch=self.nb_epoch,
                        batch_size=self.batch_size,
                        validation_split=validation_split)
        else:
            raise AttributeError("validation data is not provided!!")

    def encode(self, x):
        return self.encoder.predict(x, batch_size = self.batch_size)

    def generate(self, params):
        return self.generator.predict(params)

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(self.batch_size, self.latent_dim), mean=0.,
                                  std=self.epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def vae_loss(self, x, x_decoded_mean):
        xent_loss = self.original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        return xent_loss + kl_loss


# # display a 2D plot of the digit classes in the latent space
# x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
# plt.figure(figsize=(6, 6))
# plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
# plt.colorbar()
# plt.show()
#
#
# # display a 2D manifold of the digits
# n = 15  # figure with 15x15 digits
# digit_size = 28
# figure = np.zeros((digit_size * n, digit_size * n))
# # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# # to produce values of the latent variables z, since the prior of the latent space is Gaussian
# grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
# grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
#
# for i, yi in enumerate(grid_x):
#     for j, xi in enumerate(grid_y):
#         z_sample = np.array([[xi, yi]])
#         x_decoded = generator.predict(z_sample)
#         digit = x_decoded[0].reshape(digit_size, digit_size)
#         figure[i * digit_size: (i + 1) * digit_size,
#                j * digit_size: (j + 1) * digit_size] = digit
#
# plt.figure(figsize=(10, 10))
# plt.imshow(figure, cmap='Greys_r')
# plt.show()
