'''
Notes:
    VAE does not have the abiliy to go beyond the given dataset distribution and
    it basically ignores low probability regions which accomodate adversarial
    instance. This can be seen by the final embedding visualization.
    
    We need a better way to augment data as Denoising autoencoders. It might 
    capture more of the low probability regions by contracting the noised instances
    toward the monifold learned by the model itself.
    
    I use the latent mean dimensions as the feature but maybe it is better to
    use intermedeate decoder layer. However, it requires to prevent sampling
    of latent layer rather deterministic propagation. 
    
    One intriguing observation of VAE, as we see on the 2D embedding, it somehow unsupervisedly
    discriminates adversarials (top_right half of the figure) from normal instances
    (left-bottom half of the figure). It also disperses different classes 
    of the adversarials instances as well, to some degree (Better than the supervised model). By the first observation, 
    VAE still might be useful for the detection of adversarials when it comes to 
    secure proposed ML system against adversary attacks. By the second observation, 
    maybe the reason for the failure of this method pertains to limited generalization 
    of classifier layer.
'''

from nets.vae import VAE
from utils import load_data, plot_mnist
from scipy.stats import norm

# load pre-saved data
x_train, y_train, x_test, y_test, x_advs = load_data()

#############################################################################
# Train MNIST on VAE features (TODO: need parameter tunning)
#############################################################################

# create VAE model for feature learning
vae = VAE(batch_size=200, original_dim=784, latent_dim=128, intermediate_dim=512, 
          nb_epoch=50, epsilon_std=1.0, learning_rate=0.001)

vae.train(x_train.reshape([60000,784]), x_train.reshape([60000,784]), 
          x_test.reshape([10000,784]), x_test.reshape([10000,784]))

# compute features
x_train_feats = vae.encode(x_train.reshape([60000,784]))
x_test_feats = vae.encode(x_test.reshape([10000,784]))
# x_train_feats = vae.feat_extractor.predict(x_train, batch_size=200)
# x_test_feats  = vae.feat_extractor.predict(x_test, batch_size=200)
print(x_train_feats.shape)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.optimizers import SGD, RMSprop
from keras.regularizers import l2, activity_l2

# create classifier
clf = Sequential()
clf.add(Dense(10, input_shape=(x_train_feats.shape[1],)))
clf.add(Activation('softmax'))

# optimizer = SGD(lr=0.01, momentum=0, decay=0.0, nesterov=False)
optimizer = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
clf.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
clf.summary()

# train clf
clf.fit(x_train_feats, y_train, batch_size=128, nb_epoch=10, 
        verbose=1, validation_data=[x_test_feats, y_test], 
        shuffle=True)

# prediction on adversarial test instances
x_advs_feats = vae.encoder.predict(x_advs.reshape(10000, 784), batch_size=200)
preds = clf.predict_classes(x_advs_feats, batch_size=200)

from sklearn.metrics import accuracy_score
print()
print(accuracy_score(y_test.argmax(1), preds))

#############################################################################
# TSNE embedding
#############################################################################
from tsne import bh_sne

# train tsne projection
feats = np.concatenate([x_test_feats, x_advs_feats], axis=0)
labels = np.concatenate([y_test.argmax(1), y_test.argmax(1)], axis=0)
images = np.concatenate([x_test, x_advs])
vis_data = bh_sne(feats.astype('float64'))

# plot adversarial instances
plot_mnist(x_test, y_test.argmax(1), vis_data[10000:], 't-sne', min_dist=20.0)

# plot normal instances
plot_mnist(x_advs, y_test.argmax(1), vis_data[:10000], 't-sne', min_dist=30.0)

# plot all instances
plot_mnist(images, labels, vis_data, 't-sne', min_dist=20.0)
