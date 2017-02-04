'''
1. train a simple model on normal MNIST (~0.98 on normal test set)
2. create adversarials from that model
3. save adversarials to data/adversarials.npy
4. compute adversarial accuracy (~0.11)
5. plot sample 100 adversarials
6. train with adversarials
7. compute adversarial accuracy (~0.96 on normal testset and ~0.78 on adversarial testset)
8. embedding to 2D with tsne
9. plotting normal instances
10. plotting adversarials
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from tsne import bh_sne
from keras import backend
import numpy as np
from matplotlib import pylab as plt
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

import keras             
import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist, model_mnist
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from cleverhans.attacks import fgsm, jsma
from cleverhans.attacks_tf import jacobian_graph
from cleverhans.utils import other_classes


FLAGS = flags.FLAGS
FLAGS.nb_epochs = 6
FLAGS.train_dir = '/tmp'
FLAGS.filename = 'mnist.ckpt'
FLAGS.batch_size = 128
FLAGS.learning_rate = 0.1

FLAGS.nb_classes = 10
FLAGS.source_samples = 5
FLAGS.img_rows = 28
FLAGS.img_cols = 28

FLAGS.max_steps = 10000
FLAGS.log_dir = "./logs/"

##############################################################################
# Training a simple model on MNIST
##############################################################################
def evaluate():
    # Evaluate the accuracy of the MNIST model on legitimate test examples
    accuracy = model_eval(sess, x, y, predictions, X_test, Y_test)
    assert X_test.shape[0] == 10000, X_test.shape
    print('Test accuracy on legitimate test examples: ' + str(accuracy))
    return accuracy

# Set TF random seed to improve reproducibility
tf.set_random_seed(1234)

# Set tensorboard
keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)

# Image dimensions ordering should follow the Theano convention
if keras.backend.image_dim_ordering() != 'th':
    keras.backend.set_image_dim_ordering('th')
    print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to 'th', temporarily setting to 'tf'")

# Create TF session and set as Keras backend session
sess = tf.Session()
keras.backend.set_session(sess)

# Get MNIST test data
X_train, Y_train, X_test, Y_test = data_mnist()
assert Y_train.shape[1] == 10.
label_smooth = .1
Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, 1, 28, 28))
y = tf.placeholder(tf.float32, shape=(None, 10))

# Define TF model graph
model = model_mnist()
predictions = model(x)
print("Defined TensorFlow model graph.")

# Train an MNIST model
model_train(sess, x, y, predictions, X_train, Y_train, evaluate=evaluate)

##############################################################################
# Create Adversarials
##############################################################################
# Craft adversarial examples using Fast Gradient Sign Method (FGSM)
adv_x = fgsm(x, predictions, eps=0.2)
X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test])
assert X_test_adv.shape[0] == 10000, X_test_adv.shape

# Evaluate the accuracy of the MNIST model on adversarial examples
accuracy = model_eval(sess, x, y, predictions, X_test_adv, Y_test)
print('Test accuracy on adversarial examples: ' + str(accuracy))

# save instances
np.save('data/x_train.npy', X_train)
np.save('data/y_train.npy', Y_train)
np.save('data/x_test.npy', X_test)
np.save('data/y_test.npy', Y_test)
np.save('data/adversarials.npy', X_test_adv)

# load instances
X_train = np.load('data/x_train.npy')
Y_train = np.load('data/y_train.npy')
X_test = np.load('data/x_test.npy')
Y_test = np.load('data/y_test.npy')
X_test_adv = np.load('data/adversarials.npy')

##############################################################################
# Plot example adversarial instances
##############################################################################
# visualize
# sample a set
idxs = np.random.choice(range(10000), 100)
X_test_adv_sample = X_test_adv[idxs]
Y_test_sample = Y_test[idxs]

# plot the sample set
plt.figure()
for c,img in enumerate(X_test_adv_sample):
    plt.subplot(10,10,c+1)
    plt.axis("off")
    plt.title(np.argmax(Y_test_sample[c]))
    plt.imshow(img.squeeze(), cmap=plt.cm.gray)
plt.show()

##############################################################################
# Embedding with TSNE
##############################################################################
# plotting function
def plot_mnist(X, y, X_embedded, name, min_dist=10.0):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(frameon=False)
    plt.title("\\textbf{MNIST dataset} -- Two-dimensional "
          "embedding of 70,000 handwritten digits with %s" % name)
    plt.setp(ax, xticks=(), yticks=())
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9,
                    wspace=0.0, hspace=0.0)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
            c=y, marker="x")

    if min_dist is not None:
        from matplotlib import offsetbox
        shown_images = np.array([[15., 15.]])
        indices = np.arange(X_embedded.shape[0])
        np.random.shuffle(indices)
        for i in indices[:5000]:
            dist = np.sum((X_embedded[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist:
                continue
            shown_images = np.r_[shown_images, [X_embedded[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(X[i].reshape(28, 28),
                                      cmap=plt.cm.gray_r), X_embedded[i])
            ax.add_artist(imagebox)


from keras.models import Model

# create feature extractor
model_feat = Model(input=model.input, output=model.get_layer('flatten_2').output)

# extract last layer features
X_test_feats = model_feat.predict(X_test)
X_test_adv_feats   = model_feat.predict(X_test_adv)
assert(X_test_feats.shape == X_test_adv_feats.shape)

# concate normal and adversarial features
X_feats = np.concatenate([X_test_feats, X_test_adv_feats], axis=0)
Y_feats = np.concatenate([Y_test, Y_test]).argmax(1)
images = np.concatenate([X_test,X_test_adv], axis=0)

# train tsne projection
vis_data = bh_sne(X_feats.astype('float64'))

# plot normal instances
plot_mnist(images[:10000], Y_feats[0:10000], vis_data[:10000], 't-sne', min_dist=20.0)

# plot adversarial instances
plot_mnist(images[10000:], Y_feats[10000:], vis_data[10000:], 't-sne', min_dist=20.0)

# plot all
plot_mnist(images, Y_feats, vis_data, 't-sne', min_dist=20.0)

##############################################################################
# Train model with adversatial instances
##############################################################################

def evaluate_2():
    # Evaluate the accuracy of the adversarialy trained MNIST model on
    # legitimate test examples
    accuracy = model_eval(sess, x, y, predictions_2, X_test, Y_test)
    print('Test accuracy on legitimate test examples: ' + str(accuracy))

    # Evaluate the accuracy of the adversarially trained MNIST model on
    # adversarial examples
    accuracy_adv = model_eval(sess, x, y, predictions_2_adv, X_test, Y_test)
    print('Test accuracy on adversarial examples: ' + str(accuracy_adv))
    
# Redefine TF model graph
model_2 = model_mnist()
predictions_2 = model_2(x)
adv_x_2 = fgsm(x, predictions_2, eps=0.3)
predictions_2_adv = model_2(adv_x_2)

# Perform adversarial training
model_train(sess, x, y, predictions_2, X_train, Y_train, predictions_adv=predictions_2_adv,
        evaluate=evaluate_2)

# Evaluate the accuracy of the MNIST model on adversarial examples
accuracy = model_eval(sess, x, y, predictions_2, X_test_adv, Y_test)
print('Test accuracy on adversarial examples: ' + str(accuracy))