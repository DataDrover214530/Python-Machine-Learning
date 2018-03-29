# Based on the cnn_mnist example demonstrated in the lab, plus some elements from 
# the tutorial at;
# https://www.tensorflow.org/get_started/mnist/pros 
# and also a worked example whos code can be found at;
# https://github.com/jtopor/CUNY-MSDA-661/blob/master/LFW-CNN/TF-Layers-LFW-Github.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#Changing the code to use LFW
from sklearn.datasets import fetch_lfw_people
from sklearn.cross_validation import train_test_split

import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):

  
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  # but LFW images (with resize and slicing options used here) are 64 x 64 

  input_layer = tf.convert_to_tensor(features)
    # There are stubs throughout the code, I used them to track down
  # some errors in the shaping etc of data when I first built the code
  print("+++ I reshaped the input layer  +++")
  
  
  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # though in the second version tanh activation is used as a comparison
  # Padding is added to preserve width and height.

    # Input Tensor Shape: [batch_size, 64, 64, 3]
    # Output Tensor Shape: [batch_size, 64, 64, 32]

  print("+++ doing conv 1 +++")
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      #activation=tf.nn.tanh)
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  print("+++ doing pool 1 +++")
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 32 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 64, 64, 3]
  # Output Tensor Shape: [batch_size, 64, 64, 32]
  
  print("+++ doing conv 2 +++")
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      #activation=tf.nn.tanh)
      activation=tf.nn.relu)
      # Again tanh is used in the second version

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2

  print("+++ doing pool 2 +++")
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)


  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 16, 16, 64]
  # Output Tensor Shape: [batch_size, 4096]
  pool2_flat = tf.reshape(pool2, [-1, 16 * 16 * 32])
  print("+++ reshaping pool 2 +++")


  # Dense Layer
  # Densely connected layer with 2048 neurons
  # Input Tensor Shape: [batch_size, 8192]
  # Output Tensor Shape: [batch_size, 2048]
  dense = tf.layers.dense(inputs=pool2_flat, units=2048, activation=tf.nn.relu)
  #dense = tf.layers.dense(inputs=pool2_flat, units=2048, activation=tf.nn.tanh)
  # Tanh in second version
  
  # Add dropout operation; 0.5 probability that element will be kept
  # First version was run with 0.4 dropout rate
  print("+++ doing the drop0ut layer +++")
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.5, training=mode == learn.ModeKeys.TRAIN)
 
  # Logits layer
  # Input Tensor Shape: [batch_size, 2048]
  # Output Tensor Shape: [batch_size, n_classes]
  print("+++ doing the logits  +++")
  logits = tf.layers.dense(inputs=dropout, units= n_classes)

  loss = None
  train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
  # depth=n_classes, as oppossed to fixed size of 10 for MNIST digits
  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=n_classes)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  # As per report attempts to improve the model by changing to a different
  # optimiser such as Adam caused logging problems and so were abandonded

  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="SGD")
  print("+++  calc'd loss, done train op  +++")
  # Generate Predictions

  predictions = {
      "classes": tf.argmax(
          input=logits, axis=1),
      "probabilities": tf.nn.softmax(
          logits, name="softmax_tensor")
  }
  print("+++  done predictions  +++")
  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)
 

def main(unused_argv):

  print("+++  running main +++")
  
  # define global variable for number of classes that we will fill
  # as per the number of people refurned from the dataset
  global n_classes
  
  # Slices out images of 64x64 from the dataset. Returns images of 34 different people
  lfw_people = fetch_lfw_people(min_faces_per_person=30, 
                                slice_ = (slice(61,189),slice(61,189)),
                                resize=0.5, color = True)
  X = lfw_people.images
  y = lfw_people.target
  
  # get count of number of possible labels - need to use this as
  # number of units for dense layer in call to tf.layers.dense and
  # for defining the one-hot matrix. Here the number of possible
  # labels is 34 based on the subset of LFW that we selected above. 
  target_names = lfw_people.target_names
  n_classes = target_names.shape[0]
  y = np.asarray(y, dtype=np.int32)
  
  # split into a training and testing set
  train_data, eval_data, train_labels, eval_labels = train_test_split(
X, y, test_size=0.25, random_state=42)
 
  print("+++ I split the data  +++")
  # Create the Estimator - changed here to relfect use of LFW not MNIST
  lfw_classifier = learn.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/lfw_CNN_model")

  print("+++ I made the estimator  +++")
  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)
  
  print("+++ Train the model +++")
  lfw_classifier.fit(
      x=train_data,
      y=train_labels,
      batch_size=64,
      steps=1000,
      monitors=[logging_hook])
  print("+++ Doing the metrics  +++")
  # Configure the accuracy metric for evaluation
  metrics = {
      "accuracy":
          learn.MetricSpec(
              metric_fn=tf.metrics.accuracy, prediction_key="classes"),
  }
      
  print("+++ printing the evaluation  +++")
  # Evaluate the model and print results
  eval_results = lfw_classifier.evaluate(
      x=eval_data, y=eval_labels, metrics=metrics)
  print("+++ eval results are")
  print(eval_results)
  print("++++++++++++++++++")

if __name__ == "__main__":
  tf.app.run()
