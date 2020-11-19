# Full example for my blog post at:
# https://danijar.com/building-variational-auto-encoders-in-tensorflow/


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tfd = tf.contrib.distributions



#---------------------------------------------------------
#---------------------TEST CODE---------------------------
#---------------------------------------------------------
#CIFAR10 Branch
#This code is indev code for making this VAE work with the CIFAR10 dataset, it may not run properly
#
#Errors: Issues with plotting, see lines 191, 193, 201


#Function for returning dictionary data from Pickle
#Format info here: http://www.cs.toronto.edu/~kriz/cifar.html
import pickle
def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d

#Get from data batch 1
data_set_1 = unpickle('data_batch_1')
#Grab only the image data
all_data = data_set_1[b'data']
#Get labels
all_labels = data_set_1[b'labels']
#Reshape data into usable format
#Format, each pic = 3072 entries
#1024 - R values
#1024 - G values
#1024 - B values
# Total 3x32x32 = 3072
all_pics = all_data.reshape(-1,3,32,32)
#Normalize values (betwee 0 and 1) for tensorflow
all_pics = all_pics /255.0
#Transpose data so we get 32x32 sets of pixels in format [R,G,B]
#Result should be 1000 images in this given format:
#all_pics[IMG Number][Y Location][X Location][RGB value]
all_pics = np.transpose(all_pics, (0,2,3,1))

#Grab only cat pictures:
#Explanation:
#For every i in range(length of dataset)
#If corresponding label == 3 (ID For cats)
#Add data into datset
all_pics = [all_pics[i] for i in range(len(all_pics)) if all_labels[i] == 3]

#TODO: Wrap all the above code into a function/external package

#---------------------------------------------------------






def make_encoder(data, code_size):
  x = tf.layers.flatten(data)
  x = tf.layers.dense(x, 200, tf.nn.relu)
  x = tf.layers.dense(x, 200, tf.nn.relu)
  loc = tf.layers.dense(x, code_size)
  scale = tf.layers.dense(x, code_size, tf.nn.softplus)
  return tfd.MultivariateNormalDiag(loc, scale)


def make_prior(code_size):
  loc = tf.zeros(code_size)
  scale = tf.ones(code_size)
  return tfd.MultivariateNormalDiag(loc, scale)

#Declaring a function called decoder which takes two parameters: code and data_shape
def make_decoder(code, data_shape):
 # initializing variable x which stores code.
  x = code
  #An activation function: tf.nn.relu
  # It is applied to the output of a neural network layer,
  # #which is then passed as the input to the next layer.
  # Activation functions are an essential part of neural networks
  # as they provide non-linearity, without which the neural network
  # reduces to a mere logistic regression model
  # applying `tf.nn.relu` function and add a Dense layer as the first layer.
  x = tf.layers.dense(x, 200, tf.nn.relu)
  x = tf.layers.dense(x, 200, tf.nn.relu)
  #Logits are by definition unnormalized log probabilities
  logit = tf.layers.dense(x, np.prod(data_shape))
  #-1 Place holder for the dimension that will be calculated automatically.
  # this way, we can calculate length accurately
  logit = tf.reshape(logit, [-1] + data_shape)
  #we are returning a tfp independent Bernoulli distribution
  #width and height in our case, belong to the same data point (2)
  # even though they have independent parameters
  return tfd.Independent(tfd.Bernoulli(logit), 3)


def plot_codes(ax, codes, labels):
  ax.scatter(codes[:, 0], codes[:, 1], s=2, c=labels, alpha=0.1)
  ax.set_aspect('equal')
  ax.set_xlim(codes.min() - .1, codes.max() + .1)
  ax.set_ylim(codes.min() - .1, codes.max() + .1)
  ax.tick_params(
      axis='both', which='both', left='off', bottom='off',
      labelleft='off', labelbottom='off')


def plot_samples(ax, samples):
  for index, sample in enumerate(samples):
    ax[index].imshow(sample, cmap=plt.cm.binary)
    ax[index].axis('off')



#Create placeholder data in dimension ?x28x28, has no represented data values
#'None' represents an unknown dimension
data = tf.placeholder(tf.float32, [None, 32, 32, 3])


#Create function templates, this ensures that function specific variables are initialized first and consistent between all calls of this function
make_encoder = tf.make_template('encoder', make_encoder)
make_decoder = tf.make_template('decoder', make_decoder)



#------------------------------Model Definitions---------------------------------
#Returns a Multivariate Normal Diag Distribution with basic parameters set [0,0] and [1,1], prior is fixed with no trainable parameters so it does not need a template.
#See code explanation for 'make_prior'
#Prior is p(z)
prior = make_prior(code_size=2)
#Create a Multivariate Normal Diag Distribution based on our desired encoder
#See code explanation for 'make_encoder'
#Posterior is p(z|x)
posterior = make_encoder(data, code_size=2)
#Grab a sample of the data from our encoder that will be passed back through  our decoder, this is 'z'
code = posterior.sample()


#------------------------------Loss Function---------------------------------
#We need to compute the negative log-likelihood, so we use our decoder to find log(p(x|z))
#Data is used as a template
likelihood = make_decoder(code, [32, 32, 3]).log_prob(data)
#Find the KL divergence of the posterior and prior KL[p(z|x)||p(z)]
divergence = tfd.kl_divergence(posterior, prior)
#elbo is our loss function since it should be [-log(p(x|z)) + KL(p(z|x)||p(z))]
#Here we find the value but negative
elbo = tf.reduce_mean(likelihood - divergence)
#Setup the tensorflow optimizer that will minimize loss, we must do it according to our created loss function
#We must make elbo negative in order to correct signs since our output above is negative
#Note, learning_rate is default '0.001' so this input is pointless
optimize = tf.train.AdamOptimizer(0.001).minimize(-elbo)
#Use the same decoder as before (Since we're using templates) to grab samples of the data. This is purely used for visual output in the code below.
samples = make_decoder(prior.sample(10), [32, 32, 3]).mean()




#Used for grabbing images. Dirty function, should be changed
def get_next(input, pos):
  pos = pos*100
  while pos+100 > len(input):
    pos -= len(input)
  return [input[i] for i in range(pos,pos+100)]


mnist = input_data.read_data_sets('MNIST_data/')


#Temporary labels, later on should be based on actual image labels and dataset size
labels = [3]*1016

#-------------------------Creating Tensorflow Session--------------------------
#Create an array with single a-axis
fig, ax = plt.subplots(nrows=20, ncols=11, figsize=(10, 20))
#Set Tensorflow monitored session

max_epochs = 1000
log_file = open("run_log.log", 'w')

plot_iter = 0

with tf.train.MonitoredSession() as sess:
#------------------------------Running Session---------------------------------
  #Set max num of epoch
  for epoch in range(max_epochs):
    #Reshaping images to parameters (**NumberOfImages, ImageWidth, ImageHeight**, ColorDimension)
    feed = {data: all_pics}
    #Runs with Error Cost, Code, and Images
    test_elbo, test_codes, test_samples = sess.run([elbo, code, samples], feed)
    #Prints out epochs and error cost



    #print('Epoch', epoch, 'elbo', test_elbo)
    #Log epoch and elbo to file
    log_file.write("Epoch: " + str(epoch) + " Eblo: " + str(test_elbo) + "\n")


    
    #Plot only the last 20 images
    if epoch > max_epochs - 20 and not plot_iter > 19:
      ax[plot_iter, 0].set_ylabel('Epoch {}'.format(epoch))
      #Plots code on current epoch
      plot_codes(ax[plot_iter, 0], test_codes, labels)
      #Plots Images on current epoch
      plot_samples(ax[plot_iter, 1:], test_samples)
      plot_iter += 1
#---------------------------------Optimizer--------------------------------------
    for i in range(600):
      feed = {data: get_next(all_pics, i)}
      #Optimize based on Error Cost
      sess.run(optimize, feed)
#----------------------------------Output-----------------------------------------
#Saves images to output file
plt.savefig('vae-mnist.png', dpi=300, transparent=True, bbox_inches='tight')

#######################################################################
