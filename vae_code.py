# Full example for my blog post at:
# https://danijar.com/building-variational-auto-encoders-in-tensorflow/

############################ Yvannia ################################

# Importing the package --> "numpy" so that all the objects defined in the module can be used. Imported as "np" for easy referral (renaming)
import numpy as np
# Importing the package --> "matplotlib.pyplot" so that all the objects defined in the module can be used. Imported as "plt" for easy referral (renaming)
import matplotlib.pyplot as plt
# Importing the package --> "tensorflow" so that all the objects defined in the module can be used. Imported as "tf" for easy referral (renaming)
import tensorflow as tf
# Importing a specific object named "input_data" to have shorter calls later on in the code 
from tensorflow.examples.tutorials.mnist import input_data

# contrib makes it easier to configure, train, and evaluate a variety of machine learning models
# distributions is a class that is used for constructing and organizing properties (mean, variance, standard deviation, etc.) of variables 
# Here it is shortened to "tfd" to avoid long code 
tfd = tf.contrib.distributions


# Encoder function --------------------------------------------------------------------------------------------------------------
# Function definition of the encoder that takes in two parameters, data and code_size 
def make_encoder(data, code_size):

# Layers is an object that involves computation --> it is an important method used to create a neural network 
# Flatten is used to compress the input, which in this case is the parameter "data"
# Flatten is applied to the layer having a shape such as (size, 2,2), and then outputs the shape as (size, 4). i.e. collapses spatial dimensions 
  x = tf.layers.flatten(data)

# There are two identical calls because the first is a hidden layer (where every node is connected to every other node in the next layer) 
# and the second is an output layer (the last layer of the node connections). 
# Dense is a network layer. It feeds all outputs from the previos layer, to all its neurons. 
# x is the previous layers output defined above and 200 is the number of neurons. 
# nn stands for neural network --> provides support for many basic neural network operations 
# relu stands for Rectified Linear Unit which is a function --> it is computationally faster and allows for fewer vanishing gradients. 
  x = tf.layers.dense(x, 200, tf.nn.relu)
  x = tf.layers.dense(x, 200, tf.nn.relu)
  
# Same as above, but instead of taking the parameter 200 it takes in the input "code_size"
# loc is the mean 
  loc = tf.layers.dense(x, code_size)
  
# Same as above but now using "softplus" --> provides more stabilization and performance to deep neural network (than ReLU function).
# scale is the standard deviation. 
  scale = tf.layers.dense(x, code_size, tf.nn.softplus)

# Implement multivariate normal distributions with a diagonal covariance structure.
# Takes in parameter loc = mean and scale = standard deviation. 
# Returns a graph of the distribution that will be compared against what it should look like later. 
  return tfd.MultivariateNormalDiag(loc, scale)


# Make_Prior function --------------------------------------------------------------------------------------------------------------
# Function definition of make_prior that takes in one parameters, code_size. 
def make_prior(code_size):
# tf.zeros creates a tensor (a multi-dimensional array with a uniform type) with all elements of the input "code_size" set to zero.  
  loc = tf.zeros(code_size)
# tf.ones creates a tensor with all elements of the input "code_size" set to one. 
  scale = tf.ones(code_size)
# Same explanation as above
# Returns a graph of what the distribution should look like that is compared against the returned graph above. 
  return tfd.MultivariateNormalDiag(loc, scale)

######################################################################



############################ Botoul ##################################

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
  return tfd.Independent(tfd.Bernoulli(logit), 2)

#######################################################################

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
    ax[index].imshow(sample, cmap='gray')
    ax[index].axis('off')


############################ Logan ##################################

#Create placeholder data in dimension ?x28x28, has no represented data values
#'None' represents an unknown dimension
data = tf.placeholder(tf.float32, [None, 28, 28])

#Create function templates, this ensures that function specific variables are initialized first and consistent between all calls of this function
make_encoder = tf.make_template('encoder', make_encoder)
make_decoder = tf.make_template('decoder', make_decoder)


# Define the model------------------------------------------------------

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


# Define the loss-------------------------------------------------------
#We need to compute the negative log-likelihood, so we use our decoder to find log(p(x|z))
#Data is used as a template
likelihood = make_decoder(code, [28, 28]).log_prob(data)
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
samples = make_decoder(prior.sample(10), [28, 28]).mean()

######################################################################


############################ Sean ####################################
#Simplifies the input data into mnist
mnist = input_data.read_data_sets('MNIST_data/')
#-------------------------Creating Tensorflow Session--------------------------
#Create an array with single a-axis
fig, ax = plt.subplots(nrows=20, ncols=11, figsize=(10, 20))
#Set Tensorflow monitored session
with tf.train.MonitoredSession() as sess:
#------------------------------Running Session---------------------------------
  #Set max num of epoch
  for epoch in range(20):
    #Reshaping images to parameters (**NumberOfImages, ImageWidth, ImageHeight**, ColorDimension)
    feed = {data: mnist.test.images.reshape([-1, 28, 28])}
    #Runs with Error Cost, Code, and Images
    test_elbo, test_codes, test_samples = sess.run([elbo, code, samples], feed)
    #Prints out epochs and error cost
    print('Epoch', epoch, 'elbo', test_elbo)
    #Plots epoch as y-axis
    ax[epoch, 0].set_ylabel('Epoch {}'.format(epoch))
    #Plots code on current epoch
    plot_codes(ax[epoch, 0], test_codes, mnist.test.labels)
    #Plots Images on current epoch
    plot_samples(ax[epoch, 1:], test_samples)
#---------------------------------Optimizer--------------------------------------
    for _ in range(600):
      feed = {data: mnist.train.next_batch(100)[0].reshape([-1, 28, 28])}
      #Optimize based on Error Cost
      sess.run(optimize, feed)
#----------------------------------Output-----------------------------------------
#Saves images to output file
plt.savefig('vae-mnist.png', dpi=300, transparent=True, bbox_inches='tight')

#######################################################################
