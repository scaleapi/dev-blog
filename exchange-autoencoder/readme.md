# How to Build an Autoencoder Using Keras and TensorFlow

**Subtitle**: How to build an autoencoder for image compression, image reconstruction and supervised learning using the TensorFlow library

**Summary**: You can easily build an autoencoder using objects and APIs from the Keras and TensorFlow library. Here are the basics of what you need to know to build your own autoencoder.

**Byline**: Mehreen Saeed 

An autoencoder has two parts: an encoder and a decoder. The encoder learns a latent representation of the input data and the decoder is trained to reconstruct the original inputs from the latent representations. The autoencoder has the following applications.

- An autoencoder approximates the original input points from the latent representations. This makes it useful for data recovery from corrupt inputs. 
- As the autoencoder learns a latent representation of the input data, it can be designed so that the dimensions of this latent space is much smaller than the original input dimensions. Hence, an autoencoder can be used for data compression.
- Autoencoders find their application for data augmentation. The outputs from the autoencoder represent synthetic data and hence, can be added to the original training set to increase its size.
- The latent representation from an autoencoder can be used to learn classification and regression tasks.

In this article we'll work with a faces dataset to build a simple autoencoder. We'll use it for reconstructing the original face images. We'll also visualize the latent space and build a supervised classifier from it that performs face recognition. For the implementation part, we'll use TensorFlow and Keras library to build our model. 

It is important to note that the output shown in this article will not match the output that you get at your end. It will vary with each run of the program because of the stochastic nature of the algorithms involved. 
