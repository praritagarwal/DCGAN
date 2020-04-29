#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow.keras as keras
import time
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Reshape, Flatten
from tensorflow.keras.layers import Conv2D, Conv2DTranspose


# In[3]:


class DCGAN():
    def __init__(self, restore_scale = None):
        
        # restore_scale is the function to be used to restore the scale of images when
        # displaying a sample of generated images
        
        self.generator = None
        self.discriminator = None
        self.GAN = None
        
        # function to be used to restore the scale of images
        self.restore_scale = restore_scale
        
    # function to build a generator of given specifications
    # Assumption: the first layer is a dense layer and 
    # the rest are transpose Convolutional layers
    # list layers consists of a list of tuples except for 
    # the first entry which is a tuple: (num_units, activation) for the dense layer
    # the second entry is tuple of two objects: the height and the width of 
    # the feature maps obtained from reshaping the output of the first layer
    # the rest of the enteries are tuples: (num_units, kernel_size, stride, padding, activation) for
    # the transpose convolutional layers
    # if batch_normalization == True, a BatchNormalization layer will be included after 
    # every hidden layer except the first dense layer
    # if batch_normalization == False and drop_prob!=0, then 
    # a dropout layer is added after every layer except the dense layer    
    def build_generator(self, list_layers, coding_size = 100, batch_normalization = True,
                        drop_prob = 0, kernel_initializer = 'glorot_uniform'):  
        dense_units, dense_act = list_layers.pop(0)
        reshape_x, reshape_y = list_layers.pop(0)
        reshape_channels = int(dense_units/(reshape_x*reshape_y))
    
        generator = Sequential( name = 'generator')
        generator.add(Dense(units=dense_units, activation = dense_act, 
                        input_shape=[coding_size], kernel_initializer = kernel_initializer))
        generator.add(Reshape([reshape_x, reshape_y, reshape_channels]))
        
        for num_units, kernel_size, stride, padding, activation in list_layers:
            if batch_normalization:
                generator.add(BatchNormalization())
            elif drop_prob!=0:
                generator.add(Dropout(drop_prob))
                
            trans_conv = Conv2DTranspose(filters = num_units, kernel_size = kernel_size, 
                                         strides = stride, padding = padding, 
                                         activation = activation, 
                                         kernel_initializer = kernel_initializer)
            generator.add(trans_conv)
        
        
        self.generator = generator
    
    
    # function to build a generator with given specifications
    # list_layers consists of a list of tuples used to describe each convolutional layer
    # tuples: (num_units, kernel_size, stride, padding, activation)
    # if batch_normalization is True, then each layer is followed by 
    # a BatchNormalization layer
    # if batch_normalization is False and drop_prob!=0, then each layer is followed by 
    # a dropout layer
    def build_discriminator(self, list_layers, input_shape, batch_normalization = False, 
                        drop_prob = 0.3, kernel_initializer = 'glorot_uniform'):
        
        discriminator = Sequential( name = 'discriminator')
        filters, kernel_size, stride, padding, activation = list_layers.pop(0) 
        
        conv2d = Conv2D(filters = filters, kernel_size = kernel_size,
                        strides = stride, padding = padding, 
                        activation = activation, input_shape = input_shape, 
                        kernel_initializer = kernel_initializer)
        
        discriminator.add(conv2d)
    
        for filters, kernel_size, stride, padding, activation  in list_layers:
            
            if batch_normalization:
                discriminator.add(BatchNormalization())
            elif drop_prob!=0:
                discriminator.add(Dropout(drop_prob))
        
            conv2d = Conv2D(filters = filters, kernel_size = kernel_size,
                            strides = stride, padding = padding, activation = activation,
                            kernel_initializer = kernel_initializer)
        
            discriminator.add(conv2d)
        
        discriminator.add(Flatten())
        discriminator.add(Dense(1, activation = 'sigmoid'))
        
        self.discriminator = discriminator
        
    def build_GAN(self, optimizer, loss = 'binary_crossentropy', metrics = ['accuracy']):
        
        '''function to build and compile the GAN '''
        
        if (not self.discriminator) or (not self.generator):
            print('Error: Either the generator or the discriminator has not been built yet!')
            return
        
        self.GAN = Sequential([self.generator, self.discriminator])
        self.discriminator.compile(loss = loss , optimizer = optimizer, metrics = metrics)
        self.discriminator.trainable = False
        self.GAN.compile(loss = loss, optimizer = optimizer, metrics = metrics)
    
   
    def train_gan(self, training_images, num_epochs = 2, batch_size = 32,
                  coding_size = 100, num_plots = 10):
        
        '''function to train the GAN'''
        
        plot_after = int(num_epochs/num_plots) # num epochs after which 
                                           # to plot an image to 
                                           # track the evolution of the GAN 
            
        # sample codings to track the evolution of the GAN        
        sample_codings = tf.random.normal(shape = (10, coding_size)) 
        
        idxs = tf.range(2*batch_size) # indices of the img_batch obtained from combining 
                                      # true images with generated images
        for epoch in tf.range(num_epochs):
            start = time.time()
            batch_accuracy = []
            for true_imgs in training_images:
                # generate a set of random codings
                codings = tf.random.normal(shape = (batch_size, coding_size))
                # convert the codings into images
                gan_imgs = self.generator(codings)
                y_gan = tf.constant([[0.]]*batch_size)
                
                # mix the gan images with the true images
                y_true = tf.constant([[1.]]*batch_size)
                all_imgs = tf.concat([true_imgs, gan_imgs], axis = 0)
                all_y = tf.concat([y_true, y_gan], axis = 0)
                
                # shuffle the set of mixed images
                # shuffling the set of mixed images takes a large amount of time
                # I therefore decided to not do it
                # shuffled_idxs = tf.random.shuffle(idxs)
                # shuffled_imgs = tf.gather(all_imgs, shuffled_idxs)
                # shuffled_y = tf.gather(all_y, shuffled_idxs)
                
                # phase 1: train the discriminator
                self.discriminator.trainable = True
                #self.discriminator.train_on_batch(shuffled_imgs, shuffled_y)
                self.discriminator.train_on_batch(all_imgs, all_y)
                
                #pred = self.discriminator(shuffled_imgs)
                #_, acc = self.discriminator.evaluate(shuffled_imgs, shuffled_y, 
                #                                     batch_size = 2*batch_size, verbose = 0)
                pred = self.discriminator(all_imgs)
                _, acc = self.discriminator.evaluate(all_imgs, all_y, 
                                                     batch_size = 2*batch_size, verbose = 0)
                batch_accuracy.append(acc)
                # phase 2: train the generator on a new set of codings
                new_codings = tf.random.normal(shape = (batch_size, coding_size))
                self.discriminator.trainable = False
                self.GAN.train_on_batch(new_codings, 1-y_gan)
                
            # evaluate the GAN's performance so far
            mean_acc = tf.math.reduce_mean(tf.constant(batch_accuracy), axis = 0)
            end = time.time()
            time_taken = end - start
            print_info = (epoch+1,num_epochs, time_taken, mean_acc)
            print('\repoch:{0:}/{1:}, time:{2: .2f}s, disc. accuracy:{3: .2f} '.format(*print_info), 
                  end = '', flush = True)
            print('=', end = '')
            
            # plot the images produced by the GAN at different stages of training
            if epoch%plot_after == 0:
                sample_imgs = self.generator(sample_codings)
                predictions = self.discriminator(sample_imgs)
                # reshape the image if grayscale
                _ ,height, width, channels = sample_imgs.shape
                if channels == 1:
                    sample_imgs = tf.reshape(sample_imgs,[-1, height, width])
                # rescale the image pixels to be between 0 and 1
                if self.restore_scale:
                    sample_imgs = self.restore_scale(sample_imgs)
                fig, ax = plt.subplots(figsize = (15, 5), ncols = 10 )
                for col in range(10):
                    ax[col].imshow(sample_imgs[col])
                    ax[col].axis('off')
                    ax[col].set_title('pred: {: .2f}'.format(predictions[col][0]))
                plt.show() 
        


# In order to convert this notebook into a python module, use the following in command line:
# 
# ``` ipython nbconvert --to python my_DCGAN_class.ipynb```
# 
# This was suggested by Sarath Ak in the [this](https://stackoverflow.com/questions/52885901/how-to-save-python-script-as-py-file-on-jupyter-notebook/52886052) stackexchange post.
# 
# ``` ipython``` has been depricated. Use ```jupyter nbconvert --to python my_DCGAN_class.ipynb``` instead.
# 
