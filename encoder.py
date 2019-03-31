from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, AveragePooling3D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader_encoder import DataLoader
from head_pose.head_pose_estimation import CnnHeadPoseEstimator
import tensorflow as tf
from glob import glob
import pickle
import PIL.Image
import cv2
import config
import dnnlib
import dnnlib.tflib as tflib
import numpy as np
import os

url_ffhq = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'  # karras2019stylegan-ffhq-1024x1024.pkl

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

_Gs_cache = dict()


def load_Gs(url):
    print("loading GS")
    if url not in _Gs_cache:
        with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
            _G, _D, Gs = pickle.load(f)
        _Gs_cache[url] = Gs
    print("GS loaded")
    return _Gs_cache[url]

class Encoder():
    def __init__(self):
        # Input shape
        self.img_rows = 1024
        self.img_cols = 1024
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = 'out'
        self.data_loader = DataLoader(dataset_name=self.dataset_name, data_path='../data/out/')


        optimizer = Adam(0.0002, 0.5)

        self.Gs = load_Gs(url_ffhq)

        # Build and compile the discriminators
        self.encoder = self.build_encoder()
        self.encoder.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        print(self.encoder.summary())
        i=4

    def build_encoder(self):
        # Image input
        d0 = Input(shape=self.img_shape)

        d = Conv2D(20, (5, 5), padding="same")(d0)
        d=Activation("relu")(d)
        d=MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(d)

        #d = Dense(36, activation='tanh')(d)

        d = Conv2D(20, (5, 5), padding="same")(d)
        d=Activation("relu")(d)
        d=MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(d)

        #d = Dense(36, activation='tanh')(d)

        d = Conv2D(20, (5, 5), padding="same")(d)
        d = Activation("relu")(d)
        d = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(d)

        #d = Dense(36, activation='tanh')(d)

        d = Conv2D(20, (5, 5), padding="same")(d)
        d = Activation("relu")(d)
        d = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(d)

        #d = Dense(36, activation='tanh')(d)

        d = Conv2D(20, (5, 5), padding="same")(d)
        d = Activation("relu")(d)
        d = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(d)

        #d = Dense(36, activation='tanh')(d)

        d = Conv2D(20, (5, 5), padding="same")(d)
        d = Activation("relu")(d)
        d = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(d)

        #d = Dense(1, activation='tanh')(d)

        d=Flatten()(d)

        d=Dense(18*512, activation='tanh')(d)

        d = Reshape((18, 512))(d)

        return Model(d0, d)


    def train(self, epochs, batch_size=1, sample_interval=50):

        for epoch in range(epochs):
            for batch_i, (imgs, latent_vector) in enumerate(self.data_loader.load_batch(batch_size)):

                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                #encoder_prediction = self.encoder.predict(imgs)

                # Train the discriminators (original images = real / translated = Fake)
                loss = self.encoder.train_on_batch(imgs, latent_vector)



                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [loss: %f]" % ( epoch, epochs,
                                                                            batch_i, self.data_loader.n_batches,
                                                                            loss[0]))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    '''norm_images, images, latent_vectors = self.data_loader.load_test_data()
                    for i in range(len(images)):
                        np.save('../results/' +str(i)+'_orig_latent_vector.npy', latent_vectors[i])
                        np.save('../results/'+str(i)+'_predicted_latent_vector.npy', self.encoder.predict(norm_images)[i])
                        scipy.misc.imsave('../results/' + str(i) + '_image.png',images[i])'''

                    path_images = glob(os.path.join('../data/out/test/images_cage/*'))
                    imgs = []
                    for img_path in path_images:
                        img = scipy.misc.imread(img_path, mode='RGB').astype(np.float)
                        imgs.append(img)

                    imgs_norm = np.array(imgs) / 127.5 - 1.

                    prediction = self.encoder.predict(imgs_norm)

                    for i in range(len(imgs_norm)):
                        '''r, p, y = self.estimate_head_pose(imgs[i])
                        cv2.putText(imgs[i], str(r) + ', ' + str(p) + ', ' + str(y), (10, 50),
                                    cv2.FONT_HERSHEY_TRIPLEX, 1,
                                    color=(255, 0, 255))'''
                        scipy.misc.imsave('../results/' + str(i) + '_image.png', imgs[i])
                        np.save('../results/' + str(i) + '_predicted_latent_vector.npy',
                                prediction[i])



                    src_images = self.Gs.components.synthesis.run(prediction, randomize_noise=False,
                                                             **synthesis_kwargs)
                    i = 0
                    for image in src_images:
                        '''r, p, y = self.estimate_head_pose(image)
                        cv2.putText(image, str(r)+', '+str(p)+', '+str(y), (10, 50),
                                    cv2.FONT_HERSHEY_TRIPLEX, 1,
                                    color=(255, 0, 255))'''
                        #im = PIL.Image.fromarray(image)
                        scipy.misc.imsave('../results/' + str(i) + '.png',image)
                        i += 1
                    print('done')

                    #np.save('../results/'+str(batch_i)+'_prediction.npy',self.encoder.predict([im])[0])
                    #np.save('../results/prediction.npy', self.encoder.predict(imgs)[0])
                    #cv2.imwrite('../results/' + str(batch_i) + '.png',((imgs[0]+1)*127.5))
                    #im.save('../results/' + str(batch_i) + '.png')

    '''def estimate_head_pose(self, face_img):
         sess = tf.Session()  # Launch the graph in a session.
         my_head_pose_estimator = CnnHeadPoseEstimator(sess)  # Head pose estimation object
    #
    #     # Load the weights from the configuration folders
         my_head_pose_estimator.load_roll_variables(os.path.realpath("head_pose/model/roll/cnn_cccdd_30k.tf"))
         my_head_pose_estimator.load_pitch_variables(
             os.path.realpath("head_pose/model/pitch/cnn_cccdd_30k.tf"))
         my_head_pose_estimator.load_yaw_variables(os.path.realpath("head_pose/model/yaw/cnn_cccdd_30k.tf"))
    #
    #     # Get the angles for roll, pitch and yaw
         roll = my_head_pose_estimator.return_roll(face_img)  # Evaluate the roll angle using a CNN
         pitch = my_head_pose_estimator.return_pitch(face_img)  # Evaluate the pitch angle using a CNN
         yaw = my_head_pose_estimator.return_yaw(face_img)  # Evaluate the yaw angle using a CNN
         return roll[0, 0, 0], pitch[0, 0, 0], yaw[0, 0, 0]'''





if __name__ == '__main__':
    tflib.init_tf()
    os.makedirs(config.result_dir, exist_ok=True)
    encoder = Encoder()
    encoder.train(epochs=200, batch_size=1, sample_interval=200)