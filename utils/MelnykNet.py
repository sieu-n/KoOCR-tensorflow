#Referenced from original implementation of Melnyk-net: https://github.com/pavlo-melnyk/offline-HCCR/blob/master/src/melnyk_net.py
import os

import numpy as np

from tensorflow.keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, GlobalAveragePooling2D, Activation, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import RandomNormal

from utils.CustomLayers import GlobalWeightedAveragePooling
from utils.model_architectures import PreprocessingPipeline,build_FC_split,build_FC_regular

def melnyk_net(settings):	
	if settings['fc_link'] == 'GWAP':
		GlobalAveragePooling = GlobalWeightedAveragePooling()
	elif settings['fc_link'] == 'GAP':
		GlobalAveragePooling = GlobalAveragePooling2D()
	else:
		GlobalAveragePooling = Flatten()

    random_normal = RandomNormal(stddev=0.001) # output layer initializer
    reg=0

	input_image=tf.keras.layers.Input(shape=(settings['input_shape'],settings['input_shape']))
    preprocessed=tf.keras.layers.Reshape((settings['input_shape'],settings['input_shape'],1))(input_image)
    preprocessed=PreprocessingPipeline(settings['direct_map'])(preprocessed)

	x = Conv2D(64, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
		kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(preprocessed)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(64, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
		kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

	x = Conv2D(96, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
		kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(64, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
		kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(96, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
		kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

	x = Conv2D(128, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
		kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(96, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
		kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(128, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
		kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

	x = Conv2D(256, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
		kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
		kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(256, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
		kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

	x = Conv2D(448, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
		kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(256, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
		kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(448, (3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal', use_bias=False, 
		kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	if settings['split_components']:
        #mid1_CHO,mid1_JUNG,mid1_JONG=build_FC_split(fire5,tag='mid1_')
        #mid2_CHO,mid2_JUNG,mid2_JONG=build_FC_split(fire7,tag='mid2_')
        CHO,JUNG,JONG=build_FC_split(x,GAP=settings['fc_link'])
        
        return tf.keras.models.Model(inputs=input_image,outputs=[CHO,JUNG,JONG])
    else:
        x=build_FC_regular(x)
        return tf.keras.models.Model(inputs=input_image,outputs=x)

	return model