#Referenced from original implementation of Melnyk-net: https://github.com/pavlo-melnyk/offline-HCCR/blob/master/src/melnyk_net.py
import os

import numpy as np

from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Reshape, GlobalAveragePooling2D, Activation, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import RandomNormal

from utils.CustomLayers import GlobalWeightedAveragePooling
import utils.model_architectures as model_architectures

def melnyk_net(settings):	
	random_normal = RandomNormal(stddev=0.001)
	reg=0

	input_image=Input(shape=(settings['input_shape'],settings['input_shape']))
	preprocessed=Reshape((settings['input_shape'],settings['input_shape'],1))(input_image)
	preprocessed=model_architectures.PreprocessingPipeline(settings['direct_map'])(preprocessed)

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
		CHO,JUNG,JONG=model_architectures.build_FC_split(x,GAP=settings['fc_link'])
        
		return Model(inputs=input_image,outputs=[CHO,JUNG,JONG])
	else:
		x=model_architectures.build_FC_regular(x)
		return Model(inputs=input_image,outputs=x)

	return model