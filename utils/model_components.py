import tensorflow as tf
import numpy as np
import utils.korean_manager as korean_manager
import utils.CustomLayers as CustomLayers

def build_FC_split(x,GAP='',tag=''):
    if GAP=='':
        x=tf.keras.layers.Flatten()(x)
    elif GAP=='GAP':
        x=tf.keras.layers.GlobalAveragePooling2D()(x)
    elif GAP=='GWAP':
        x=CustomLayers.GlobalWeightedAveragePooling()(x)
    #x=tf.keras.layers.Dense(1024)(x)

    CHO=tf.keras.layers.Dense(len(korean_manager.CHOSUNG_LIST),activation='softmax',name=tag+'CHOSUNG')(x)
    JUNG=tf.keras.layers.Dense(len(korean_manager.JUNGSUNG_LIST),activation='softmax',name=tag+'JUNGSUNG')(x)
    JONG=tf.keras.layers.Dense(len(korean_manager.JONGSUNG_LIST),activation='softmax',name=tag+'JONGSUNG')(x)
    return CHO,JUNG,JONG

def build_FC_regular(x):
    x=tf.keras.layers.Flatten()(x)
    x=tf.keras.layers.Dense(1024)(x)
    x=tf.keras.layers.Dense(len(korean_manager.load_charset()),activation='softmax',name='output')(x)
    return x

def PreprocessingPipeline(direct_map):
    preprocessing=tf.keras.models.Sequential()

    #[0, 255] to [0, 1] with black white reversed
    preprocessing.add(tf.keras.layers.experimental.preprocessing.Rescaling(scale=-1/255,offset=1))
    #DirectMap normalization
    if direct_map==True:
        preprocessing.add(DirectMapGeneration())
    return preprocessing

def DirectMapGeneration():
    #Generate sobel filter for 8 direction maps
    sobel_filters=[
        [[-1,0,1],
        [-2,0,2],
        [-1,0,1]],

        [[1,0,-1],
        [2,0,-2],
        [1,0,-1]],

        [[1,2,1],
        [0,0,0],
        [-1,-2,-1]],

        [[-1,-2,-1],
        [0,0,0],
        [1,2,1]],

        [[0,1,2],
        [-1,0,1],
        [-2,-1,0]],

        [[0,-1,-2],
        [1,0,-1],
        [2,1,0]],

        [[-2,-1,0],
        [-1,0,1],
        [0,1,2]],

        [[2,1,0],
        [1,0,-1],
        [0,-1,-2]]]
    sobel_filters=np.array(sobel_filters).astype(np.float).reshape(8,3,3,1)
    sobel_filters=np.moveaxis(sobel_filters, 0, -1)

    DirectMap = tf.keras.models.Sequential()
    DirectMap.add(tf.keras.layers.Conv2D(8, (3,3), padding='same',input_shape=(None, None, 1),use_bias=False))
    DirectMap.set_weights([sobel_filters])
    return DirectMap