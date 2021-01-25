import tensorflow as tf
import numpy as np
import korean_manager

def build_FC_split(x):
    x=tf.keras.layers.Flatten()(x)
    x=tf.keras.layers.Dense(1024)(x)

    CHO=tf.keras.layers.Dense(len(korean_manager.CHOSUNG_LIST),activation='softmax',name='CHOSUNG')(x)
    JUNG=tf.keras.layers.Dense(len(korean_manager.JUNGSUNG_LIST),activation='softmax',name='JUNGSUNG')(x)
    JONG=tf.keras.layers.Dense(len(korean_manager.JONGSUNG_LIST),activation='softmax',name='JONGSUNG')(x)
    return CHO,JUNG,JONG

def build_FC_regular(x):
    x=tf.keras.layers.Flatten()(x)
    x=tf.keras.layers.Dense(1024)(x)
    x=tf.keras.layers.Dense(len(korean_manager.load_charset()),activation='softmax',name='output')(x)
    return x

def build_model(split_components=True,input_shape=256):
    def down_conv(channels,kernel_size=3,bn=True,activation='lrelu'):
        #Define single downsampling operation
        init = tf.random_normal_initializer(0., 0.02)

        down=tf.keras.models.Sequential()
        down.add(tf.keras.layers.Conv2D(channels,kernel_size,strides=(2,2),padding='same',kernel_initializer=init,use_bias=False))
        if bn:
            down.add(tf.keras.layers.BatchNormalization())
        
        if activation=='lrelu':
            down.add(tf.keras.layers.LeakyReLU())
        elif activation=='relu':
            down.add(tf.keras.layers.ReLU())
        elif activation=='sigmoid':
            down.add(tf.keras.layers.Activation(tf.nn.sigmoid))
        return down

    input_image=tf.keras.layers.Input(shape=(input_shape,input_shape),name='input_image')
    x=tf.keras.layers.Reshape((256,256,1))(input_image)
    x=down_conv(32)(x)  #(128,128,32)
    x=down_conv(64)(x)  #(64,64,64)
    x=down_conv(128)(x) #(32,32,128)
    x=down_conv(256)(x) #(16,16,256)
    x=down_conv(512)(x) #(8,8,512)
    x=down_conv(512)(x) #(4,4,512)

    #Define exterior
    if split_components:
        CHO,JUNG,JONG=build_FC_split(x)
        return tf.keras.models.Model(inputs=input_image,outputs=[CHO,JUNG,JONG])
    else:
        x=build_FC_regular(x)
        return tf.keras.models.Model(inputs=input_image,outputs=x)

def VGG16(split_components=True,input_shape=256):
    VGG_net = tf.keras.applications.VGG16(input_shape=(input_shape,input_shape,3),
                                               include_top=False,
                                               weights='imagenet')

    input_image=tf.keras.layers.Input(shape=(input_shape,input_shape))
    concat=tf.keras.layers.Reshape((input_shape,input_shape,1))(input_image)
    concat=tf.keras.layers.concatenate([concat,concat,concat])
    feature=VGG_net(concat)
    
    if split_components:
        CHO,JUNG,JONG=build_FC_split(feature)
        return tf.keras.models.Model(inputs=input_image,outputs=[CHO,JUNG,JONG])
    else:
        x=build_FC_regular(feature)
        return tf.keras.models.Model(inputs=input_image,outputs=x)

model_list={'custom':build_model,'VGG16':VGG16}