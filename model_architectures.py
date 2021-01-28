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

def build_model(split_components=True,input_shape=256,direct_map=True):
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

def VGG16(split_components=True,input_shape=256,direct_map=True):
    if direct_map:
        input_channels=8
    else:
        input_channels=1
    VGG_net = tf.keras.applications.VGG16(input_shape=(input_shape,input_shape,input_channels),
                                               include_top=False,weights=None)

    input_image=tf.keras.layers.Input(shape=(input_shape,input_shape))
    preprocessed=PreprocessingPipeline(direct_map=direct_map)(input_image)
    
    feature=VGG_net(preprocessed)
    
    if split_components:
        CHO,JUNG,JONG=build_FC_split(feature)
        return tf.keras.models.Model(inputs=input_image,outputs=[CHO,JUNG,JONG])
    else:
        x=build_FC_regular(feature)
        return tf.keras.models.Model(inputs=input_image,outputs=x)

def InceptionResnetV2(split_components=True,input_shape=256,direct_map=True):
    if direct_map:
        input_channels=8
    else:
        input_channels=1
    InceptionResnet = tf.keras.applications.InceptionResNetV2(input_shape=(input_shape,input_shape,input_channels),
                                               include_top=False,weights=None)

    input_image=tf.keras.layers.Input(shape=(input_shape,input_shape))
    preprocessed=PreprocessingPipeline(direct_map=direct_map)(input_image)

    feature=InceptionResnet(preprocessed)
    
    if split_components:
        CHO,JUNG,JONG=build_FC_split(feature)
        return tf.keras.models.Model(inputs=input_image,outputs=[CHO,JUNG,JONG])
    else:
        x=build_FC_regular(feature)
        return tf.keras.models.Model(inputs=input_image,outputs=x)

def MobilenetV3(split_components=True,input_shape=256,direct_map=True):
    if direct_map:
        input_channels=8
    else:
        input_channels=1
    Mobilenet = tf.keras.applications.MobileNetV3Small(input_shape=(input_shape,input_shape,input_channels),
                                               include_top=False, weights=None)

    input_image=tf.keras.layers.Input(shape=(input_shape,input_shape))
    preprocessed=PreprocessingPipeline(direct_map=direct_map)(input_image)
    
    feature=Mobilenet(preprocessed)
    
    if split_components:
        CHO,JUNG,JONG=build_FC_split(feature)
        return tf.keras.models.Model(inputs=input_image,outputs=[CHO,JUNG,JONG])
    else:
        x=build_FC_regular(feature)
        return tf.keras.models.Model(inputs=input_image,outputs=x)

model_list={'custom':build_model,'VGG16':VGG16,'inception-resnet':InceptionResnetV2,'mobilenet':MobilenetV3}

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
    DirectMap.add(tf.keras.layers.Conv2D(8, (3,3), input_shape=(None, None, 1),use_bias=False))
    DirectMap.set_weights([sobel_filters])
    return DirectMap