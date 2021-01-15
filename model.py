import numpy as np
import tensorflw as tf
import matplotlib.pyplot as plt
import dataset
from crawl_data import load_charset
from PIL import Image

class KoOCR():

    def __init__(self,split_components=True):
        self.split_components=split_components

        self.charset=load_charset()

        self.model=self.build_model()

    def predict(self,image_path):
        image=Image.open(image_path).convert('LA')

    def build_model(self):
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
        korean=dataset.KoreanManager()

        input_image=tf.keras.layers.Input(shape=(256,256))
        x=tf.keras.layers.Reshape((256,256,1))(input_image)
        x=down_conv(32)(x)  #(128,128,32)
        x=down_conv(64)(x)  #(64,64,64)
        x=down_conv(128)(x) #(32,32,128)
        x=down_conv(256)(x) #(16,16,256)
        x=down_conv(512)(x) #(8,8,512)
        x=down_conv(512)(x) #(4,4,512)
        
        x=tf.keras.layers.Flatten()(x)
        x=tf.keras.layers.Dense(1024)(x)

        if self.split_components:
            CHO=tf.keras.layers.Dense(len(korean.CHOSUNG_LIST),activation='softmax',name='CHOSUNG')(x)
            JUNG=tf.keras.layers.Dense(len(korean.JUNGSUNG_LIST),activation='softmax',name='JUNGSUNG')(x)
            JONG=tf.keras.layers.Dense(len(korean.JONGSUNG_LIST),activation='softmax',name='JONGSUNG')(x)

            return tf.keras.models.Model(input=input_image,output=[CHO,JUNG,JONG])
        else:
            x=tf.keras.layers.Dense(len(self.charset),activation='softmax',name='output')(x)
            return tf.keras.models.Model(input=input_image,output=x)
    def compile_model(self,lr):
        #Compile model 
        optimizer=tf.keras.optimizers.Adam(lr)
        if self.split_components:
            losses = {
                "CHOSUNG": "categorical_crossentropy",
                "JUNGSUNG": "categorical_crossentropy",
                "JONGSUNG": "categorical_crossentropy"}
        else:
            losses="categorical_crossentropy"

        self.model.compile(optimizer=optimizer, loss=losses,metrics=["accuracy"])

    def train(self,eopchs=10,lr=0.001,data_path='./data',patch_size=10):
        self.dataset=dataset.DataPickleLoader(split_components=self.split_components,data_path=data_path,patch_size=patch_size)

        self.compile_model(lr)

        for epoch in range(eopchs):
            print('Training epoch',epoch)

            epoch_end=False
            while epoch_end==False:
                images,labels,epoch_end=self.dataset.get()

                self.model.fit(x=images,y=labels)
