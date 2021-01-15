import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import dataset
import korean_manager
from PIL import Image
import os

class KoOCR():
    def __init__(self,split_components=True):
        self.split_components=split_components

        self.charset=korean_manager.load_charset()

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

        input_image=tf.keras.layers.Input(shape=(256,256),name='input_image')
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
            CHO=tf.keras.layers.Dense(len(korean_manager.CHOSUNG_LIST),activation='softmax',name='CHOSUNG')(x)
            JUNG=tf.keras.layers.Dense(len(korean_manager.JUNGSUNG_LIST),activation='softmax',name='JUNGSUNG')(x)
            JONG=tf.keras.layers.Dense(len(korean_manager.JONGSUNG_LIST),activation='softmax',name='JONGSUNG')(x)

            return tf.keras.models.Model(inputs=input_image,outputs=[CHO,JUNG,JONG])
        else:
            x=tf.keras.layers.Dense(len(self.charset),activation='softmax',name='output')(x)
            return tf.keras.models.Model(inputs=input_image,outputs=x)
            
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

    def train(self,epochs=10,lr=0.001,data_path='./data',patch_size=10,epoch_checkpoint=True):
        self.dataset=dataset.DataPickleLoader(split_components=self.split_components,data_path=data_path,patch_size=patch_size)

        self.compile_model(lr)
        
        checkpoint_dir = './training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(KoOCR=self.model)

        for epoch in range(epochs):
            print('Training epoch',epoch)

            epoch_end=False
            while epoch_end==False:
                train_x,train_y,epoch_end=self.dataset.get()

                self.model.fit(x=train_x,y=train_y,verbose=0,epochs=1)

            #Save weights in checkpoint
            if epoch_checkpoint:
                checkpoint.save(file_prefix = checkpoint_prefix)

        self.model.save('./logs')