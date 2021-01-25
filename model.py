import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import dataset
import korean_manager
from PIL import Image
import random
import os
from IPython.display import clear_output
import gc
class KoOCR():
    def __init__(self,split_components=True,weight_path=''):
        self.split_components=split_components

        self.charset=korean_manager.load_charset()

        self.model=self.build_model()
        if weight_path:
            self.model=tf.keras.models.load_model(weight_path)
    
    def predict(self,image,n=1):
        #Predict the top-n classes of the image
        #k: top classes for each component to generate
        k=int(n**0.3)+1
        if image.shape==(256,256):
            image=image.reshape((1,256,256))
        #Predict top n classes
        cho_pred,jung_pred,jong_pred=self.model.predict(image)
        cho_idx,jung_idx,jong_idx=np.argsort(cho_pred,axis=1)[-k:],np.argsort(jung_pred,axis=1)[-k:],np.argsort(jong_pred,axis=1)[-k:]
        cho_pred,jung_pred,jong_pred=np.sort(cho_pred,axis=1)[-k:],np.sort(jung_pred,axis=1)[-k:],np.sort(jong_pred,axis=1)[-k:]
        #Convert indicies to korean character
        pred_hangeul=[]
        for idx in range(image.shape[0]):
            pred_hangeul.append([])

            cho_prob,jung_prob,jong_prob=cho_pred[idx],jung_pred[idx].reshape(-1,1),jong_pred[idx].reshape(-1,1)

            mult=((cho_prob*jung_prob).flatten()*jong_prob).flatten().argsort()[-5:][::-1]
            for max_idx in mult:
                print(n%3,(n%9)//3,n//9)
                pred_hangeul[-1].append(korean_manager.index_to_korean((cho_idx[idx][max_idx%k],jung_idx[idx][(max_idx%(k*k))//k],jong_idx[idx][max_idx//9])))

        return pred_hangeul
    
    def plot_val_image(self,data_path='./data'):
        #Load validation data
        train_dataset=dataset.DataPickleLoader(split_components=self.split_components,data_path=data_path,patch_size=1)
        val_x,val_y=train_dataset.get_val()
        #Predict classes
        indicies=random.sample(range(len(val_x)),10)
        val_x=val_x[indicies]
        pred_y=self.predict(val_x)

        fig = plt.figure(figsize=(10,1))
        for idx in range(10):
            plt.subplot(1,10,idx+1)
            plt.imshow(val_x[idx],cmap='gray')
            plt.axis('off')
        plt.savefig('./logs/image.png')
        print(pred_y)

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

    def train(self,epochs=10,lr=0.001,data_path='./data',patch_size=10):
        train_dataset=dataset.DataPickleLoader(split_components=self.split_components,data_path=data_path,patch_size=patch_size)
        val_x,val_y=train_dataset.get_val()

        self.compile_model(lr)
        
        for epoch in range(epochs):
            print('Training epoch',epoch)
            self.plot_val_image(data_path=data_path)
            epoch_end=False
            while epoch_end==False:
                
                train_x,train_y,epoch_end=train_dataset.get()

                self.model.fit(x=train_x,y=train_y,epochs=1,validation_data=(val_x,val_y))

                #Clear garbage memory
                tf.keras.backend.clear_session()
                gc.collect()
                clear_output(wait=True)
                
            #Save weights in checkpoint
            self.model.save('./logs/weights.h5')

        