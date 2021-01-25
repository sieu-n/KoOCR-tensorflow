import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import dataset
import korean_manager
from PIL import Image
import random
import model_architectures
import os
from IPython.display import clear_output
import gc

class KoOCR():
    def __init__(self,split_components=True,weight_path='',model_type='custom'):
        self.split_components=split_components

        self.charset=korean_manager.load_charset()

        self.model=model_architectures.model_list[model_type](split_components=split_components,input_shape=256)

        if weight_path:
            self.model=tf.keras.models.load_model(weight_path)
    
    def predict(self,image,n=1):
        #Predict the top-n classes of the image
        #k: top classes for each component to generate
        #Returns top n characters that maximize (p|chosung)*(p|jungsung)*(p|jongsung)
        k=int(n**(1/3))+2
        print(k)
        if image.shape==(256,256):
            image=image.reshape((1,256,256))
        #Predict top n classes
        cho_pred,jung_pred,jong_pred=self.model.predict(image)
        cho_idx,jung_idx,jong_idx=np.argsort(cho_pred,axis=1)[:,-k:],np.argsort(jung_pred,axis=1)[:,-k:],np.argsort(jong_pred,axis=1)[:,-k:]
        cho_pred,jung_pred,jong_pred=np.sort(cho_pred,axis=1)[:,-k:],np.sort(jung_pred,axis=1)[:,-k:],np.sort(jong_pred,axis=1)[:,-k:]
        #Convert indicies to korean character
        pred_hangeul=[]
        for idx in range(image.shape[0]):
            pred_hangeul.append([])

            cho_prob,jung_prob,jong_prob=cho_pred[idx],jung_pred[idx].reshape(-1,1),jong_pred[idx].reshape(-1,1)

            mult=((cho_prob*jung_prob).flatten()*jong_prob).flatten().argsort()[-5:][::-1]
            for max_idx in mult:
                pred_hangeul[-1].append(korean_manager.index_to_korean((cho_idx[idx][max_idx%k],jung_idx[idx][(max_idx%(k*k))//k]\
                    ,jong_idx[idx][max_idx//(k*k)])))

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

        