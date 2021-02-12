import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import dataset
import utils.korean_manager as korean_manager
from PIL import Image
import random
import os
from IPython.display import clear_output
import gc
import datetime

from utils.adabound import AdaBound
from utils.model_architectures import VGG16,InceptionResnetV2,MobilenetV3,EfficientCNN
from utils.Melnyk import melnyk_net
class KoOCR():
    def __init__(self,split_components=True,weight_path='',fc_link='',network_type='custom',image_size=256,direct_map=True):
        self.split_components=split_components
        self.charset=korean_manager.load_charset()

        #Build and load model
        if weight_path:
            self.model = tf.keras.models.load_model('./logs/model.h5')
        else:
            model_list={'VGG16':VGG16,'inception-resnet':InceptionResnetV2,'mobilenet':MobilenetV3,'efficient-net':EfficientCNN,'melnyk':melnyk_net}
            settings={'split_components':split_components,'input_shape':image_size,'direct_map':direct_map,'fc_link':fc_link}
            self.model=model_list[network_type](settings)

    def predict(self,image,n=1):
        if self.split_components:
            return self.predict_split(image,n)
        else:
            return self.predict_complete(image,n)
            
    def predict_complete(self,image,n=1):
        #Predict the top-n classes of the image
        #Returns top n characters that maximize the probability
        charset=korean_manager.load_charset()
        if image.shape==(256,256):
            image=image.reshape((1,256,256))
        pred_class=self.model.predict(image)
        pred_class=np.argsort(pred_class,axis=1)[:,-n:]
        pred_hangeul=[]
        for idx in range(image.shape[0]):
            pred_hangeul.append([])
            for char in pred_class[idx]:
                pred_hangeul[-1].append(charset[char])
        return pred_hangeul

        
    def predict_split(self,image,n=1):
        #Predict the top-n classes of the image
        #k: top classes for each component to generate
        #Returns top n characters that maximize pred(chosung)*pred(jungsung)*pred(jongsung)
        k=int(n**(1/3))+2
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
    
    def plot_val_image(self,val_data):
        #Load validation data
        val_x,val_y=val_data
        #Predict classes
        indicies=random.sample(range(len(val_x)),10)
        val_x=val_x[indicies]
        pred_y=self.predict(val_x,10)

        fig = plt.figure(figsize=(10,1))
        for idx in range(10):
            plt.subplot(1,10,idx+1)
            plt.imshow(val_x[idx],cmap='gray')
            plt.axis('off')
        plt.savefig('./logs/image.png')
        print(pred_y)
            
    def compile_model(self,lr,opt):
        #Compile model 
        if opt =='sgd':
            optimizer=tf.keras.optimizers.SGD(lr)
        elif opt=='adam':
            optimizer=tf.keras.optimizers.Adam(lr)
        elif opt=='adabound':
            optimizer=AdaBound(lr=lr,final_lr=lr*100,amsbound=False)
        elif opt=='amsbound':
            optimizer=AdaBound(lr=lr,final_lr=lr*100,amsbound=True)
            
        if self.split_components:
            losses = {
                "CHOSUNG": "categorical_crossentropy",
                "JUNGSUNG": "categorical_crossentropy",
                "JONGSUNG": "categorical_crossentropy"}
        else:
            losses="categorical_crossentropy"

        self.model.compile(optimizer=optimizer, loss=losses,metrics=["accuracy"])

    def train(self,epochs=10,lr=0.001,data_path='./data',patch_size=10,batch_size=32,optimizer='adabound'):
        def write_tensorboard(summary_writer,history,step):
             with summary_writer.as_default():
                if self.split_components:
                    tf.summary.scalar('training_loss', history.history['loss'][0], step=step)
                    tf.summary.scalar('CHOSUNG_accuracy', history.history['CHOSUNG_accuracy'][0], step=step)
                    tf.summary.scalar('JUNGSUNG_accuracy', history.history['JUNGSUNG_accuracy'][0], step=step)
                    tf.summary.scalar('JONGSUNG_accuracy', history.history['JONGSUNG_accuracy'][0], step=step)

                    tf.summary.scalar('val_loss', history.history['val_loss'][0], step=step)
                    tf.summary.scalar('val_CHOSUNG_accuracy', history.history['val_CHOSUNG_accuracy'][0], step=step)
                    tf.summary.scalar('val_JUNGSUNG_accuracy', history.history['val_JUNGSUNG_accuracy'][0], step=step)
                    tf.summary.scalar('val_JONGSUNG_accuracy', history.history['val_JONGSUNG_accuracy'][0], step=step)
                else:
                    tf.summary.scalar('training_loss', history.history['loss'][0], step=step)
                    tf.summary.scalar('val_loss', history.history['accuracy'][0], step=step)
                    tf.summary.scalar('training_accuracy', history.history['val_loss'][0], step=step)
                    tf.summary.scalar('val_accuracy', history.history['val_accuracy'][0], step=step)
        

        train_dataset=dataset.DataPickleLoader(split_components=self.split_components,data_path=data_path,patch_size=patch_size)
        val_x,val_y=train_dataset.get_val()

        self.compile_model(lr,optimizer)
        summary_writer = tf.summary.create_file_writer("./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        step=0
        
        for epoch in range(epochs):
            print('Training epoch',epoch)
            self.plot_val_image(val_data=(val_x,val_y))
            epoch_end=False
            while epoch_end==False:
                #Train on loaded dataset batch
                train_x,train_y,epoch_end=train_dataset.get()
                history=self.model.fit(x=train_x,y=train_y,epochs=1,validation_data=(val_x,val_y),batch_size=batch_size)

                #Log losses to Tensorboard
                write_tensorboard(summary_writer,history,step)
                step+=1
                #Clear garbage memory
                tf.keras.backend.clear_session()
                gc.collect()
                clear_output(wait=True)
                
            #Save weights in checkpoint
            self.model.save_weights('./logs/weights.h5')