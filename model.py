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
import shutil

from keras_adabound import AdaBound
import utils.predict_char as predict_char
from utils.model_architectures import VGG16,InceptionResnetV2,MobilenetV3,EfficientCNN
from utils.MelnykNet import melnyk_net
class KoOCR():
    def __init__(self,split_components=True,weight_path='',fc_link='',network_type='melnyk',image_size=96,direct_map=False,refinement_t=4,\
            iterative_refinement=False,data_augmentation=False):
        self.split_components=split_components
        self.iterative_refinement=iterative_refinement
        self.refinement_t=refinement_t
        self.charset=korean_manager.load_charset()

        #Build and load model
        if weight_path:
            self.model = tf.keras.models.load_model(weight_path,compile=False)
        else:
            model_list={'VGG16':VGG16,'inception-resnet':InceptionResnetV2,'mobilenet':MobilenetV3,'efficient-net':EfficientCNN,'melnyk':melnyk_net}
            settings={'split_components':split_components,'input_shape':image_size,'direct_map':direct_map,'fc_link':fc_link,'refinement_t':refinement_t,\
                'iterative_refinement':iterative_refinement,'data_augmentation':data_augmentation}
            self.model=model_list[network_type](settings)
        
        if iterative_refinement:
            self.decoders=self.find_decoders()
    def find_decoders(self):
         return 0
         
    def predict(self,image,n=1):
        if self.split_components:
            return predict_char.predict_split(self.model,image,n)
        else:
            return predict_char.predict_complete(self.model,image,n)
    
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
            optimizer=tf.keras.optimizers.SGD(lr,clipvalue=0.1)
        elif opt=='adam':
            optimizer=tf.keras.optimizers.Adam(lr,clipvalue=0.1)
        elif opt=='adabound':
            optimizer=AdaBound(lr=lr,final_lr=lr*100,clipvalue=0.1)
        
        if self.iterative_refinement:
            losses="categorical_crossentropy"
        elif self.split_components:
            losses = {
                "CHOSUNG": "categorical_crossentropy",
                "JUNGSUNG": "categorical_crossentropy",
                "JONGSUNG": "categorical_crossentropy"}
        else:
            losses="categorical_crossentropy"

        self.model.compile(optimizer=optimizer, loss=losses,metrics=["accuracy"])

    def train(self,epochs=10,lr=0.001,data_path='./data',patch_size=10,batch_size=32,optimizer='adabound',zip_weights=False):
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
        if self.iterative_refinement:
            val_y=[val_y['CHOSUNG'],val_y['JUNGSUNG'],val_y['JONGSUNG']]*self.refinement_t

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
                if self.iterative_refinement:
                    train_y=[train_y['CHOSUNG'],train_y['JUNGSUNG'],train_y['JONGSUNG']]*self.refinement_t
                history=self.model.fit(x=train_x,y=train_y,epochs=1,validation_data=(val_x,val_y),batch_size=batch_size)

                #Log losses to Tensorboard
                write_tensorboard(summary_writer,history,step)
                step+=1
                #Clear garbage memory
                tf.keras.backend.clear_session()
                gc.collect()
                clear_output(wait=True)
                
            #Save weights in checkpoint
            self.model.save('./logs/weights', save_format='tf')
            if zip_weights:
                shutil.make_archive('weights_epoch_'+str(epoch), 'zip', './logs/weights')