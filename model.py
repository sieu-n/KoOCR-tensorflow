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
from tqdm import tqdm
from keras_adabound import AdaBound
import utils.predict_char as predict_char
from utils.model_architectures import VGG16,InceptionResnetV2,MobilenetV3,EfficientCNN
from utils.MelnykNet import melnyk_net
class KoOCR():
    def __init__(self,split_components=True,weight_path='',fc_link='',network_type='melnyk',image_size=96,direct_map=False,refinement_t=4,\
            iterative_refinement=False,data_augmentation=False,adversarial_learning=False):
        self.split_components=split_components
        self.iterative_refinement=iterative_refinement
        self.refinement_t=refinement_t
        self.charset=korean_manager.load_charset()
        self.adversarial_learning=adversarial_learning
        #Build and load model
        if weight_path:
            self.model = tf.keras.models.load_model(weight_path,compile=False)
        else:
            model_list={'VGG16':VGG16,'inception-resnet':InceptionResnetV2,'mobilenet':MobilenetV3,'efficient-net':EfficientCNN,'melnyk':melnyk_net}
            settings={'split_components':split_components,'input_shape':image_size,'direct_map':direct_map,'fc_link':fc_link,'refinement_t':refinement_t,\
                'iterative_refinement':iterative_refinement,'data_augmentation':data_augmentation,'adversarial_learning':adversarial_learning}
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

    def compile_adversarial_model(self,lr,opt,adversarial_ratio=0):
        #build adversarial model for training
        input_image=self.model.input

        self.model.trainable=False

        feature_map=self.model.get_layer('disc_start')
        disc_output=self.model.get_layer('DISC')

        feature_map.trainable=True
        disc_output.trainable=True

        self.discriminator=tf.keras.models.Model(self.model.input,disc_output.output)

        lr=lr*adversarial_ratio*3
        if opt =='sgd':
            optimizer=tf.keras.optimizers.SGD(lr)
        elif opt=='adam':
            optimizer=tf.keras.optimizers.Adam(lr)
        elif opt=='adabound':
            optimizer=AdaBound(lr=lr,final_lr=lr*100)

        self.discriminator.compile(optimizer=optimizer,loss='binary_crossentropy')

    def compile_model(self,lr,opt,adversarial_ratio=0):
        def inverse_bce(y_true,y_pred):
            y_true=y_true*-1+1
            return tf.keras.losses.binary_crossentropy(y_true,y_pred)

        #Compile model 
        if opt =='sgd':
            optimizer=tf.keras.optimizers.SGD(lr)
        elif opt=='adam':
            optimizer=tf.keras.optimizers.Adam(lr)
        elif opt=='adabound':
            optimizer=AdaBound(lr=lr,final_lr=lr*100)
        
        if self.iterative_refinement:
            losses="sparse_categorical_crossentropy"
        elif self.split_components:
            if self.adversarial_learning:
                losses = {
                    "CHOSUNG": "sparse_categorical_crossentropy",
                    "JUNGSUNG": "sparse_categorical_crossentropy",
                    "JONGSUNG": "sparse_categorical_crossentropy",
                    'DISC':inverse_bce}
                lossWeights = {"CHOSUNG": 1.0-adversarial_ratio, "JUNGSUNG": 1.0-adversarial_ratio,
                    "JONGSUNG":1.0-adversarial_ratio,"DISC":3*adversarial_ratio}
            else:
                losses = {
                    "CHOSUNG": "sparse_categorical_crossentropy",
                    "JUNGSUNG": "sparse_categorical_crossentropy",
                    "JONGSUNG": "sparse_categorical_crossentropy"}
                lossWeights = {"CHOSUNG": 1.0, "JUNGSUNG": 1.0,"JONGSUNG":1.0}
        else:
            losses="sparse_categorical_crossentropy"
            lossWeights=None

        self.model.compile(optimizer=optimizer, loss=losses,metrics=["accuracy"],loss_weights=lossWeights)
        
    def fit_adversarial(self,train_x,train_y,val_x,val_y,batch_size):
        train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(batch_size)
        pbar=tqdm(train_dataset)
        for image,label in pbar:
            out=tself.model.train_on_batch(image,label)
            self.discriminator.train_on_batch(image,label['DISC'])g
            pbar.set_description("Loss:",out[:len(out)//2],"  Accuracy:",out[len(out)//2:])
        results = model.evaluate(x_test, y_test, batch_size=128)
        print("Results:", results)

    def train(self,epochs=10,lr=0.001,data_path='./data',patch_size=10,batch_size=32,optimizer='adabound',zip_weights=False,
            adversarial_ratio=0.15):
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
        

        train_dataset=dataset.DataPickleLoader(split_components=self.split_components,data_path=data_path,patch_size=patch_size,
            return_image_type=self.adversarial_learning)
        val_x,val_y=train_dataset.get_val()
        if self.iterative_refinement:
            val_y=[val_y['CHOSUNG'],val_y['JUNGSUNG'],val_y['JONGSUNG']]*self.refinement_t

        self.compile_model(lr,optimizer,adversarial_ratio)
        if self.adversarial_learning:
            self.compile_adversarial_model(lr,optimizer,adversarial_ratio)

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

                if self.adversarial_learning:
                    history=self.fit_adversarial(train_x,train_y,val_x,val_y,batch_size)
                else:
                    history=self.model.fit(x=train_x,y=train_y,epochs=1,validation_data=(val_x,val_y),batch_size=batch_size)
                #Log losses to Tensorboard
                write_tensorboard(summary_writer,history,step)
                step+=1
                #Clear garbage memory
                tf.keras.backend.clear_session()
                gc.collect()
                
            #Save weights in checkpoint
            self.model.save('./logs/weights', save_format='tf')
            if zip_weights:
                shutil.make_archive('weights_epoch_'+str(epoch), 'zip', './logs/weights')