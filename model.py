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
import wandb
import datetime
import shutil
from tqdm import tqdm
from keras_adabound import AdaBound
import utils.predict_char as predict_char
from utils.model_architectures import VGG16,InceptionResnetV2,MobilenetV3,EfficientCNN,AFL_Model
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
            model_list={'VGG16':VGG16,'inception-resnet':InceptionResnetV2,'mobilenet':MobilenetV3,'efficient-net':EfficientCNN,'melnyk':melnyk_net,
                'afl':AFL_Model}
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
        disc_output=self.model.get_layer('DISC')

        self.discriminator=tf.keras.models.Model(self.model.input,disc_output.output)

        for l in self.model.layers:
            l.trainable=False
            
        self.model.get_layer('disc_start').trainable=True
        self.model.get_layer('DISC').trainable=True

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
            losses="categorical_crossentropy"
        elif self.split_components:
            if self.adversarial_learning:
                losses = {
                    "CHOSUNG": "categorical_crossentropy",
                    "JUNGSUNG": "categorical_crossentropy",
                    "JONGSUNG": "categorical_crossentropy",
                    'DISC':inverse_bce}
                if self.fit_discriminator:
                    lossWeights = {"CHOSUNG": 1.0-adversarial_ratio, "JUNGSUNG": 1.0-adversarial_ratio,
                        "JONGSUNG":1.0-adversarial_ratio,"DISC":3*adversarial_ratio}
                else:
                    lossWeights = {"CHOSUNG": 1.0, "JUNGSUNG": 1.0,"JONGSUNG":1.0,"DISC":0}
            else:
                losses = {
                    "CHOSUNG": "categorical_crossentropy",
                    "JUNGSUNG": "categorical_crossentropy",
                    "JONGSUNG": "categorical_crossentropy"}
                lossWeights = {"CHOSUNG": 1.0, "JUNGSUNG": 1.0,"JONGSUNG":1.0}
        else:
            losses="categorical_crossentropy"
            lossWeights=None
            
        if self.adversarial_learning:
            self.model.trainable=True
            self.model.get_layer('disc_start').trainable=False
            self.model.get_layer('DISC').trainable=False
            
        self.model.compile(optimizer=optimizer, loss=losses,metrics=["accuracy"],loss_weights=lossWeights)

    def fit_adversarial(self,train_x,train_y,val_x,val_y,batch_size):
        train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(batch_size)
        loss_arr,total_p=0,0
        
        if self.verbose==1:
            pbar=tqdm(train_dataset)
        else:
            pbar=train_dataset
        for image,label in pbar:
            out=self.model.train_on_batch(image,label)
            loss_arr+=np.array(out)
            total_p+=1
            
            if self.fit_discriminator:
                self.discriminator.train_on_batch(image,label['DISC'])
        results = self.model.evaluate(val_x, val_y, batch_size=128,verbose=self.verbose)
        print("Training L:", list(loss_arr/total_p))

        loss_dict = {name+'_loss': pred for name, pred in zip(self.model.output_names, out[:len(out)//2])}
        acc_dict = {name+'_accuracy': pred for name, pred in zip(self.model.output_names, out[len(out)//2:])}
        z = {**loss_dict, **acc_dict}
        return z

    def train(self,epochs=10,lr=0.001,data_path='./data',patch_size=10,batch_size=32,optimizer='adabound',zip_weights=False,
            adversarial_ratio=0.15,log_tensorboard=True,log_wandb=False,setup_wandb=False,fit_discriminator=True,silent_mode=False):
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
        
        def setup_wandboard():
            wandb.init(project="KoOCR", config={
                'AFL': self.adversarial_learning,
                'Iterative Refinement': self.iteratve_refinement,
                "optiminzer": optimizer,
                "batch_size": batch_size,
                'learning_rate':lr,
                'AFL ratio':adversarial_ratio,
                'Split components':self.split_components
            })
        def write_wandb(history):
            wandb.log(history)
        train_dataset=dataset.DataPickleLoader(split_components=self.split_components,data_path=data_path,patch_size=patch_size,
            return_image_type=self.adversarial_learning,silent_mode=silent_mode)
        val_x,val_y=train_dataset.get_val()
        if self.iterative_refinement:
            val_y=[val_y['CHOSUNG'],val_y['JUNGSUNG'],val_y['JONGSUNG']]*self.refinement_t
        self.fit_discriminator=fit_discriminator
        self.compile_model(lr,optimizer,adversarial_ratio)
        if self.adversarial_learning:
            self.compile_adversarial_model(lr,optimizer,adversarial_ratio)

        summary_writer = tf.summary.create_file_writer("./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        if setup_wandb:
            setup_wandboard()
        step=0
        
        if silent_mode:
            self.verbose=2
        else:
            self.verbose=1

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
                    history=self.model.fit(x=train_x,y=train_y,epochs=1,validation_data=(val_x,val_y),batch_size=batch_size,verbose=self.verbose)
                #Log losses to Tensorboard
                if log_tensorboard:
                    write_tensorboard(summary_writer,history,step)
                if log_wandb:
                    write_wandb(history)
                step+=1
                #Clear garbage memory
                tf.keras.backend.clear_session()
                gc.collect()
                
            #Save weights in checkpoint
            self.model.save('./logs/weights', save_format='tf')
            if zip_weights:
                shutil.make_archive('weights_epoch_'+str(epoch), 'zip', './logs/weights')