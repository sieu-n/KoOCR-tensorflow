import numpy as np
import os
import tensorflow as tf
import fnmatch
import _pickle as pickle    #cPickle
import korean_manager
import random
import progressbar
class DataPickleLoader():
    #Load data patch by patch
    def __init__(self,patch_size=10,data_path='./data',split_components=True,val_data=.1):
        random.seed(42)
        self.data_path=data_path
        self.patch_size=patch_size
        self.split_components=split_components

        self.current_idx=0

        file_list=fnmatch.filter(os.listdir(data_path), '*.pickle')

        train_test_split=int(val_data*len(file_list))
        random.shuffle(file_list)
        self.file_list=file_list[train_test_split:]
        self.val_file_list=file_list[:train_test_split]

        self.mix_indicies()

    def load_pickle(self,path):
        with open(path,'rb') as handle:
            data=pickle.load(handle)
        return data

    def get_val(self):
        data=self.load_pickle(os.path.join(self.data_path,self.val_file_list[0]))
        images=data['image']

        if self.split_components==True:
            cho,jung,jong=korean_manager.korean_split_numpy(data['label'])
        else:
            labels=korean_manager.korean_numpy(data['label'])
        
        for pkl in self.val_file_list[1:]:
            data=self.load_pickle(os.path.join(self.data_path,pkl))
            images=np.concatenate((images,data['image']),axis=0)

            if self.split_components==True:
                cho_,jung_,jong_=korean_manager.korean_split_numpy(data['label'])
                
                cho=np.concatenate((cho,cho_))
                jung=np.concatenate((jung,jung_))
                jong=np.concatenate((jong,jong_))
            else:
                labels=np.concatenate((labels,korean_manager.korean_numpy(data['label'])))
        if self.split_components==True:
            #One hot encode labels and return
            cho=tf.one_hot(cho,len(korean_manager.CHOSUNG_LIST))
            jung=tf.one_hot(jung,len(korean_manager.JUNGSUNG_LIST))
            jong=tf.one_hot(jong,len(korean_manager.JONGSUNG_LIST))
            return images,{'CHOSUNG':cho,'JUNGSUNG':jung,'JONGSUNG':jong}
        else:
            labels=tf.one_hot(labels,len(korean_manager.load_charset()))
            return images,labels

    def get(self):
        print("Loading dataset patch...")
        #Check if end of list
        did_reset=False
        next_idx=self.current_idx+self.patch_size
        if next_idx>len(self.file_list):
            next_idx=len(self.file_list)
            did_reset=True

        data=self.load_pickle(os.path.join(self.data_path,self.file_list[self.current_idx]))
        images=data['image']

        if self.split_components==True:
            cho,jung,jong=korean_manager.korean_split_numpy(data['label'])
        else:
            labels=korean_manager.korean_numpy(data['label'])

        path_slice=self.file_list[self.current_idx+1:next_idx]
        for pkl in progressbar.progressbar(path_slice):
            data=self.load_pickle(os.path.join(self.data_path,pkl))
            images=np.concatenate((images,data['image']),axis=0)

            if self.split_components==True:
                cho_,jung_,jong_=korean_manager.korean_split_numpy(data['label'])
                
                cho=np.concatenate((cho,cho_))
                jung=np.concatenate((jung,jung_))
                jong=np.concatenate((jong,jong_))
            else:
                labels=np.concatenate((labels,korean_manager.korean_numpy(data['label'])))
            

        #Reset if final chunk of image
        self.current_idx=next_idx
        if self.current_idx==len(self.file_list):
            self.mix_indicies()
            self.current_idx=0
        
        if self.split_components==True:
            #One hot encode labels and return
            cho=tf.one_hot(cho,len(korean_manager.CHOSUNG_LIST))
            jung=tf.one_hot(jung,len(korean_manager.JUNGSUNG_LIST))
            jong=tf.one_hot(jong,len(korean_manager.JONGSUNG_LIST))
            return images,{'CHOSUNG':cho,'JUNGSUNG':jung,'JONGSUNG':jong},did_reset
        else:
            labels=tf.one_hot(labels,len(korean_manager.load_charset()))
            return images,labels,did_reset

    def mix_indicies(self):
        np.random.shuffle(self.file_list)