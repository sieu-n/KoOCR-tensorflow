import numpy as np
import os
import tensorflow as tf
import fnmatch
import _pickle as pickle    #cPickle
import korean_manager
class DataPickleLoader():
    #Load data patch by patch
    def __init__(self,patch_size=10,data_path='./data',split_components=True):
        self.data_path=data_path
        self.patch_size=patch_size
        self.split_components=split_components

        self.current_idx=0
        self.file_list=fnmatch.filter(os.listdir(data_path), '*.pickle')

        self.mix_indicies()

    def get(self):
        #Function returning next patch
        def load_pickle(path):
            with open(path,'rb') as handle:
                data=pickle.load(handle)
            return data
        #Check if end of list
        did_reset=False
        next_idx=self.current_idx+self.patch_size
        if next_idx>len(self.file_list):
            next_idx=len(self.file_list)
            did_reset=True

        data=load_pickle(os.path.join(self.data_path,self.file_list[self.current_idx]))
        images=data['image']

        if self.split_components==True:
            cho,jung,jong=korean_manager.korean_split_numpy(data['label'])
        else:
            labels=data['label']

        path_slice=self.file_list[self.current_idx+1:next_idx]
        for pkl in path_slice:
            data=load_pickle(os.path.join(self.data_path,pkl))
            images=np.concatenate((images,data['image']),axis=0)

            if self.split_components==True:
                cho_,jung_,jong_=korean_manager.korean_split_numpy(data['label'])
                
                cho=np.concatenate((cho,cho_))
                jung=np.concatenate((jung,jung_))
                jong=np.concatenate((jong,jong_))
            else:
                labels=np.concatenate((labels,data['label']))
            

        #Reset if final chunk of image
        self.current_idx=next_idx
        if self.current_idx==len(self.file_list):
            self.mix_indicies()
            self.current_idx=0
        
        if self.split_components==True:
            return tf.data.Dataset.from_tensor_slices({'input_image':images,'CHOSUNG':cho,'JUNGSUNG':jung,'JONGSUNG':jong}),did_reset
        else:
            return tf.data.Dataset.from_tensor_slices({'input_image':images,'output':labels}),did_reset

    def mix_indicies(self):
        np.random.shuffle(self.file_list)