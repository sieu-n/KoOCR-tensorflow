import numpy as np
import os
import tensorflow as tf
import fnmatch
import _pickle as pickle    #cPickle
import utils.korean_manager as korean_manager
import random
import progressbar

class DataPickleLoader():
    #Load data patch by patch
    def __init__(self,patch_size=10,data_path='./data',val_data_path='./val_data',split_components=True,val_data=.1,
            return_image_type=False):
        self.data_path=data_path
        self.val_data_path=val_data_path
        self.patch_size=patch_size
        self.split_components=split_components
        self.return_image_type=return_image_type
        self.current_idx=0

        file_list=fnmatch.filter(os.listdir(data_path), '*.pickle')
        val_file_list=fnmatch.filter(os.listdir(val_data_path), '*.pickle')

        self.file_list=file_list
        self.val_file_list=val_file_list
        np.random.shuffle(self.val_file_list)
        self.mix_indicies()

    def load_pickle(self,path):
        with open(path,'rb') as handle:
            data=pickle.load(handle)
        return data

    def get_val(self,prob=0.3):
        data=self.load_pickle(os.path.join(self.val_data_path,self.val_file_list[0]))
        images=data['image']

        if self.split_components==True:
            cho,jung,jong=korean_manager.korean_split_numpy(data['label'])
        else:
            labels=korean_manager.korean_numpy(data['label'])
        types=np.repeat(int(self.val_file_list[0].split('_')=='handwritten'),images.shape[0])

        for pkl in self.val_file_list[1:int(len(self.val_file_list)*prob)]:
            data=self.load_pickle(os.path.join(self.val_data_path,pkl))
            images=np.concatenate((images,data['image']),axis=0)

            if self.split_components==True:
                cho_,jung_,jong_=korean_manager.korean_split_numpy(data['label'])
                
                cho=np.concatenate((cho,cho_))
                jung=np.concatenate((jung,jung_))
                jong=np.concatenate((jong,jong_))
                types=np.concatenate((types, np.repeat(int(pkl.split('_')=='handwritten'),data['image'].shape[0])))
            else:
                labels=np.concatenate((labels,korean_manager.korean_numpy(data['label'])))
        
        #Random shuffle data
        ind_list = list(range(types.shape[0]))
        random.shuffle(ind_list)

        if self.split_components==True:
            #One hot encode labels and return
            #cho=tf.one_hot(cho,len(korean_manager.CHOSUNG_LIST))
            #jung=tf.one_hot(jung,len(korean_manager.JUNGSUNG_LIST))
            #jong=tf.one_hot(jong,len(korean_manager.JONGSUNG_LIST))
            
            if self.return_image_type:
                return images,{'CHOSUNG':cho[ind_list],'JUNGSUNG':jung[ind_list],'JONGSUNG':jong[ind_list],'DISC':types[ind_list]}
            else:
                return images,{'CHOSUNG':cho[ind_list],'JUNGSUNG':jung[ind_list],'JONGSUNG':jong[ind_list]}
        else:
            return images,labels[ind_list]

    def get(self):
        print(f"Loading dataset patch {self.current_idx//self.patch_size}/{len(self.file_list)//self.patch_size+1}...")
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
        types=np.repeat(int(self.file_list[self.current_idx].split('_')=='handwritten'),images.shape[0])

        path_slice=self.file_list[self.current_idx+1:next_idx]
        for pkl in progressbar.progressbar(path_slice):
            data=self.load_pickle(os.path.join(self.data_path,pkl))
            images=np.concatenate((images,data['image']),axis=0)

            if self.split_components==True:
                cho_,jung_,jong_=korean_manager.korean_split_numpy(data['label'])
                
                cho=np.concatenate((cho,cho_))
                jung=np.concatenate((jung,jung_))
                jong=np.concatenate((jong,jong_))
                types=np.concatenate((types, np.repeat(int(pkl.split('_')=='handwritten'),data['image'].shape[0])))
            else:
                labels=np.concatenate((labels,korean_manager.korean_numpy(data['label'])))
            

        #Reset if final chunk of image
        self.current_idx=next_idx
        if self.current_idx==len(self.file_list):
            self.mix_indicies()
            self.current_idx=0
            
        ind_list = list(range(types.shape[0]))
        random.shuffle(ind_list)
        if self.split_components==True:
            #One hot encode labels and return
            if self.return_image_type:
                return images,{'CHOSUNG':cho[ind_list],'JUNGSUNG':jung[ind_list],'JONGSUNG':jong[ind_list],'DISC':types[ind_list]},did_reset
            else:
                return images,{'CHOSUNG':cho[ind_list],'JUNGSUNG':jung[ind_list],'JONGSUNG':jong[ind_list]},did_reset
        else:
            labels=tf.one_hot(labels,len(korean_manager.load_charset()))
            return images,labels,did_reset

    def mix_indicies(self):
        np.random.shuffle(self.file_list)