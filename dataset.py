import numpy as np
import os
import tensorflow as tf
import fnmatch
import _pickle as pickle    #cPickle
class KoreanManager():
  def __init__(self):
    self.CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', ' ']
    # 중성 리스트. 00 ~ 20
    self.JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ', ' ']
    # 종성 리스트. 00 ~ 27 + 1(1개 없음)
    self.JONGSUNG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
  def korean_split_numpy(self,words,to_text=False):
    # 한글 글자의 np array를 입력받아 초성, 중성, 종성을 각각의 array로 내보내는 함수
    cho,jung,jong=[],[],[]
    for w in words:
      ch1 = (ord(w) - ord('가'))//588
      ch2 = ((ord(w) - ord('가')) - (588*ch1)) // 28
      ch3 = (ord(w) - ord('가')) - (588*ch1) - 28*ch2
      
      #ㄱ,ㄴ,ㄷ,... 처리 
      if ch1==-54:
        ch1,ch2,ch3=19,22,ord(w) - ord('ㄱ')+1

      if to_text:
        cho.append(self.CHOSUNG_LIST[ch1])
        jung.append(self.JUNGSUNG_LIST[ch2])
        jong.append(self.JONGSUNG_LIST[ch3])
      else:
        cho.append(ch1)
        jung.append(ch2)
        jong.append(ch3)
    return np.array(cho),np.array(jung),np.array(jong)

class DataPickleLoader():
    #Load data patch by patch
    def __init__(self,patch_size=10,data_path='./data',split_components=True):
        self.data_path=data_path
        self.patch_size=patch_size
        self.split_components=split_components

        self.korean=KoreanManager()

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
        images,labels=data['image']

        if self.split_components==True:
            cho,jung,jong=self.korean.korean_split_numpy(data['label'])
        else:
            labels=data['label']

        path_slice=self.file_list[self.current_idx+1:next_idx]
        for pkl in path_slice:
            data=load_pickle(os.path.join(self.data_path,pkl))
            images=np.concatenate((images,data['image']),axis=0)

            if self.split_components==True:
                cho_,jung_,jong_=self.korean.korean_split_numpy(data['label'])
                
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
            return tf.data.Dataset.from_tensor_slices(images), tf.data.Dataset.from_tensor_slices({'CHOSUNG':cho,'JUNGSUNG':jung,'JONGSUNG':jong}),did_reset
        else:
            return tf.data.Dataset.from_tensor_slices(images), tf.data.Dataset.from_tensor_slices(labels),did_reset

    def mix_indicies(self):
        np.random.shuffle(self.file_list)