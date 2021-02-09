import model
import dataset
import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import fnmatch
import utils.korean_manager as korean_manager
import progressbar
import _pickle as pickle    #cPickle
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
#bool type for arguments
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
#Define arguments
parser = argparse.ArgumentParser(description='Download dataset')
parser.add_argument("--data_path", type=str,default='./val_data')
parser.add_argument("--image_size", type=int,default=256)
parser.add_argument("--split_components", type=str2bool,default=True)
parser.add_argument("--patch_size", type=int,default=10)
parser.add_argument("--confusion_matrix", type=str2bool,default=True)

parser.add_argument("--weights", type=str,default='')
parser.add_argument("--top_n", type=int,default=5)

def generate_confusion_matrix(model,key_text):
    #Generate confusion matrix of each component based on sklearn, 
    #Only call when split_components is True.
    if not args.split_components:
        return
    types=['CHOSUNG','JUNGSUNG','JONGSUNG']
    
    confusion_list={'CHOSUNG':0,'JUNGSUNG':0,'JONGSUNG':0}

    file_list=fnmatch.filter(os.listdir(args.data_path), f'{key_text}*.pickle')
    for x in progressbar.progressbar(file_list):
        #Read pickle
        path=os.path.join(args.data_path,x)
        with open(path,'rb') as handle:
            data=pickle.load(handle)
        #Predict image 
        pred=model.model.predict(data['image'])
        cho,jung,jong=korean_manager.korean_split_numpy(data['label'])
        truth_label={'CHOSUNG':cho, 'JUNGSUNG':jung,'JONGSUNG':jong}
        #Add to confusion_matrix
        for t in types:
            confusion_list[t]=confusion_matrix(truth_label[t],pred[t])+confusion_list[t]
    
    #Plot confusion matrix using seaborn
    index_list={'CHOSUNG':korean_manager.CHOSUNG_LIST,'JUNGSUNG':korean_manager.JUNGSUNG_LIST,'JONGSUNG':korean_manager.JONGSUNG_LIST}
    for t in types:
        df_cm = pd.DataFrame(confusion_list[t], index=index_list[t], columns=index_list[t])
        ax = sn.heatmap(df_cm, cmap='Oranges', annot=True)
        ax.savefig(os.path.join('./logs',key_text+"_heatmap.png"))
        
def evaluate(model,key_text):
    #Evaluate top-n accuracy
    correct_num,total_num=0,0
    file_list=fnmatch.filter(os.listdir(args.data_path), f'{key_text}*.pickle')
    for x in progressbar.progressbar(file_list):
        #Read pickle
        path=os.path.join(args.data_path,x)
        with open(path,'rb') as handle:
            data=pickle.load(handle)
        #Predict image 
        pred=model.predict(data['image'],n=args.top_n)
        #Compare/calculate acc
        for idx,pred_ in enumerate(pred):
            total_num+=1
            correct_num+=data['label'][idx] in pred_
    return 100*correct_num/total_num

if __name__=='__main__':
    args = parser.parse_args()

    KoOCR=model.KoOCR(split_components=args.split_components,weight_path=args.weights)

    acc=evaluate(KoOCR,'handwritten')
    print('Handwritten OCR Accuracy:',acc)
    if args.confusion_matrix:
        generate_confusion_matrix(KoOCR,'handwritten')

    acc=evaluate(KoOCR,'printed')
    print('Printed OCR Accuracy:',acc)
    if args.confusion_matrix:
        generate_confusion_matrix(KoOCR,'printed')