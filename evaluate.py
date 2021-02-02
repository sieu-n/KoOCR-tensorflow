import model
import dataset
import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import fnmatch
import progressbar
import _pickle as pickle    #cPickle

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

parser.add_argument("--weights", type=str,default='')
parser.add_argument("--top_n", type=int,default=5)

def evaluate(model,key_text):
    #Evaluate data from specific directory
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
            
if __name__=='__main__':
    args = parser.parse_args()

    KoOCR=model.KoOCR(split_components=args.split_components,weight_path=args.weights)

    acc=evaluate(KoOCR,'handwritten')
    print('Handwritten OCR Accuracy:',acc)
    acc=evaluate(KoOCR,'printed')