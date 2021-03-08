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
from utils.CustomLayers import MultiOutputGradCAM
from utils.model_components import PreprocessingPipeline
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.font_manager as font_manager
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
parser=argparse.ArgumentParser(description='Download dataset')
parser.add_argument("--data_path", type=str,default='./val_data')
parser.add_argument("--train_data_path", type=str,default='./data')
parser.add_argument("--image_size", type=int,default=256)
parser.add_argument("--split_components", type=str2bool,default=True)
parser.add_argument("--patch_size", type=int,default=10)

parser.add_argument("--show_augmentation", type=str2bool,default=False)
parser.add_argument("--accuracy", type=str2bool,default=False)
parser.add_argument("--confusion_matrix", type=str2bool,default=False)
parser.add_argument("--class_activation", type=str2bool,default=False)

parser.add_argument("--class_activation_n", type=int,default=10)

parser.add_argument("--weights", type=str,default='')
parser.add_argument("--top_n", type=int,default=5)

def generate_CAM(model,key_text):
    #Pick one pickle, and load 10 data from file
    file_list=fnmatch.filter(os.listdir(args.data_path), f'{key_text}*.pickle')
    np.random.shuffle(file_list)
    with open(os.path.join(args.data_path,file_list[0]),'rb') as handle:
        data=pickle.load(handle)
    indicies=np.random.choice(len(data['image']), args.class_activation_n, replace=False)
    
    fig = plt.figure(figsize=(args.class_activation_n*2,8))
    for idx,data_idx in enumerate(indicies):
        #Get Class Activation Map
        cam=MultiOutputGradCAM(model.model,data['label'][data_idx])
        heatmap_list=cam.compute_heatmap(data['image'][data_idx])
        heatmap_list.append(None)

        for comp in range(4):
            plt.subplot(4,args.class_activation_n,idx+1+args.class_activation_n*comp)
            plt.imshow(data['image'][data_idx],cmap='gray')
            if comp!=3:
                plt.imshow(heatmap_list[comp],alpha=0.5,cmap='jet')
            plt.axis('off')
    plt.savefig(f'./logs/CAM_{key_text}.png')
    plt.clf()

def plot_augmentation():
    images_per_type=3
    augment_times=5
    key_texts=['clova','handwritten','printed']
    width,height=images_per_type*3,augment_times+1
    #Define PreprocessingPipeline with data augmentation.
    aug_model=PreprocessingPipeline(False,True)

    fig = plt.figure(figsize=(width,height))
    for key_text_idx in range(3):
        #Read data from randomly selected batch
        key_text=key_texts[key_text_idx]
        file_list=fnmatch.filter(os.listdir(args.train_data_path), f'{key_text}*.pickle')
        np.random.shuffle(file_list)
        with open(os.path.join(args.train_data_path,file_list[0]),'rb') as handle:
            data=pickle.load(handle)
        #Plot first 3 images and augment_times augmented versions. 
        for idx in range(images_per_type):
            plt.subplot(key_text_idx*3+idx+1,width,height)
            plt.imshow(data['image'][idx])
            plt.axis('off')
            for k in range(1,augment_times+1):
                plt.subplot(key_text_idx*3+idx+1 + width*k,width,height)
                new_image=aug_model(data['image'][idx].reshape(1,96,96,1))
                plt.imshow(new_image[0])
                plt.axis('off')
    plt.savefig('./logs/Augmentation_Sample.png')
    plt.clf()

def generate_confusion_matrix(model,key_text):
    #Generate confusion matrix of each component based on sklearn, 
    #Only call when split_components is True.
    if not args.split_components: 
        return
    try:
        path = './files/batang.ttf'
        prop = font_manager.FontProperties(fname=path)
    except:
        print('No font in location, using default font.')
        prop=None
    types=['CHOSUNG','JUNGSUNG','JONGSUNG']
    
    confusion_list={'CHOSUNG':0,'JUNGSUNG':0,'JONGSUNG':0}
    index_list={'CHOSUNG':korean_manager.CHOSUNG_LIST,'JUNGSUNG':korean_manager.JUNGSUNG_LIST,'JONGSUNG':korean_manager.JONGSUNG_LIST}
    file_list=fnmatch.filter(os.listdir(args.data_path), f'{key_text}*.pickle')
    for x in progressbar.progressbar(file_list):
        #Read pickle
        path=os.path.join(args.data_path,x)
        with open(path,'rb') as handle:
            data=pickle.load(handle)
        #Predict image 
        pred=model.model.predict(data['image'])
        pred_dict={}
        for idx,t in enumerate(types):
            pred_dict[t]=np.argmax(pred[idx],axis=1)

        cho,jung,jong=korean_manager.korean_split_numpy(data['label'])
        truth_label={'CHOSUNG':cho, 'JUNGSUNG':jung,'JONGSUNG':jong}
        #Add to confusion_matrix
        for t in types:
            labels=range(len(index_list[t]))
            confusion_list[t]=confusion_matrix(truth_label[t],pred_dict[t],labels=labels)+confusion_list[t]
    
    #Plot confusion matrix using seaborn
    
    for t in types:
        df_cm = pd.DataFrame(confusion_list[t], index=index_list[t], columns=index_list[t])

        sn.set(rc={'figure.figsize':(20,18)})
        ax = sn.heatmap(df_cm, cmap='Oranges', annot=True, fontproperties=prop)
        fig = ax.get_figure()
        fig.savefig(os.path.join('./logs','confusion_matrix_'+key_text+'_'+t+".png"))
        plt.clf()
        
def evaluate(model,key_text,plot_wrong=True):
    #Evaluate top-n accuracy
    correct_num,total_num=0,0
    file_list=fnmatch.filter(os.listdir(args.data_path), f'{key_text}*.pickle')

    wrong_list=[]
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
            if data['label'][idx] in pred_:
                correct_num+=1
            else:
                wrong_list.append((data['image'][idx],data['label'][idx]))

    if plot_wrong:
        fig = plt.figure(figsize=(10,10))
        for x in range(100):
            plt.subplot(x,10,10)
            plt.imshow(wrong_list[x][0])
            plt.xlabel('Pred:'+str(wrong_list[x][1]))
        plt.savefig(f'./logs/{key_text}_Wrong_examples.png')
        plt.clf()
    return 100*correct_num/total_num

if __name__=='__main__':
    args = parser.parse_args()
    plt.rc('font', family='NanumBarunGothic') 
    
    KoOCR=model.KoOCR(split_components=args.split_components,weight_path=args.weights)

    if args.accuracy:
        acc=evaluate(KoOCR,'handwritten')
        print('Handwritten OCR Accuracy:',acc)
        acc=evaluate(KoOCR,'printed')
        print('Printed OCR Accuracy:',acc)

    if args.confusion_matrix:
        generate_confusion_matrix(KoOCR,'handwritten')
        print('Handwritten confusion matrix generated.')
        generate_confusion_matrix(KoOCR,'printed')
        print('Printed confusion matrix generated.')

    if args.class_activation:
        generate_CAM(KoOCR,'handwritten')
        print('Handwritten CAM image generated.')
        generate_CAM(KoOCR,'printed')
        print('Printed CAM image generated.')

    if args.show_augmentation:
        plot_augmentation()