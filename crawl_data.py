import os
import urllib.request
import argparse
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import random
import PIL
import json
import _pickle as pickle    #cPickle
import progressbar
import threading
import collections
import tensorflow as tf
import utils.korean_manager as korean_manager
from google_drive_downloader import GoogleDriveDownloader as gdd
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
parser.add_argument("--font_path", type=str,default='./fonts')
parser.add_argument("--AIHub_path", type=str,default='./AIhub')
parser.add_argument("--pickle_path", type=str,default='./data')
parser.add_argument("--val_path", type=str,default='./val_data')
parser.add_argument("--pickle_path_val", type=str,default='./data')
parser.add_argument("--val_ratio", type=float,default=.1)

parser.add_argument("--clova", type=str2bool,default=False)
parser.add_argument("--image_test", type=str2bool,default=False)
parser.add_argument("--image_size", type=int,default=96)
parser.add_argument("--x_offset", type=int,default=50)
parser.add_argument("--y_offset", type=int,default=10)
parser.add_argument("--char_size", type=int,default=200)

parser.add_argument("--AIHub", type=str2bool,default=False)
parser.add_argument("--pickle_size", type=int,default=5000)

def crawl_dataset():
  if args.clova:
    crawl_clova_fonts()
    charset=korean_manager.load_charset()
    convert_all_fonts(charset)
  if args.AIHub:
    download_AIHub_GoogleDrive()
    pickle_AIHub_images()

def pickle_AIHub_images():
    #Pickle AIHub data into handwritten, printed
    if os.path.isdir(args.pickle_path)==False:
        os.mkdir(args.pickle_path)
    if os.path.isdir(args.val_path)==False:
        os.mkdir(args.val_path)
    random.seed(42)
    #Unpickle Handwritten files
    f = open(os.path.join(args.AIHub_path,'handwritten_label.json')) 
    anno = json.load(f) 
    f.close() 
    random.shuffle(anno['annotations'])

    images_before_pickle=args.pickle_size
    pickle_idx=0
    image_arr,label_arr=[],[]

    for x in progressbar.progressbar(anno['annotations']):
        #Save data split into pickle
        if images_before_pickle==0:
            images_before_pickle=args.pickle_size
            #Split train and test data
            if random.random()>args.val_ratio:
                file_path=args.pickle_path
            else:
                file_path=args.val_path

            with open(os.path.join(file_path,'handwritten_'+str(pickle_idx)+'.pickle'),'wb') as handle:
                    pickle.dump({'image':np.array(image_arr),'label':np.array(label_arr)},handle)
            pickle_idx+=1
            image_arr,label_arr=[],[]
        #Append to list if character type of data
        if (x['attributes']['type']=='글자(음절)'):
            #Find the path between 2 directories
            true_path=''
            path1=os.path.join(args.AIHub_path,'1_syllable/'+x['image_id']+'.png')
            path2=os.path.join(args.AIHub_path,'2_syllable/'+x['image_id']+'.png')
            if os.path.isfile(path1)==True:
                true_path=path1
            elif os.path.isfile(path2)==True:
                true_path=path2
            #Save image and text
            if true_path:
                im=tf.keras.preprocessing.image.load_img(true_path,color_mode='grayscale',target_size=(args.image_size,args.image_size))
                image_arr.append(tf.keras.preprocessing.image.img_to_array(im)[:,:,0])
                label_arr.append(x['text'])
                images_before_pickle-=1
    #Unpickle Printed files
    f = open(os.path.join(args.AIHub_path,'printed_label.json')) 
    anno = json.load(f) 
    f.close() 
    random.shuffle(anno['annotations'])

    images_before_pickle=args.pickle_size
    pickle_idx=0
    image_arr,label_arr=[],[]

    for x in progressbar.progressbar(anno['annotations']):
        #Save data split into pickle
        if images_before_pickle==0:
            images_before_pickle=args.pickle_size
            #Split train and test data
            if random.random()>args.val_ratio:
                file_path=args.pickle_path
            else:
                file_path=args.val_path
            with open(os.path.join(file_path,'printed_'+str(pickle_idx)+'.pickle'),'wb') as handle:
                pickle.dump({'image':np.array(image_arr),'label':np.array(label_arr)},handle)
            pickle_idx+=1
            image_arr,label_arr=[],[]
        #Append to list if character type of data
        if (x['attributes']['type']=='글자(음절)'):
            path=os.path.join(args.AIHub_path,'syllable/'+x['image_id']+'.png')
            if os.path.isfile(path)==True:
                im=tf.keras.preprocessing.image.load_img(path,color_mode='grayscale',target_size=(args.image_size,args.image_size))
                image_arr.append(tf.keras.preprocessing.image.img_to_array(im)[:,:,0])
                label_arr.append(x['text'])

                images_before_pickle-=1

def download_AIHub_GoogleDrive():
    #Download AIHUB OCR data from Google Drive
    handwritten_file_id_1='13GCWsztfD00mHxKGNVO_c6uxS_9J_JOY'
    handwritten_file_id_2='1N2dTwZ8TgYRFBeNDKgjxjDHqULk_JX6X'
    handwritten_label_id='1rX979OhUHCKSYRbBPaMIHtFQa0eVdSXt'
    printed_file_id_1='1MNYnv4aO0kWaDigb9iEcIdpxO_pF2s-m'
    printed_label_id='1ibZrGauMoM1E9Bx2fMGtiJEqQ6nh8Qy8'
    
    idlist_file=[[handwritten_file_id_1,'handwritten-1'],[handwritten_file_id_2,'handwritten-2'],[printed_file_id_1,'printed-1']]
    idlist_label=[[handwritten_label_id,'handwritten_label'],[printed_label_id,'printed_label']]

    if os.path.isdir(args.AIHub_path)==False:
      os.mkdir(args.AIHub_path)

    print("Downloading AIHUB OCR from Google Drive...")

    for file_id in idlist_file:
        zip_dest_path=os.path.join(args.AIHub_path,f'{file_id[1]}.zip')
        gdd.download_file_from_google_drive(file_id=file_id[0],dest_path=zip_dest_path,unzip=True)
        os.remove(zip_dest_path)
    for file_id in idlist_label:
        json_dest_path=os.path.join(args.AIHub_path,f'{file_id[1]}.json')
        gdd.download_file_from_google_drive(file_id=file_id[0],dest_path=json_dest_path) 

    print("Download complete")
     
def crawl_clova_fonts():
    #Crawl and download .ttf files listed in ttf_links.txt
    #Make directory
    print('Downloading fonts in', args.font_path)
    download_path=args.font_path
    if os.path.isdir(download_path)==False:
      os.mkdir(download_path)

    #Retrieve all file locations 
    url_path='files/ttf_links.txt'
    f = open(url_path, 'r') 

    while True: 
        url = f.readline()
        # if line is empty -> EOF
        if not url: 
            break
        file_name=url.split('/')[-1].replace('\n','')
        url=url.replace('https://','')   #Parse 'https://'
        url=urllib.parse.quote(url.replace('\n',''))    #Change encoding

        urllib.request.urlretrieve('https://'+url, os.path.join(download_path,file_name))
    f.close()

def draw_single_char(ch, font):
    img = Image.new("L", (args.image_size, args.image_size), 255)
    draw = ImageDraw.Draw(img)
    draw.text((args.x_offset, args.y_offset), ch, 0, font=font)
    return img

def font2img(font_path,font_idx,charset,save_dir):
    font=ImageFont.truetype(font_path,size=args.char_size)
    image_arr,label_arr=[],[]
    
    try:
        for c in charset:
            e = draw_single_char(c, font)
            image_arr.append(np.array(e))
            label_arr.append(c)

        with open(os.path.join(save_dir,'clova_'+str(font_idx)+'.pickle'),'wb') as handle:
            pickle.dump({'image':np.array(image_arr),'label':np.array(label_arr)},handle)
    except:
        pass
        
def convert_all_fonts(charset):
    font_directory=args.font_path
    save_directory=args.pickle_path

    if os.path.isdir(save_directory)==False:
      os.mkdir(save_directory)
    if os.path.isdir(args.val_path)==False:
        os.mkdir(args.val_path)

    fonts=os.listdir(font_directory)

    for idx,font in progressbar.progressbar(enumerate(fonts)):
        full_path=os.path.join(font_directory,font)

        font2img(full_path,idx,charset,save_directory)

if __name__=='__main__':
    args = parser.parse_args()
    if args.image_test==False:
        crawl_dataset()
        
    else:
        default_font=ImageFont.truetype('files/batang.ttf',size=args.char_size)
        arr=draw_single_char('가',default_font)
        arr=np.array(arr)
        result = Image.fromarray(arr.astype(np.uint8))
        result.save('./logs/가.jpg')

        arr=draw_single_char('나',default_font)
        arr=np.array(arr)
        result = Image.fromarray(arr.astype(np.uint8))
        result.save('./logs/나.jpg')