import os
import urllib.request
import argparse
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import PIL
import json
import _pickle as pickle    #cPickle
import progressbar
import collections

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
parser.add_argument("--pickle_path", type=str,default='./data')

parser.add_argument("--image_size", type=int,default=256)
parser.add_argument("--x_offset", type=int,default=50)
parser.add_argument("--y_offset", type=int,default=10)
parser.add_argument("--char_size", type=int,default=200)

def crawl_dataset():
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

def filter_recurring_hash(charset, font):
    hash_list=collections.defaultdict(int)
    for c in charset:
        img = draw_single_char(c, font)
        hash_list[hash(img.tobytes())] += 1
    recurring_hashes = filter(lambda d: d[1] > 2, hash_list.items())
    return [rh[0] for rh in recurring_hashes]

def draw_single_char(ch, font):
    img = Image.new("L", (args.image_size, args.image_size), 255)
    draw = ImageDraw.Draw(img)
    draw.text((args.x_offset, args.y_offset), ch, 0, font=font)
    return img

def font2img(font_path,font_idx,charset,save_dir):
    font=ImageFont.truetype(font_path,size=args.char_size)
    image_arr,label_arr=[],[]
    
    try:
        for c in progressbar.progressbar(charset):
            e = draw_single_char(c, font)
            image_arr.append(np.array(e))
            label_arr.append(c)

        with open(os.path.join(save_dir,str(font_idx)+'.pickle'),'wb') as handle:
            pickle.dump({'image':np.array(image_arr),'label':np.array(label_arr)},handle)
    except:
        print('Some error occured while processing',font)
        
def convert_all_fonts(charset):
    font_directory=args.font_path
    save_directory=args.pickle_path

    if os.path.isdir(save_directory)==False:
      os.mkdir(save_directory)

    fonts=os.listdir(font_directory)

    for idx,font in enumerate(fonts):
        full_path=os.path.join(font_directory,font)

        print('Converting',font)
        font2img(full_path,idx,charset,save_directory)

if __name__=='__main__':
    args = parser.parse_args()

    crawl_dataset()
    charset=load_charset()
    convert_all_fonts(charset)