from google_drive_downloader import GoogleDriveDownloader as gdd
import concurrent.futures
import os
import zipfile
from py7zr import unpack_7zarchive
import shutil

import argparse


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Download dataset')
parser.add_argument("--sevenzip", type=str2bool,default=True)

def unzip_7z():
    val_data_7z_link='17NusZQw2RKpBIvCKp6hW6RuJKY43SWqz'
    data_7z_link='1-I3BtCzYE7swpKERGygRhlMnim2xX0_2'

    print("Downloading data...")
    
    gdd.download_file_from_google_drive(file_id=data_7z_link,dest_path=os.path.join(data_path,'data.7z'),unzip=False)
    gdd.download_file_from_google_drive(file_id=val_data_7z_link,dest_path=os.path.join(val_data_path,'val_data.7z'),unzip=False)
    
    print('Unzipping data...')
    shutil.register_unpack_format('7zip', ['.7z'], unpack_7zarchive)

    shutil.unpack_archive(os.path.join(data_path,'data.7z'), data_path)
    os.remove(os.path.join(data_path,'data.7z'))

    shutil.unpack_archive(os.path.join(val_data_path,'val_data.7z'), val_data_path)
    os.remove(os.path.join(val_data_path,'data.7z'))
    print("Downloading complete...")

def unzip_zip():
    val_data_link='1WOP_sQsu4vXCY739VGgiWIbHyOozcjHw'
    data_link='1HBu43eBO-vXJsJp8crEp_Iih3_U7QSR2'

    print("Downloading data...")

    
    print('Unzipping data...')
    with zipfile.ZipFile(os.path.join(data_path,'data.zip'), 'r') as zip_ref:
        zip_ref.extractall(data_path)
    os.remove(os.path.join(data_path,'data.zip'))

    with zipfile.ZipFile(os.path.join(val_data_path,'val_data.zip'), 'r') as zip_ref:
        zip_ref.extractall(val_data_path)
    os.remove(os.path.join(val_data_path,'val_data.zip'))
    print("Downloading complete...")

if __name__ =='__main__':
    args = parser.parse_args()

    val_data_path='./val_data'
    data_path='./data'
    #Create data directory
    if os.path.isdir(val_data_path)==False:
        os.makedirs(val_data_path)
    if os.path.isdir(data_path)==False:
        os.makedirs(data_path)

    if args.sevenzip:
        unzip_7z()
    else:
        unzip_zip()