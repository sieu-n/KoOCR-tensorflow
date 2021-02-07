
from google_drive_downloader import GoogleDriveDownloader as gdd
import concurrent.futures
import os
import zipfile

def unzip_data(file):
    zf.extract(file,path='./data/')
def unzip_val_data(file):
    zf.extract(file,path='./val_data/')
if __name__ =='__main__':
    val_data_link='1WOP_sQsu4vXCY739VGgiWIbHyOozcjHw'
    data_link='1HBu43eBO-vXJsJp8crEp_Iih3_U7QSR2'

    val_data_path='./val_data'
    data_path='./data'
    #Create data directory
    if os.path.isdir(val_data_path)==False:
        os.makedirs(val_data_path)
    if os.path.isdir(data_path)==False:
        os.makedirs(data_path)

    print("Downloading data...")

    print('Unzipping data...')
    zf = zipfile.ZipFile(os.path.join(data_path,'data.zip'))    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(unzip_data, zf.infolist())
    os.remove(os.path.join(data_path,'data.zip'))

    zf = zipfile.ZipFile(os.path.join(val_data_path,'val_data.zip'))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(unzip_val_data, zf.infolist())
    os.remove(os.path.join(val_data_path,'val_data.zip'))
    print("Downloading complete...")