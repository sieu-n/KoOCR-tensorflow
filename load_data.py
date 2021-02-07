
from google_drive_downloader import GoogleDriveDownloader as gdd
import concurrent.futures
import os
import zipfile

def unzip(file,p):
    zf.extract(file,path=p)
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
    gdd.download_file_from_google_drive(file_id=data_link,dest_path=os.path.join(data_path,'data.zip'),unzip=False)
    gdd.download_file_from_google_drive(file_id=val_data_link,dest_path=os.path.join(val_data_path,'val_data.zip'),unzip=False)

    print('Unzipping data...')
    unzip_lambda=lambda f,p:unzip(f,p)
    zf = zipfile.ZipFile(os.path.join(data_path,'data.zip'))    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(unzip_lambda(p=data_path), zf.infolist())
    os.remove(os.path.join(data_path,'data.zip'))

    zf = zipfile.ZipFile(os.path.join(val_data_path,'val_data.zip'))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(unzip_lambda(p=val_data_path), zf.infolist())
    os.remove(os.path.join(val_data_path,'val_data.zip'))
    print("Downloading complete...")