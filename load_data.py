
from google_drive_downloader import GoogleDriveDownloader as gdd
import os
if __name__ =='__main__':
    val_data_link='1WOP_sQsu4vXCY739VGgiWIbHyOozcjHw'
    data_link='1HBu43eBO-vXJsJp8crEp_Iih3_U7QSR2'
    zip_dest_path=''
    print("Downloading data...")
    gdd.download_file_from_google_drive(file_id=data_link,dest_path=os.path.join(zip_dest_path,'data.zip'),unzip=True)
    gdd.download_file_from_google_drive(file_id=val_data_link,dest_path=os.path.join(zip_dest_path,'val_data.zip'),unzip=True)

    os.remove(os.path.join(zip_dest_path,'data.zip'))
    os.remove(os.path.join(zip_dest_path,'val_data.zip'))
    print("Downloading complete...")