import argparse
import boto3
import os
import re
import shutil
import io

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from io import BytesIO

class DatasetSyncAWS:
    def __init__(
            self,
            local_path,
            bucket_name,
            prefix='',
            ignore_patterns=None,
            aws_access_key_id=None,
            aws_secret_access_key=None
            ):
        self.local_path = local_path
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.ignore_patterns = ignore_patterns or []
        self.s3 = boto3.client(
            service_name='s3',
            region_name='eu-west-3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )

    def push_dataset(self):
        # Retrieve the list of objects in the S3 bucket
        response = self.s3.list_objects_v2(Bucket=self.bucket_name)
        existing_keys = [obj['Key'] for obj in response.get('Contents', [])]

        # Walk through the local directory
        for root, dirs, files in os.walk(self.local_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                s3_key = os.path.join(self.prefix, os.path.relpath(local_file_path, self.local_path))

                # Check if the file matches any of the ignore patterns
                if any(re.search(pattern, s3_key) for pattern in self.ignore_patterns):
                    print(f'Skipping {local_file_path}, matches ignore pattern')
                    continue

                # Check if the file already exists in the S3 bucket
                if s3_key in existing_keys:
                    print(f'Skipping {local_file_path}, already exists in S3')
                    existing_keys.remove(s3_key)  # Remove from existing keys list
                    continue

                self.s3.upload_file(local_file_path, self.bucket_name, s3_key)
                print(f'Uploaded {local_file_path} to S3: {s3_key}')

        # Delete any files in S3 that are not present in the local directory
        for key in existing_keys:
            self.s3.delete_object(Bucket=self.bucket_name, Key=key)
            print(f'Deleted {key} from S3')

    def pull_dataset(self):
        # Retrieve the list of objects in the S3 bucket
        response = self.s3.list_objects_v2(Bucket=self.bucket_name)

        s3_keys = [obj['Key'] for obj in response.get('Contents', [])]

        # Walk through the local directory
        for s3_key in s3_keys:
                
            local_file_path = os.path.join(self.local_path,s3_key)

            # Skip if the file already exists locally
            if os.path.exists(local_file_path):
                print(f'Skipping {s3_key}, already exists locally')
                continue

            # Make sure the directory exists
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Download the file from S3
            self.s3.download_file(self.bucket_name, s3_key, local_file_path)
            print(f'Downloaded {s3_key} from S3: {local_file_path}')

        # Delete any files in local that are not present in S3 bucket
        for root, dirs, files in os.walk(self.local_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                s3_key = os.path.join(os.path.relpath(local_file_path, self.local_path))

                # Check if the file matches any of the ignore patterns
                if any(re.search(pattern, file) for pattern in self.ignore_patterns):
                    print(f'Skipping {local_file_path}, matches ignore pattern')
                    continue

                # Delete the file if it's not present in S3
                if s3_key not in s3_keys:
                    os.remove(local_file_path)
                    print(f'Deleted {local_file_path} from local directory')

class DatasetSyncGoogleDrive:
    def __init__(
            self,
            local_path,
            folder_id,
            ignore_patterns=None,
            credentials=None
            ):
        self.local_path = local_path
        self.folder_id = folder_id
        self.ignore_patterns = ignore_patterns or []
        self.credentials = credentials or service_account.Credentials.from_service_account_file('path/to/credentials.json')
        self.drive_service = build('drive', 'v3', credentials=self.credentials)

    def push_dataset(self):
        pass

    # Function to download a file from Google Drive
    def download_file(self, file_id, destination_path):

        if os.path.exists(destination_path):
            print('File already exists:', destination_path)
            return

        request = self.drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False

        while done is False:
            status, done = downloader.next_chunk()

        fh.seek(0)
        with open(destination_path, 'wb') as f:
            shutil.copyfileobj(fh, f)

        print('File downloaded successfully.')


    # Function to recursively download files within a folder and its subfolders
    def download_folder(self, folder_id, destination_folder):
        results = self.drive_service.files().list(q=f"'{folder_id}' in parents and trashed=false", fields='files(id, name, mimeType)').execute()
        files = results.get('files', [])

        if not files:
            print('No files found in the folder.')
        else:
            for file in files:
                file_name = file['name']
                file_id = file['id']
                file_mime_type = file['mimeType']

                # If it's a file, download it
                if file_mime_type != 'application/vnd.google-apps.folder':
                    destination_path = os.path.join(destination_folder, file_name)
                    self.download_file(file_id, destination_path)
                    self.full_paths.append(os.path.relpath(destination_path, args.path))
                    print('Downloaded file:', file_name)

                # If it's a folder, create a subfolder and recursively download files within it
                else:
                    print('Folder ID:', file_id)
                    subfolder_path = os.path.join(destination_folder, file_name)
                    os.makedirs(subfolder_path, exist_ok=True)
                    self.download_folder(file_id, subfolder_path)


    def pull_dataset(self):

        self.full_paths = []

        # Create the local folder if it doesn't exist
        os.makedirs(self.local_path, exist_ok=True)
        
        # Download files from Google Drive not present locally
        self.download_folder(self.folder_id, self.local_path)

        # Delete any files in local that are not present in Google Drive
        for root, dirs, files in os.walk(self.local_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                current_path = os.path.join(os.path.relpath(local_file_path, self.local_path))

                # Check if the file matches any of the ignore patterns
                if any(re.search(pattern, file) for pattern in ignore_patterns):
                    print(f'Skipping {local_file_path}, matches ignore pattern')
                    continue

                # Delete the file if it's not present in Google Drive
                if current_path not in self.full_paths:
                    # os.remove(local_file_path)
                    print(f'Deleted {local_file_path} from local directory')

        print('Synchronization complete.')

if __name__ == '__main__':
    
    ignore_patterns = ['\.txt$', '\.log$', '\.DS_Store$', '\.py$', '\.pyc$', '\.sh$', '\.json$']

    ####### GOOGLE DRIVE #######

    parser = argparse.ArgumentParser(description='Sync or download dataset from Google Drive')
    parser.add_argument('-cred', '--credentials', type=str, required=True, help='Path to credentials JSON file', dest='credentials')
    parser.add_argument('-mode', '--mode', type=str, required=True, choices=['PUSH', 'PULL'], help='Sync mode', dest='mode')
    parser.add_argument('-path', '--local_dataset_path', type=str, required=True, help='Local dataset path', dest='path')
    parser.add_argument('-folder', '--folder_id', type=str, required=True, help='Google Drive folder ID', dest='folder_id')
    args = parser.parse_args()

    credentials = service_account.Credentials.from_service_account_file(args.credentials)

    sync = DatasetSyncGoogleDrive(args.path, args.folder_id, ignore_patterns=ignore_patterns, credentials=credentials)

    # python src/sync_dataset.py -cred client_secret.json -folder 1StzxTYaPJBFqlxfXQhdxbcgfKdotaxoF -path "data/GDrive" -mode PULL

    
    ####### AWS S3 #######

    # parser = argparse.ArgumentParser(description='Sync or download dataset from S3')
    # parser.add_argument('-id', '--aws_access_key_id', type=str, required=True, help='AWS access key ID')
    # parser.add_argument('-key', '--aws_secret_access_key', type=str, required=True, help='AWS secret access key')
    # parser.add_argument('-mode', '--mode', type=str, required=True, choices=['PUSH', 'PULL'], help='Sync mode')
    # parser.add_argument('-path', '--local_dataset_path', type=str, required=True, help='Local dataset path')
    # args = parser.parse_args()

    # s3_bucket_name = 'sfh-dataset'

    # sync = DatasetSyncAWS(args.local_dataset_path, s3_bucket_name, ignore_patterns=ignore_patterns, aws_access_key_id=args.aws_access_key_id, aws_secret_access_key=args.aws_secret_access_key)

    ##############################

    if args.mode == 'PUSH':
        sync.push_dataset()
    elif args.mode == 'PULL':
        sync.pull_dataset()
    else:
        raise ValueError(f'Unknown sync mode: {args.mode}')
