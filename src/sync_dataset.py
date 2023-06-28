import argparse
import boto3
import os
import re

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
        # Retrieve the list of objects in the Google Drive folder
        query = f"'{self.folder_id}' in parents and trashed = false"
        response = self.drive_service.files().list(q=query, fields='nextPageToken, files(id, name, mimeType)').execute()
        existing_files = {file['name']: file['id'] for file in response.get('files', [])}

        # Walk through the local directory
        for root, dirs, files in os.walk(self.local_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                file_name = os.path.basename(local_file_path)

                # Check if the file matches any of the ignore patterns
                if any(re.search(pattern, file_name) for pattern in self.ignore_patterns):
                    print(f'Skipping {local_file_path}, matches ignore pattern')
                    continue

                # Check if the file already exists in the Google Drive folder
                if file_name in existing_files:
                    print(f'Skipping {local_file_path}, already exists in Google Drive')
                    existing_files.pop(file_name)  # Remove from existing files list
                    continue

                # Upload the file to Google Drive
                file_metadata = {'name': file_name, 'parents': [self.folder_id]}
                media = MediaFileUpload(local_file_path, resumable=True)
                file = self.drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
                print(f'Uploaded {local_file_path} to Google Drive: {file.get("id")}')

        # Delete any files in Google Drive that are not present in the local directory
        for file_id in existing_files.values():
            try:
                self.drive_service.files().delete(fileId=file_id).execute()
                print(f'Deleted {file_id} from Google Drive')
            except HttpError as error:
                print(f'An error occurred: {error}')

    def pull_dataset(self):
        # Retrieve the list of objects in the Google Drive folder
        query = f"'{self.folder_id}' in parents and trashed = false"
        response = self.drive_service.files().list(q=query, fields='nextPageToken, files(id, name)').execute()
        #drive_files = {file['name']: file['id'] for file in response.get('files', [])}
        items = response.get('files', [])
        print(self.drive_service.files())
        return
        

        # Walk through the local directory
        for file_name, file_id in drive_files.items():
            local_file_path = os.path.join(self.local_path, file_name)

            # Skip if the file already exists locally
            if os.path.exists(local_file_path):
                print(f'Skipping {file_name}, already exists locally')
                continue

            # Download the file from Google Drive
            request = self.drive_service.files().get_media(fileId=file_id)
            file = BytesIO()
            downloader = MediaIoBaseDownload(file, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                print(f'Download {int(status.progress() * 100)}.')
            file.seek(0)

            with open(local_file_path, 'wb') as f:
                f.write(file.read())
            print(f'Downloaded {file_name} from Google Drive: {local_file_path}')

        # # Delete any files in local that are not present in Google Drive folder
        # for root, dirs, files in os.walk(self.local_path):
        #     for file in files:
        #         local_file_path = os.path.join(root, file)

        #         # Check if the file matches any of the ignore patterns
        #         if any(re.search(pattern, file) for pattern in self.ignore_patterns):
        #             print(f'Skipping {local_file_path}, matches ignore pattern')
        #             continue

        #         # Delete the file if it's not present in Google Drive
        #         if file not in drive_files:
        #             os.remove(local_file_path)
        #             print(f'Deleted {local_file_path} from local directory')

if __name__ == '__main__':
    
    ignore_patterns = ['\.txt$', '\.log$', '\.DS_Store$']

    ####### GOOGLE DRIVE #######

    parser = argparse.ArgumentParser(description='Sync or download dataset from Google Drive')
    parser.add_argument('-cred', '--credentials', type=str, required=True, help='Path to credentials JSON file')
    parser.add_argument('-mode', '--mode', type=str, required=True, choices=['PUSH', 'PULL'], help='Sync mode')
    parser.add_argument('-path', '--local_dataset_path', type=str, required=True, help='Local dataset path')
    parser.add_argument('-folder', '--folder_id', type=str, required=True, help='Google Drive folder ID')
    args = parser.parse_args()

    credentials = service_account.Credentials.from_service_account_file(args.credentials)

    sync = DatasetSyncGoogleDrive(args.local_dataset_path, args.folder_id, ignore_patterns=ignore_patterns, credentials=credentials)

    # python src/sync_dataset.py -cred cred.json -folder 1StzxTYaPJBFqlxfXQhdxbcgfKdotaxoF -path "data/s3-bucket" -mode PULL

    
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
