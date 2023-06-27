import argparse
import boto3
import os
import re

class DatasetSync:
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
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sync or download dataset from S3')
    parser.add_argument('-id', '--aws_access_key_id', type=str, required=True, help='AWS access key ID')
    parser.add_argument('-key', '--aws_secret_access_key', type=str, required=True, help='AWS secret access key')
    parser.add_argument('-mode', '--mode', type=str, required=True, choices=['PUSH', 'PULL'], help='Sync mode')
    parser.add_argument('-path', '--local_dataset_path', type=str, required=True, help='Local dataset path')
    args = parser.parse_args()

    s3_bucket_name = 'sfh-dataset'
    ignore_patterns = ['\.txt$', '\.log$', '\.DS_Store$']

    sync = DatasetSync(args.local_dataset_path, s3_bucket_name, ignore_patterns=ignore_patterns, aws_access_key_id=args.aws_access_key_id, aws_secret_access_key=args.aws_secret_access_key)

    if args.mode == 'PUSH':
        sync.push_dataset()
    elif args.mode == 'PULL':
        sync.pull_dataset()
    else:
        raise ValueError(f'Unknown sync mode: {args.mode}')
