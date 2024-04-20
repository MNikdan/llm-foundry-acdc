import glob
import os 
from google.cloud import storage
from pathlib import Path

def download_directory(bucket_name, prefix):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)  # Get list of files
    prefix = prefix.rstrip("/")
    dir_name = prefix.split("/")[-1]
    print(f"dir_name: {dir_name}")
    for blob in blobs:
        breakpoint()
        blob_name = blob.name.replace(prefix + '/', dir_name + '/')
        print(blob.name, blob_name)
        if blob_name.endswith("/"):
            continue
        file_split = blob_name.split("/")
        directory = "/".join(file_split[0:-1])
        Path(directory).mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(blob_name)

def upload_from_directory(directory_path: str, dest_bucket_name: str, dest_blob_name: str):
    GCS_CLIENT = storage.Client()
    rel_paths = glob.glob(directory_path + '/**', recursive=True)
    bucket = GCS_CLIENT.get_bucket(dest_bucket_name)
    for local_file in rel_paths:
        print(local_file)
        remote_path = f'{dest_blob_name}/{"/".join(local_file.split(os.sep)[1:])}'
        if os.path.isfile(local_file):
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)
