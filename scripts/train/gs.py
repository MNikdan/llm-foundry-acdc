import glob
import os 
from google.cloud import storage

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
