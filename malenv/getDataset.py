import os
import shutil
import concurrent.futures
from tqdm import tqdm

def copy_file(src_file, dst_dir):
    if src_file.endswith('.exe'):
        shutil.copy(src_file, dst_dir)

def copy_exe_files(src_dir, dst_dir):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for dirpath, dirnames, filenames in os.walk(src_dir):
            for filename in tqdm(filenames, desc="Copying files"):
                src_file = os.path.join(dirpath, filename)
                executor.submit(copy_file, src_file, dst_dir)

# Replace 'source_directory' with the path of the directory you want to search
# Replace 'destination_directory' with the path of the directory where you want to copy the .exe files
# copy_exe_files('/media/nerdcoder/DATA', '/media/nerdcoder/8aa66b6f-a773-4c23-91aa-24ea87632802/nerd-coder/benign')

import os

def delete_large_files(dir_path, size_limit):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.getsize(file_path) > size_limit * 1024 * 1024:  # size in bytes
                os.remove(file_path)
                print(f"Deleted {file_path}")

def deleteemptyfiles(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.getsize(file_path) ==0:  # size in bytes
                os.remove(file_path)
                print(f"Deleted {file_path}")


# Replace 'directory_path' with the path of the directory where you want to delete the files
# Replace 'size_limit' with the size limit in MB
# delete_large_files('/media/nerdcoder/8aa66b6f-a773-4c23-91aa-24ea87632802/nerd-coder/benign', size_limit=100)
deleteemptyfiles('Data\malware')