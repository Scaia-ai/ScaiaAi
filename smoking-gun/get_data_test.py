from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import urllib.error
import os
import shutil
from pathlib import Path

def clone_test_data(download_to_folder = os.path.join("input_data", "emails")):
    source = Path('../test-data/test-03')
    destination = Path('../smoking-gun/input-data/emails')
    output=Path('../smoking-gun/output-data')
    #mkdir -p ../smoking-gun/input-data/emails
    if os.path.exists(output):
        shutil.rmtree(output)
    os.makedirs(output)
    # mkdir -p ../smoking-gun/output-data
    if os.path.exists(destination):
        shutil.rmtree(destination)
    os.makedirs(destination)  # make the directory
    # copy files ../test-data/test-03/* to ../smoking-gun/input_data/emails
    for item in os.listdir(source):
        s = source / item
        d = destination / item
        if s.is_dir():
            copy_dir(s, d)
        else:
            shutil.copy2(str(s), str(d))

    return
if __name__ == "__main__":

    clone_test_data(download_to_folder = os.path.join("input_data", "emails"))



