from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import urllib.error
import os

def download_enron_data(download_to_folder = os.path.join("input_data", "emails")):

    assert os.path.exists(download_to_folder), "Folder name 'input_data' or subfolder 'emails' donot exist. Please create empty folders with these names & rerun again"

    for file_number in list(range(3, 4)):
                
        file_name       = "enron{0:0=3d}".format(file_number)
        download_url    = "https://s3.amazonaws.com/freeeed.org/enron/results/{}.zip".format(file_name)

        try:
            resp = urlopen(download_url)
        except urllib.error.HTTPError:
            print("File {} - doesn't exist. Skipping this download".format(download_url))
            continue

        print("Downloading File... {}".format(download_url))
        try:
            zipfile = ZipFile(BytesIO(resp.read()))
            for file in zipfile.namelist():
                if file.startswith("text/"):
                    with open(os.path.join(download_to_folder, "{}".format(file_name+"_"+file.replace("text/", ""))), "wb") as fn:
                        fn.writelines(zipfile.open(file).readlines())
        except:
            print("Bad File Format {}. Skipping this download".format(download_url))
        
    return

if __name__ == "__main__":

    download_enron_data(download_to_folder = os.path.join("input_data", "emails"))

