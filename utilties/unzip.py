import os
import tarfile
from zipfile import ZipFile


def unzip(file):
    """ """
    if os.path.exists(file) and os.path.isfile(file):
        main_dir = os.path.split(os.path.abspath(file))[0]
        with ZipFile(file, "r") as zip_ref:
            zip_ref.extractall(main_dir + file.split(".")[0])


def untar(tar_file_path):

    try:
        directory = tar_file_path.split(".")[0]

        destination_directory = (
            os.path.dirname(os.path.abspath(tar_file_path)) + "/" + directory
        )
        os.makedirs(destination_directory + "/" + directory, exist_ok=True)
        with tarfile.open(tar_file_path, "r") as tar:
            tar.extractall(path=destination_directory)
        print(f"Successfully extracted '{tar_file_path}' to '{destination_directory}'")
    except tarfile.ReadError:
        print(f"Error: Could not open or read the tar file '{tar_file_path}'.")
    except FileNotFoundError:
        print(f"Error: The file '{tar_file_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
