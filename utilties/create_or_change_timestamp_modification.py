import os

# import time
import ast
import datetime
import time

from PIL.Image import Image

# Create a dummy file for demonstration


def change_modification_time(file_path=None, new_time=None):
    """
    If file is not provided a dummy file is created for demonstration, otherwise
    use with precaution for changing any file time
    :param file_path: path of file to change it's modified time
    :param new_time: desired time, or default, if None is provided
    :return: the new time
    """

    if not os.path.exists(file_path):
        file_path = "test.py"
        code = ast.parse('print("Hello World")')
        with open(file_path, "w") as f:
            f.write(ast.unparse(code))
        print("Since file path was not provided, a dummy file has been created")
    file_path = os.path.abspath(file_path)
    print(os.path.abspath(file_path))
    time_stamp = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
    print(f"Original modification time: {time_stamp}")

    # Define the new desired modification time
    # For instance, set it to a specific date and time
    new_year = 2021
    new_month = 1
    new_day = 15
    new_hour = 10
    new_minute = 30
    new_second = 0
    if not new_time:
        desired_datetime = datetime.datetime(
            new_year, new_month, new_day, new_hour, new_minute, new_second
        )
    else:
        desired_datetime = new_time
    print(desired_datetime)
    # Convert the datetime object to a Unix timestamp (seconds since epoch)
    new_timestamp = desired_datetime
    print(time_stamp.timestamp())
    print(new_timestamp.timestamp())
    # Change the file's access time and modification time
    # The second argument to os.utime() is a tuple (atime, mtime)
    # We are setting both to the same new_timestamp for simplicity

    os.utime(file_path, (time_stamp.timestamp(), new_timestamp.timestamp()))

    print(
        f"New modification time: {new_timestamp.fromtimestamp(os.path.getmtime(file_path))}"
    )

    # Clean up the dummy file
    # os.remove(file_path)


def main():
    folder = "/Users/deven/Developer/images/"

    for file in os.listdir(folder):
        print(os.path.getmtime(os.path.join(folder, file)))
        # Image.getdata(file)
        change_modification_time(
            file_path=os.path.join(folder, file)
        )  # ,time.strptime("dd/mm/yyyy"[(12/12/2011),]))


if __name__ == "__main__":
    main()
