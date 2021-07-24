import argparse
import os

def rename(path: str):
    """Rename the file in the specific folder
    with its number has length 5

    Parameters
    ----------
    path : str
        Path for the folder
    """
    for root, dir, files in os.walk(path):
        for filename in files:
            if filename[-4:] == ".jpg":
                pre_name = filename.split("_")[0]
                file_num = filename.split("_")[1].split(".")[0]
                new_file_num = file_num.zfill(5)

                new_file_name = pre_name + "_" + new_file_num + ".jpg"

                os.rename(f"{root}/{filename}", f"{root}/{new_file_name}")


if __name__ == "__main__":
    # Argument parse
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path",
                        type=str,
                        required=True)
    args = parser.parse_args()

    rename(args.path)
