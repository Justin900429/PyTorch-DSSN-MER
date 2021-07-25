import os
import argparse
from typing import Union
import numpy as np
import pandas as pd
import cv2
from preprocess import (
    center_crop,
    TVL1_optical_flow,
    TVL1_magnitude,
    gray_frame,
    optical_strain
)
from read_file import read_csv


def save_SAMM_preprocess_weight(data_info: pd.DataFrame, root: str,
                                path: str, crop_size: Union[tuple, int],
                                new_size:int = 227):
    """Save the preprocess weight for the repeating usage

    Parameters
    ----------
    data_info : pd.DataFrame
        DataFrame for the data info
    root : str
        Root of the original image
    path : str
        Path to save the weight
    crop_size: Union[tuple, int]
        Center crop size for the image
    
    """
    os.makedirs(path, exist_ok=True)

    for idx in range(len(data_info)):
        # Subject of the data folder
        subject = data_info.loc[idx, "Subject"]
        img_root = f"{root}/{subject}/{data_info.loc[idx, 'Filename']}/"
        onset_frame_name = img_root + f"{subject}_{data_info.loc[idx, 'Onset Frame']:05}.jpg"
        apex_frame_name = img_root + f"{subject}_{data_info.loc[idx, 'Apex Frame']:05}.jpg"

        onset_frame = cv2.imread(onset_frame_name)
        if onset_frame is None:
            print(onset_frame_name)
        onset_frame = center_crop(onset_frame, (500, 500))

        apex_frame = cv2.imread(apex_frame_name)
        if apex_frame is None:
            print(apex_frame_name)
        apex_frame = center_crop(apex_frame, (500, 500))

        # Convert to optical flow and save
        flow = TVL1_optical_flow(prev_frame=onset_frame, 
                                 next_frame=apex_frame)
        # Save magnitude
        mag = TVL1_magnitude(flow)
        # Save gray
        gray = gray_frame(apex_frame)
        # Save strain
        strain = optical_strain(flow)
        np.savez(file=f"{path}/{data_info.loc[idx, 'Filename']}.npz",
                 flow=np.transpose(cv2.resize(flow, (new_size, new_size)), (2, 0, 1)),
                 mag=cv2.resize(mag, (new_size, new_size)),
                 gray=cv2.resize(gray, (new_size, new_size)),
                 strain=cv2.resize(strain, (new_size, new_size)))
        print(f"Finish processing {path}/{data_info.loc[idx, 'Filename']}.npz")


def save_CASME_preprocess_weight(data_info: pd.DataFrame, root: str,
                                 path: str, new_size: int = 227):
    """Save the preprocess weight for the repeating usage

    Parameters
    ----------
    data_info : pd.DataFrame
        DataFrame for the data info
    root : str
        Root of the original image
    path : str
        Path to save the weight
    """
    os.makedirs(path, exist_ok=True)

    for idx in range(len(data_info)):
        # Subject of the data folder
        subject = data_info.loc[idx, "Subject"]
        img_root = f"{root}/sub{subject}/{data_info.loc[idx, 'Filename']}/"
        onset_frame_name = img_root + f"img{data_info.loc[idx, 'Onset Frame']}.jpg"
        apex_frame_name = img_root + f"img{data_info.loc[idx, 'Apex Frame']}.jpg"

        onset_frame = cv2.imread(onset_frame_name)
        if onset_frame is None:
            print(onset_frame_name)
        onset_frame = cv2.resize(onset_frame, (new_size, new_size))

        apex_frame = cv2.imread(apex_frame_name)
        if apex_frame is None:
            print(apex_frame_name)
        apex_frame = cv2.resize(apex_frame, (new_size, new_size))

        # Convert to optical flow and save
        flow = TVL1_optical_flow(prev_frame=onset_frame, 
                                 next_frame=apex_frame)
        # Save magnitude
        mag = TVL1_magnitude(flow)
        # Save gray
        gray = gray_frame(apex_frame)
        # Save strain
        strain = optical_strain(flow)
        np.savez(file=f"{path}/{subject}_{data_info.loc[idx, 'Filename']}.npz",
                 flow=np.transpose(flow, (2, 0, 1)),
                 mag=mag,
                 gray=gray,
                 strain=strain)
        print(f"Finish processing {path}/{subject}_{data_info.loc[idx, 'Filename']}.npz")


if __name__ == "__main__":
    # Argument parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path",
                        type=str,
                        required=True)
    parser.add_argument("--root",
                        type=str,
                        required=True)
    parser.add_argument("--save_path",
                        type=str,
                        required=True)
    parser.add_argument("--CASME",
                        action="store_true",
                        default=False)
    args = parser.parse_args()

    # Save the weight
    data, _ = read_csv(args.csv_path)
    
    if args.CASME:
        save_CASME_preprocess_weight(data_info=data,
                                     root=args.root,
                                     path=args.save_path)
    else:
        save_SAMM_preprocess_weight(data_info=data,
                                    root=args.root,
                                    path=args.save_path,
                                    crop_size=(500, 500))