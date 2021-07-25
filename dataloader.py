from typing import Union
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import (
    Dataset,
    DataLoader
)
from torchvision import transforms
from preprocess import (
    TVL1_optical_flow,
    TVL1_magnitude,
    gray_frame,
    optical_strain
)


def LOSO_specific_subject_sequence_generate(data: pd.DataFrame, sub_column: str,
                                            sub: str) -> tuple:
    """Generate train and test data using leave-one-subject-out for
    specific subject

    Parameters
    ----------
    data : pd.DataFrame
        Original DataFrame
    sub_column : str
        Subject column in DataFrame
    sub : str
        Subject to be leave out in training

    Returns
    -------
    tuple
        Return training and testing DataFrame
    """

    # Mask for the training
    mask = data["Subject"].isin([sub])

    # Masking for the specific data
    train_data = data[~mask]
    test_data = data[mask]

    return train_data, test_data


def LOSO_sequence_generate(data: pd.DataFrame, sub_column: str) -> tuple:
    """Generate train and test data using leave-one-subject-out for
    specific subject

    Parameters
    ----------
    data : pd.DataFrame
        Original DataFrame
    sub_column : str
        Subject column in DataFrame

    Returns
    -------
    tuple
        Return training and testing list DataFrame
    """
    # Save the training and testing list for all subject
    train_list = []
    test_list = []

    # Unique subject in `sub_column`
    subjects = np.unique(data[sub_column])

    for subject in subjects:
        # Mask for the training
        mask = data["Subject"].isin([subject])

        # Masking for the specific data
        train_data = data[~mask].reset_index(drop=True)
        test_data = data[mask].reset_index(drop=True)

        train_list.append(train_data)
        test_list.append(test_data)

    return train_list, test_list


class MyDataset(Dataset):
    def __init__(self, data_info: pd.DataFrame, label_mapping: dict,
                 preprocess_path: str, mode: tuple,
                 catego: str):
        
        # Check if mode is correct or not
        assert len(mode) <= 2, f"Image mode contain at most 2 categories, give {len(mode)}"

        self.data_info = data_info
        self.label_mapping = label_mapping
        self.preprocess_path = preprocess_path
        self.catego = catego
        
        if len(mode) == 2:
            self.mode = mode
        else:
            self.mode = mode[0]

        self.transforms = transforms.Compose([
            transforms.Lambda(lambda x: x / 255.0),
            transforms.Normalize(
                mean=(0.406, 0.456, 0.485),
                std=(1.0, 1.0, 1.0)
            )
        ])

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx: int):
        # Label for the image
        label = self.label_mapping[self.data_info.loc[idx, "Estimated Emotion"]]
        
        # Load in the matrix that already been preprocessed
        if self.catego == "SAMM":
            npz_path = f"{self.preprocess_path}/{self.data_info.loc[idx, 'Filename']}.npz"
        else:
            subject = self.data_info.loc[idx, "Subject"]
            npz_path = f"{self.preprocess_path}/{subject}_{self.data_info.loc[idx, 'Filename']}.npz"
        preprocess_image = np.load(npz_path)
        
        # Load in the flow
        flow = torch.FloatTensor(preprocess_image["flow"])

        if isinstance(self.mode, (tuple, list)):
            # Create a space for the flow
            stream_one = torch.FloatTensor(preprocess_image[MyDataset.build_mode(self.mode[0])])
            stream_one = stream_one.unsqueeze(0)
            image_one = torch.cat([flow, stream_one], dim=0)
            image_one = self.transforms(image_one)
            
            stream_two = torch.FloatTensor(preprocess_image[MyDataset.build_mode(self.mode[1])])
            stream_two = stream_two.unsqueeze(0)
            image_two = torch.cat([flow, stream_two], dim=0)
            image_two = self.transforms(image_two)

            return (image_one, image_two), label
        else:
            stream = torch.FloatTensor(preprocess_image[MyDataset.build_mode(self.mode)])
            stream = stream.unsqueeze(0)
            image = torch.cat([flow, stream], dim=0)
            image = self.transforms(image)

            return image, label
        
    @staticmethod
    def build_mode(mode):
        if mode == "F":
            return "mag"
        elif mode == "G":
            return "gray"
        elif mode == "S":
            return "strain"


def get_loader(csv_file, preprocess_path,
               label_mapping, mode, batch_size,
               catego, shuffle=True):
    dataset = MyDataset(data_info=csv_file,
                        label_mapping=label_mapping,
                        preprocess_path=preprocess_path,
                        catego=catego,
                        mode=mode)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            pin_memory=True)

    return dataset, dataloader
