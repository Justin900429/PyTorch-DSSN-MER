from typing import Union
import numpy as np
import pandas as pd
import cv2


def center_crop(img: np.array, crop_size: Union[tuple, int]) -> np.array:
    """Returns center cropped image

    Parameters
    ----------
    img : [type]
        Image to do center crop
    crop_size : Union[tuple, int]
        Crop size of the image

    Returns
    -------
    np.array
        Image after being center crop
    """
    width, height = img.shape[1], img.shape[0]

    # Height and width of the image
    mid_x, mid_y = int(width / 2), int(height / 2)

    if isinstance(crop_size, tuple):
        crop_width, crop_hight = int(crop_size[0] / 2), int(crop_size[1] / 2)
    else:
        crop_width, crop_hight = int(crop_size / 2), int(crop_size / 2)
    crop_img = img[mid_y-crop_hight:mid_y+crop_hight, mid_x-crop_width:mid_x+crop_width]

    return crop_img


def TVL1_optical_flow(prev_frame: np.array, next_frame: np.array):
    """Compute the TV-L1 optical flow and normalized the result"""
    # Transform the image from BGR to Gray
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # Create TV-L1 optical flow
    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = optical_flow.calc(prev_frame, next_frame, None)

    # Do the normalization
    return normalized(flow)


def TVL1_magnitude(flow: np.array):
    """Compute the magnitude of the frame"""
    return np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)


def optical_strain(flow: np.array) -> np.array:
    """Compute the optical strain for the given u, v
    Refer to: https://github.com/mariaoliverparera/mod-opticalStrain/blob/master/get_contours.py

    Parameters
    ----------
    flow : np.array
        Normalized horizontal and vertical optical flow fields

    Returns
    -------
    np.array
        Return the optical strain magnitude for u, v
    """
    u = flow[..., 0]
    v = flow[..., 1]

    # Compute the gradient
    u_x, u_y = np.gradient(u)
    v_x, v_y = np.gradient(v)

    e_xy = 0.5 * (u_y + v_x)

    e_mag = u_x**2 + 2 * e_xy**2 + v_y**2
    e_mag = np.sqrt(e_mag)

    return e_mag


def gray_frame(frame):
    """Transform the frame into Gray and normalized the result"""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Do the normalization
    return normalized(frame, lambda_=16)


def normalized(frame: np.array,
               g_min: float = -128,
               g_max: float = 128,
               lambda_: int = 16):
    # Do the normalization
    if len(frame.shape) > 2:
        f_min = np.amin(frame, axis=(0, 1))
        f_max = np.amax(frame, axis=(0, 1))
    else:
        f_min = np.min(frame)
        f_max = np.max(frame)

    frame = lambda_ * (frame - f_min) * (g_max - g_min) / (f_max - f_min) + g_min

    return frame

