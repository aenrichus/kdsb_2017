import numpy as np
import pandas as pd
import dicom
import os
import sqlite3
import scipy.ndimage
from collections import OrderedDict


image_dir = 'stage1/'
patients = os.listdir(image_dir)
patients.sort()


def load_scan(path):
    """
    This function loads the slices of a scan for a patient.

    :param path:
    :return:
    """
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def resample(image, scan, new_spacing=[1, 1, 1]):  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< COULD THIS BE IMPROVED?
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image, new_spacing


MIN_BOUND = -1000.0
MAX_BOUND = 400.0


def normalize(image):  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Is there no numpy function for this already?
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


PIXEL_MEAN = 0.25


def zero_center(image):  # <<<<<<<<<<<<<<<<<<<<<<<< Do we want to do this? If we do, we should calculate the real mean.
    image = image - PIXEL_MEAN
    return image


# load all of the patients
patient_scans = OrderedDict()
patient_pixels = OrderedDict()
patient_resampled = OrderedDict()
patient_spacing = OrderedDict()

for i in patients:
    patient_scans[i] = load_scan(image_dir + i)
    patient_pixels[i] = get_pixels_hu(patient_scans[i])
    patient_resampled[i], patient_spacing[i] = resample(patient_pixels[i], patient_scans[i])

# code used for testing the above
first_patient = load_scan(image_dir + patients[0])  # returns raw dicom data for each slice
first_patient_pixels = get_pixels_hu(first_patient)  # returns arrays with HU values
first_patient_pixels.shape  # 145 slices that are 512x512 squares

first_patient_resampled, spacing = resample(first_patient_pixels, first_patient)
first_patient_resampled.shape

first_patient_normalized = normalize(first_patient_resampled)
first_patient_zeroed = zero_center(first_patient_normalized)
