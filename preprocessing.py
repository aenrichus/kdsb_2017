import numpy as np
import dicom
import os
import scipy.ndimage


image_dir = 'stage1/'
MIN_BOUND = -1000.0
MAX_BOUND = 400.0
PIXEL_MEAN = 0.25

patients = os.listdir(image_dir)
patients.sort()


def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    outside_image = image.min()
    image[image == outside_image] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing


def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def zero_center(image):
    image = image - PIXEL_MEAN
    return image

# pre-process all patients (no padding here, to save space)
for i in patients:
    patient_scans = load_scan(image_dir + i)
    patient_pixels = get_pixels_hu(patient_scans)
    patient_resampled, patient_spacing = resample(patient_pixels, patient_scans)
    patient_normalized = normalize(patient_resampled)
    patient_zeroed = zero_center(patient_normalized)

    patient_save = 'stage1_processed/' + str(i) + '.npy'
    np.save(patient_save, patient_zeroed)
    print("Processed patient " + str(i))

# get padding dimensions
x_val = 0
y_val = 0
z_val = 0

for i in patients:
    pat_npy = np.load('stage1_processed/' + i + '.npy')
    pat_shape = pat_npy.shape
    x_val = max(x_val, pat_shape[0])
    y_val = max(y_val, pat_shape[1])
    z_val = max(z_val, pat_shape[2])

print(x_val)
print(y_val)
print(z_val)
