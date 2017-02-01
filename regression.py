"""

Notes: Only about 26% of patients actually have cancer.

"""

import numpy as np
import pandas as pd
import dicom
import os
import seaborn as sns
import statsmodels.formula.api as smf

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


# read in labels csv file
cancer = pd.read_csv('stage1_labels.csv')
np.mean(cancer['cancer'])  # skewed data: about 26% positive for cancer

# load all of the patients, scans, and means
patient_data = pd.DataFrame(columns=['id', 'mean', 'scan'])

for i in patients:
    scan = get_pixels_hu(load_scan(image_dir + i))
    scan_mean = np.mean(scan)
    scan_flat = scan.flatten()
    current_patient = [(i, scan_mean, scan_flat)]
    patient_append = pd.DataFrame(current_patient, columns=['id', 'mean', 'scan'])
    patient_data = pd.concat([patient_data, patient_append], ignore_index=True)

# join the data frames by id; set index to id
patient_data = patient_data.merge(cancer, on='id', how='left')  # oddly missing some labels
patient_data = patient_data.dropna()
patient_data.set_index(['id'], inplace=True)

# run linear regression
lm = smf.ols(formula='cancer ~ mean', data=patient_data).fit()
lm.summary()  # really bad

# plot (but this is really worthless)
sns.regplot(patient_data['mean'], patient_data['cancer'])

# trim values to between -400 and +400 (this can take a while)
for i, j in patient_data.iterrows():
    in_range = np.where((patient_data['scan'][i] >= -400) & (patient_data['scan'][i] <= 400))
    patient_data['scan'][i] = patient_data['scan'][i][in_range]
    patient_data['mean'][i] = np.mean(patient_data['scan'][i])

# retry regression
lm = smf.ols(formula='cancer ~ mean', data=patient_data).fit()
lm.summary()  # still really bad

sns.regplot(patient_data['mean'], patient_data['cancer'])
