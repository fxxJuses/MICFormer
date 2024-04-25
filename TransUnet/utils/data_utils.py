import os
import shutil
import numpy as np
import nibabel as nib

from tqdm import tqdm


def split_images_and_labels(input_folder):
    '''
    Split the images and labels from .npz files into separate .npy (numpy) files
    
    Args:
        input_folder: str 
            Path to the folder containing the images and labels
    '''
    image_dir = os.path.join(input_folder, 'imagesTr')
    label_dir = os.path.join(input_folder, 'labelsTr')
    for file in tqdm(os.listdir(image_dir)):
        if file.endswith('.npz'):
            data = np.load(os.path.join(image_dir, file))['data']
            idx = file.find('_')+1
            np.save(os.path.join(image_dir, f'image_{file[idx:idx+3]}.npy'), data[0])
            np.save(os.path.join(label_dir, f'label_{file[idx:idx+3]}.npy'), data[1])
            os.remove(os.path.join(image_dir, file))

def convert_niigz_to_numpy(input_folder):
    '''
    Convert the images and labels from .nii.gz files into .npy (numpy) files
    
    Args:
        input_folder: str
            Path to the folder containing the images and labels
    '''
    for subdir in ['imagesTr', 'labelsTr', 'imagesTs']:
        dir = os.path.join(input_folder, subdir)
        if os.path.exists(dir):
            for file in tqdm(sorted(os.listdir(dir))):
                if file.endswith('.nii.gz'):
                    data = nib.load(os.path.join(dir, file)).get_fdata()
                    if data.ndim == 3:
                        data = np.transpose(data, (2, 0, 1))
                        # add channel dimension
                        data = np.expand_dims(data, axis=0)
                    elif data.ndim == 4:
                        # move channel dimension
                        data = np.transpose(data, (3, 2, 0, 1))
                    idx_start = file.find('_')+1
                    idx_end = file.find('.nii.gz')
                    np.save(os.path.join(dir, f'{subdir[:-3]}_{file[idx_start:idx_end].zfill(3)}.npy'), data)
                    os.remove(os.path.join(dir, file))

def convert_to_numpy(input_folder):
    '''
    Convert the images and labels into numpy arrays
    
    Args:
        input_folder: str 
            Path to the folder containing the images and labels
    '''
    image_dir = os.path.join(input_folder, 'imagesTr')
    if os.listdir(image_dir)[0].endswith('.npz'):
        split_images_and_labels(input_folder)
    elif os.listdir(image_dir)[0].endswith('.nii.gz'):
        convert_niigz_to_numpy(input_folder)
    else:
        raise ValueError('Images format not recognized (should be .npz or .nii.gz)')
    
def normalize_3d_array(array):
    '''
    Normalize a 3D array
    
    Args:
        array: np.ndarray
            Array to normalize
    
    Returns:
        array_norm: np.ndarray
            Normalized array
    '''
    # normalize the image
    array_norm = (array - array.mean()) / array.std()

    # rescale the image to [0, 1]
    array_norm = (array_norm - np.min(array_norm)) / (np.max(array_norm) - np.min(array_norm))

    return array_norm

def save_nifti(image, affine, filename):
    '''
    Save a 3D image to a nifti file.
    
    Args:
        image (np.array): 
            3D image
        affine (np.array):
            Affine matrix
        filename (str):
            Path to the nifti file
    '''
    img = nib.Nifti1Image(image, affine)
    nib.save(img, filename)
