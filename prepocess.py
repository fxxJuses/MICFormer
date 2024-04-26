import ants 
import numpy as np 
from glob import glob 
import nibabel as nib 
from tqdm import tqdm

# paths = glob("/home/fanxx/fxx/sdc/luoluo/MMWHS/MMWHS/ct_train/ct*image.nii.gz")
path = "/home/fanxx/fxx/sdc/luoluo/MMWHS/MMWHS/ct_train/ct_train_1009_image.nii.gz"
# for path in tqdm(paths):
ct_path = path 
ct_label_path = path.replace("image","label")
mr_path = path.replace("ct","mr")
mr_label_path = mr_path.replace("image", "label")

ct_label = ants.image_read(ct_label_path)
mr_label = ants.image_read(mr_label_path)

ct = ants.image_read(ct_path)

ans = ants.registration(mr_label , ct_label)
reg_ct_label = ants.apply_transforms(mr_label , ct_label , ans['fwdtransforms'] , "nearestNeighbor")
reg_ct = ants.apply_transforms(mr_label , ct , ans['fwdtransforms'] , "linear")

ants.image_write(reg_ct_label , ct_label_path.replace("ct_train","ct_temp"))
ants.image_write(reg_ct , ct_path.replace("ct_train","ct_temp"))

ct_image = nib.load(ct_path.replace("ct_train","ct_temp"))
affine = ct_image.affine
ct_image = ct_image.get_fdata()
ct_label = nib.load(ct_label_path.replace("ct_train","ct_temp")).get_fdata()
mr_image = nib.load(mr_path).get_fdata()
mr_label = nib.load(mr_label_path).get_fdata()

z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(ct_image[None ,...], axis=0) != 0)
# Add 1 pixel in each side
zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]

nib.save(nib.Nifti1Image(ct_image[zmin:zmax , ymin:ymax , xmin:xmax] , affine) , ct_path.replace("ct_train","ct_crop"))
nib.save(nib.Nifti1Image(ct_label[zmin:zmax , ymin:ymax , xmin:xmax] , affine) , ct_label_path.replace("ct_train","ct_crop"))
nib.save(nib.Nifti1Image(mr_image[zmin:zmax , ymin:ymax , xmin:xmax] , affine) , mr_path.replace("mr_train","mr_crop"))
nib.save(nib.Nifti1Image(mr_label[zmin:zmax , ymin:ymax , xmin:xmax] , affine) , mr_label_path.replace("mr_train","mr_crop"))