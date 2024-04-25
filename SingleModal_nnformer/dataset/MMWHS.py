import pathlib
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Dataset

# from config import get_brats_folder, get_test_brats_folder
from dataset.image_utils import pad_or_crop_image, irm_min_max_preprocess, zscore_normalise , normalize
# from image_utils import pad_or_crop_image, irm_min_max_preprocess, zscore_normalise , normalize
from glob import glob
import matplotlib.pyplot as plt

class MMWHS(Dataset):
    def __init__(self, patients_dir, benchmarking=False, training=True, data_aug=False,
                 no_seg=False, normalisation="minmax") -> None:
        super().__init__()
        self.benchmarking = benchmarking
        self.normalisation = normalisation
        self.data_aug = data_aug
        self.training = training
        self.datas = []
        self.validation = no_seg
        self.patterns = ["mr", "ct"]
        self.label_value = [205,420,500 ,550 ,600 ,820 ,850]
        
        # CT_images = glob(patients_dir + "/ct_*_image.nii.gz")
        CT_images = patients_dir
        for path in CT_images:
            patient_id = path.split("_")[-2]
            ct_path = path 
            ct_label_path = ct_path.replace("image","label")
            mr_path = path.replace("ct","mr")
            mr_label_path = mr_path.replace("image" , "label")

            self.datas.append({
                "id":patient_id,
                "ct":ct_path,
                "ct_label" : ct_label_path,
                "mr": mr_path,
                "mr_label" : mr_label_path
            })
    
    def __getitem__(self, idx):
        _patient = self.datas[idx]
        patient_image = {key: self.load_nii(_patient[key]) for key in _patient if key not in ["id", "ct_label" , "mr_label"]}
        if _patient["ct_label"] is not None:
            patient_ct_label = self.load_nii(_patient["ct_label"])
        if _patient["mr_label"] is not None :
            patient_mri_label = self.load_nii(_patient["mr_label"])
            
        if self.normalisation == "minmax":
            patient_image = {key: normalize(patient_image[key]) for key in patient_image}
        elif self.normalisation == "zscore":
            patient_image = {key: zscore_normalise(patient_image[key]) for key in patient_image}
        
        

        target_shape = (256, 256, 256)
        for key,value in patient_image.items():
            
            # 将图像下采样至目标尺寸
            downsampled_image = F.interpolate(torch.Tensor(value).unsqueeze(0).unsqueeze(0), size=target_shape, mode='trilinear')
            if key == 'mr':
                downsampled_image = torch.flip(downsampled_image , dim = 2)
                downsampled_image = torch.permute(0,1,3,2,4)
            patient_image[key] = downsampled_image.squeeze(0).squeeze(0).numpy()

        patient_image = np.stack([patient_image[key] for key in patient_image])
        if _patient["ct_label"] is not None:
            patient_ct_label = self.label_to_one_hot(patient_ct_label)
            # 将图像下采样至目标尺寸
            patient_ct_label = F.interpolate(torch.Tensor(patient_ct_label).unsqueeze(0), size=target_shape, mode='nearest')
            patient_ct_label = patient_ct_label.squeeze(0).numpy()
        else:
            patient_ct_label = np.zeros(patient_image.shape)  # placeholders, not gonna use it
            et_present = 0
        
        if _patient["mr_label"] is not None:
            patient_mri_label = self.label_to_one_hot(patient_mri_label)
            patient_mri_label = F.interpolate(torch.Tensor(patient_mri_label).unsqueeze(0), size=target_shape, mode='nearest')
            
            patient_mri_label = torch.flip(patient_mri_label , dim = 2)
            patient_mri_label = torch.permute(0,1,3,2,4)
            patient_mri_label = patient_mri_label.squeeze(0).numpy()
        else:
            patient_mri_label = np.zeros(patient_image.shape)  # placeholders, not gonna use it
            
            et_present = 0
        patient_label = np.concatenate([patient_ct_label,patient_mri_label])

        if self.training:
            # Remove maximum extent of the zero-background to make future crop more useful
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            # Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]

            
            # default to 128, 128, 128 64, 64, 64 32, 32, 32
            patient_image, patient_label = pad_or_crop_image(patient_image, patient_label, target_size=(128, 128, 128))
        else:
           
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            # Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]

        patient_image, patient_label = patient_image.astype("float16"), patient_label.astype("bool")
        patient_image, patient_label = [torch.from_numpy(x) for x in [patient_image, patient_label]]
        return dict(patient_id=_patient["id"],
                    image=patient_image, label=patient_label[7:],
                    seg_path=str(_patient["ct_label"]) if not self.validation else str(_patient["ct"]),
                    crop_indexes=((zmin, zmax), (ymin, ymax), (xmin, xmax)),
                    et_present=0,
                    supervised=True,
                    )

    @staticmethod
    def load_nii(path_folder):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path_folder)))

    def __len__(self):
        return len(self.datas)

    def label_to_one_hot(self,label:np.ndarray):
        labels = []
        for i in self.label_value:
            label_copy = label.copy()
            label_copy[label_copy != i] = 0
            label_copy[label_copy == i] = 1
            labels.append(label_copy)
        return np.array(labels,dtype=np.int16)



class MMWHS_noCrop(Dataset):
    def __init__(self, patients_dir, benchmarking=False, training=True, data_aug=False,
                 no_seg=False, normalisation="minmax") -> None:
        super().__init__()
        self.benchmarking = benchmarking
        self.normalisation = normalisation
        self.data_aug = data_aug
        self.training = training
        self.datas = []
        self.validation = no_seg
        self.patterns = ["mr", "ct"]
        self.label_value = [205,420,500 ,550 ,600 ,820 ,850]
        
        # CT_images = glob(patients_dir + "/ct_*_image.nii.gz")
        CT_images = patients_dir
        for path in CT_images:
            patient_id = path.split("_")[-2]
            ct_path = path 
            ct_label_path = ct_path.replace("image","label")
            mr_path = path.replace("ct","mr")
            mr_label_path = mr_path.replace("image" , "label")

            self.datas.append({
                "id":patient_id,
                "ct":ct_path,
                "ct_label" : ct_label_path,
                "mr": mr_path,
                "mr_label" : mr_label_path
            })
    
    def __getitem__(self, idx):
        _patient = self.datas[idx]
        patient_image = {key: self.load_nii(_patient[key]) for key in _patient if key not in ["id", "ct_label" , "mr_label"]}
        if _patient["ct_label"] is not None:
            patient_ct_label = self.load_nii(_patient["ct_label"])
        if _patient["mr_label"] is not None :
            patient_mri_label = self.load_nii(_patient["mr_label"])
            
        if self.normalisation == "minmax":
            patient_image = {key: normalize(patient_image[key]) for key in patient_image}
        elif self.normalisation == "zscore":
            patient_image = {key: zscore_normalise(patient_image[key]) for key in patient_image}
        
        

        target_shape = (128, 128, 128)
        for key,value in patient_image.items():
            

            if key == 'mr':
                value = np.flip(value, axis = 0)
                value = value.transpose(1,0,2)
                value = np.ascontiguousarray(value)
            # 将图像下采样至目标尺寸
            downsampled_image = F.interpolate(torch.Tensor(value).unsqueeze(0).unsqueeze(0), size=target_shape, mode='trilinear')
            # if key == 'mr':
            #     downsampled_image = torch.flip(downsampled_image , dims = 2)
            #     downsampled_image = torch.permute(0,1,3,2,4)
            patient_image[key] = downsampled_image.squeeze(0).squeeze(0).numpy()

        patient_image = np.stack([patient_image[key] for key in patient_image])
        if _patient["ct_label"] is not None:
            patient_ct_label = self.label_to_one_hot(patient_ct_label)
            # 将图像下采样至目标尺寸
            patient_ct_label = F.interpolate(torch.Tensor(patient_ct_label).unsqueeze(0), size=target_shape, mode='nearest')
            patient_ct_label = patient_ct_label.squeeze(0).numpy()
        else:
            patient_ct_label = np.zeros(patient_image.shape)  # placeholders, not gonna use it
            et_present = 0
        
        if _patient["mr_label"] is not None:
            
            patient_mri_label = np.flip(patient_mri_label, axis = 0)
            patient_mri_label = patient_mri_label.transpose(1,0,2)
            patient_mri_label = np.ascontiguousarray(patient_mri_label)

            patient_mri_label = self.label_to_one_hot(patient_mri_label)
            
            patient_mri_label = F.interpolate(torch.Tensor(patient_mri_label).unsqueeze(0), size=target_shape, mode='nearest')
            
            # patient_mri_label = torch.flip(patient_mri_label , dims = 2)
            # patient_mri_label = torch.permute(0,1,3,2,4)
            patient_mri_label = patient_mri_label.squeeze(0).numpy()
        else:
            patient_mri_label = np.zeros(patient_image.shape)  # placeholders, not gonna use it
            
            et_present = 0
        patient_label = np.concatenate([patient_ct_label,patient_mri_label])

        if self.training:
            # Remove maximum extent of the zero-background to make future crop more useful
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            # Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            

            
            # default to 128, 128, 128 64, 64, 64 32, 32, 32
            patient_image, patient_label = pad_or_crop_image(patient_image, patient_label, target_size=(128, 128, 128))
        else:
           
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            # Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            

        patient_image, patient_label = patient_image.astype("float16"), patient_label.astype("bool")
        patient_image, patient_label = [torch.from_numpy(x) for x in [patient_image, patient_label]]
        return dict(patient_id=_patient["id"],
                    image=patient_image, label=patient_label[:7],
                    seg_path=str(_patient["ct_label"]) if not self.validation else str(_patient["ct"]),
                    crop_indexes=((zmin, zmax), (ymin, ymax), (xmin, xmax)),
                    et_present=0,
                    supervised=True,
                    )

    @staticmethod
    def load_nii(path_folder):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path_folder)))

    def __len__(self):
        return len(self.datas)

    def label_to_one_hot(self,label:np.ndarray):
        labels = []
        for i in self.label_value:
            label_copy = label.copy()
            label_copy[label_copy != i] = 0
            label_copy[label_copy == i] = 1
            labels.append(label_copy)
        return np.array(labels,dtype=np.int16)



class MMWHS_noCrop_Augment(Dataset):
    def __init__(self, patients_dir, benchmarking=False, training=True, data_aug=False,
                 no_seg=False, normalisation="minmax",transform = None) -> None:
        super().__init__()
        self.benchmarking = benchmarking
        self.normalisation = normalisation
        self.data_aug = data_aug
        self.training = training
        self.datas = []
        self.validation = no_seg
        self.patterns = ["mr", "ct"]
        self.label_value = [205,420,500 ,550 ,600 ,820 ,850] # 应该把0也考虑进来啊 那么就是8类
        self.transform = transform
        # CT_images = glob(patients_dir + "/ct_*_image.nii.gz")
        CT_images = patients_dir
        for path in CT_images:
            patient_id = path.split("_")[-2]
            ct_path = path 
            ct_label_path = ct_path.replace("image","label")
            mr_path = path.replace("ct","mr")
            mr_label_path = mr_path.replace("image" , "label")

            self.datas.append({
                "id":patient_id,
                "ct":ct_path,
                "ct_label" : ct_label_path,
                "mr": mr_path,
                "mr_label" : mr_label_path
            })
    
    def __getitem__(self, idx):
        _patient = self.datas[idx]
        patient_image = {key: self.load_nii(_patient[key]) for key in _patient if key not in ["id", "ct_label" , "mr_label"]}
        if _patient["ct_label"] is not None:
            patient_ct_label = self.load_nii(_patient["ct_label"])
        if _patient["mr_label"] is not None :
            patient_mri_label = self.load_nii(_patient["mr_label"])
            
        if self.normalisation == "minmax":
            patient_image = {key: normalize(patient_image[key]) for key in patient_image}
        elif self.normalisation == "zscore":
            patient_image = {key: zscore_normalise(patient_image[key]) for key in patient_image}
        
        

        target_shape = (128, 128, 128)
        for key,value in patient_image.items():
            

            # if key == 'mr':
            #     value = np.flip(value, axis = 0)
            #     value = value.transpose(1,0,2)
            #     value = np.ascontiguousarray(value)
            # 将图像下采样至目标尺寸
            downsampled_image = F.interpolate(torch.Tensor(value).unsqueeze(0).unsqueeze(0), size=target_shape, mode='trilinear')
            # if key == 'mr':
            #     downsampled_image = torch.flip(downsampled_image , dims = 2)
            #     downsampled_image = torch.permute(0,1,3,2,4)
            patient_image[key] = downsampled_image.squeeze(0).squeeze(0).numpy()

        patient_image = np.stack([patient_image[key] for key in patient_image])
        if _patient["ct_label"] is not None:
            patient_ct_label = self.label_to_one_hot(patient_ct_label)
            # 将图像下采样至目标尺寸
            patient_ct_label = F.interpolate(torch.Tensor(patient_ct_label).unsqueeze(0), size=target_shape, mode='nearest')
            patient_ct_label = patient_ct_label.squeeze(0).numpy()
        else:
            patient_ct_label = np.zeros(patient_image.shape)  # placeholders, not gonna use it
            et_present = 0
        
        if _patient["mr_label"] is not None:
            
            # patient_mri_label = np.flip(patient_mri_label, axis = 0)
            # patient_mri_label = patient_mri_label.transpose(1,0,2)
            # patient_mri_label = np.ascontiguousarray(patient_mri_label)

            patient_mri_label = self.label_to_one_hot(patient_mri_label)
            
            patient_mri_label = F.interpolate(torch.Tensor(patient_mri_label).unsqueeze(0), size=target_shape, mode='nearest')
            
            # patient_mri_label = torch.flip(patient_mri_label , dims = 2)
            # patient_mri_label = torch.permute(0,1,3,2,4)
            patient_mri_label = patient_mri_label.squeeze(0).numpy()
        else:
            patient_mri_label = np.zeros(patient_image.shape)  # placeholders, not gonna use it
            
            et_present = 0
        patient_label = np.concatenate([patient_ct_label,patient_mri_label])

        if self.training:
            # Remove maximum extent of the zero-background to make future crop more useful
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            # Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            

            
            # default to 128, 128, 128 64, 64, 64 32, 32, 32
            patient_image, patient_label = pad_or_crop_image(patient_image, patient_label, target_size=(128, 128, 128))
        else:
           
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            # Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            

        patient_image, patient_label = patient_image.astype("float16"), patient_label.astype("bool")
        patient_image, patient_label = [torch.from_numpy(x) for x in [patient_image, patient_label]]
        
        if self.transform is not None :
        
            return self.transform(dict(patient_id=_patient["id"],
                        image=patient_image[:1], label=patient_label[:8],
                        seg_path=str(_patient["ct_label"]) if not self.validation else str(_patient["ct"]),
                        crop_indexes=((zmin, zmax), (ymin, ymax), (xmin, xmax)),
                        et_present=0,
                        supervised=True,
                        ))
        else :
            return dict(patient_id=_patient["id"],
                        image=patient_image[:1], label=patient_label[:8],
                        seg_path=str(_patient["ct_label"]) if not self.validation else str(_patient["ct"]),
                        crop_indexes=((zmin, zmax), (ymin, ymax), (xmin, xmax)),
                        et_present=0,
                        supervised=True,
                        )

    @staticmethod
    def load_nii(path_folder):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path_folder)))

    def __len__(self):
        return len(self.datas)

    def label_to_one_hot(self,label:np.ndarray):
        labels = []
        label_copy = label.copy()
        label_copy[label_copy != 0] = 1
        labels.append(1-label_copy)

        for i in self.label_value:
            label_copy = label.copy()
            label_copy[label_copy != i] = 0
            label_copy[label_copy == i] = 1
            labels.append(label_copy)
        return np.array(labels,dtype=np.int16)


def get_datasets(seed, on="train", fold_number=0, normalisation="minmax"):
    
    patients_dir = sorted(glob("/home/fanxx/fxx/sdc/luoluo/MMWHS/MMWHS/ct_train/" + "/ct_*_image.nii.gz"))

    kfold = KFold(5, shuffle=True, random_state=seed)
    splits = list(kfold.split(patients_dir))
    train_idx, val_idx = splits[fold_number]
    len_val = len(val_idx)
    val_index = val_idx[: len_val//2]
    test_index = val_idx[len_val // 2 :]

    train = [patients_dir[i] for i in train_idx]
    val = [patients_dir[i] for i in val_index]
    test = [patients_dir[i] for i in test_index]

    # return patients_dir
    train_dataset = MMWHS(train, training=True,
                          normalisation=normalisation)
    val_dataset = MMWHS(val, training=False, data_aug=False,
                        normalisation=normalisation)
    bench_dataset = MMWHS(test, training=False, benchmarking=True,
                          normalisation=normalisation)
    return train_dataset, val_dataset, bench_dataset


def get_datasets_noPad(seed, on="train", fold_number=0, normalisation="minmax"):
    
    patients_dir = sorted(glob("/home/fanxx/fxx/sdc/luoluo/MMWHS/MMWHS/ct_train/" + "/ct_*_image.nii.gz"))

    kfold = KFold(5, shuffle=True, random_state=seed)
    splits = list(kfold.split(patients_dir))
    train_idx, val_idx = splits[fold_number]
    len_val = len(val_idx)
    val_index = val_idx[: len_val//2]
    test_index = val_idx[len_val // 2 :]

    train = [patients_dir[i] for i in train_idx]
    val = [patients_dir[i] for i in val_index]
    test = [patients_dir[i] for i in test_index]

    # return patients_dir
    train_dataset = MMWHS_noCrop(train, training=True,
                          normalisation=normalisation)
    val_dataset = MMWHS_noCrop(val, training=False, data_aug=False,
                        normalisation=normalisation)
    bench_dataset = MMWHS_noCrop(test, training=False, benchmarking=True,
                          normalisation=normalisation)
    return train_dataset, val_dataset, bench_dataset



def get_datasets_Aug(seed, on="train", fold_number=0, normalisation="minmax",train_transforms = None ,val_transforms = None ):
    
    patients_dir = sorted(glob("/home/fanxx/fxx/sdc/luoluo/MMWHS/MMWHS/ct_crop/" + "/ct_*_image.nii.gz"))

    kfold = KFold(5, shuffle=True, random_state=seed)
    splits = list(kfold.split(patients_dir))
    train_idx, val_idx = splits[fold_number]
    len_val = len(val_idx)
    val_index = val_idx[: len_val//2]
    test_index = val_idx[len_val // 2 :]

    train = [patients_dir[i] for i in train_idx]
    val = [patients_dir[i] for i in val_index]
    test = [patients_dir[i] for i in test_index]

    # return patients_dir
    train_dataset = MMWHS_noCrop_Augment(train, training=True,
                          normalisation=normalisation, transform= train_transforms)
    val_dataset = MMWHS_noCrop_Augment(val, training=False, data_aug=False,
                        normalisation=normalisation, transform= val_transforms)
    bench_dataset = MMWHS_noCrop_Augment(test, training=False, benchmarking=True,
                          normalisation=normalisation, transform= val_transforms)
    return train_dataset, val_dataset, bench_dataset



def get_test_datasets(seed, on="train", fold_number=0, normalisation="minmax"):
    patients_dir = sorted(glob("/home/fanxx/fxx/sdc/luoluo/MMWHS/mmwhs_test/test_ct/" + "/ct_*_image.nii.gz"))

    bench_dataset = MMWHS(patients_dir, training=False, benchmarking=True,
                          normalisation=normalisation)
    return bench_dataset

def plot_3d(image,name):
    plt.subplot(131)
    plt.imshow(image[:,:,128],'gray')
    plt.subplot(132)
    plt.imshow(image[:,128,:],'gray')
    plt.subplot(133)
    plt.imshow(image[128,:,:],'gray')
    plt.savefig(f"{name}.png")

if __name__ == "__main__":
    train_dataset, val_dataset, bench_dataset = get_datasets_noPad(seed = 1234)
    for batch in train_dataset:
        image , label = batch['image'] , batch['label']
        print(image.shape)
        break

