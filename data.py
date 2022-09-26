import torch
import os
import numpy as np
from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
from torch.utils.data import Dataset
from einops import rearrange

class HCPDataset(Dataset):
    def __init__(self, args, mode="train") -> None:
        super().__init__()

        self.mode = mode;
        self.data_dir = args.data_dir
        self.b = args.b
        # self.label_select_index = args.label_select_index

        self.cache_path = "/".join(self.data_dir.split("/")[:-2]) + "/cache"
        
        if self.mode == "train":
            self.data_dir = os.path.join(self.data_dir, "train/")
        elif self.mode == "val":
            self.data_dir = os.path.join(self.data_dir, "val/")
        elif self.mode == "test":
            self.data_dir = os.path.join(self.data_dir, "test/")
        print(self.data_dir)

        self.data_paths = [os.path.join(self.data_dir, name, "T1w/Diffusion/") for name in os.listdir(self.data_dir)]

        self.q_mask_path = args.q_mask_path
        self.label_select_index_path = args.label_select_index_path

        self.q_mask = np.load(self.q_mask_path)
        self.label_select_index = np.load(self.label_select_index_path)

        self.data = self.__prepare_data()

    def __prepare_data(self):
        all_data = None
        cache_path = f"{self.cache_path}/{self.mode}.npy"
        if os.path.exists(cache_path):
            all_data = np.load(cache_path)[()]
            print(all_data.shape)
            return all_data
        if not os.path.exists(f"{self.cache_path}"):
            os.makedirs(f"{self.cache_path}")
        if len(self.data_paths) == 0:
            print("检查路径或数据集是否存在")
            return None
        for data_path in self.data_paths:
            data, _ = load_nifti(os.path.join(data_path, "data.nii.gz"))
            # print(data.shape)
            bvals, bvecs = read_bvals_bvecs(os.path.join(data_path, "bvals"), os.path.join(data_path, "bvecs"))
            if self.b >= 0 and self.b in [0, 1000, 2000, 3000]:
                select_b_mask = (bvals <= self.b + 200) * (bvals >= self.b - 200)
                # select_b_mask = np.abs(bvals - self.b) <= 200
                data = data[:, :, :, select_b_mask]
                data = data[:, :, :, self.label_select_index]

            if all_data is None:
                all_data = data
            else:
                all_data = np.append(all_data, data, axis=-2)
        print(all_data.shape)
        np.save(cache_path, all_data)
        return all_data

    def __getitem__(self, index):

        full = self.data[:, :, index, :]

        input_x = full[:, :, self.q_mask]
        label = full[:, :, self.q_mask+1]

        label = torch.from_numpy(label)
        input_x = torch.from_numpy(input_x)

        label = rearrange(label, "h w c -> c h w")
        input_x = rearrange(input_x, "h w c -> c h w")
        
        return input_x.float(), label.float()

    def __len__(self):
        return self.data.shape[-2]
