import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


class Eventvot(Dataset):
    def __init__(self, data_path, split='train', args=None):
        self.data_path = data_path
        self.data_name = sorted(os.listdir(data_path))
        self.data_ls = sorted(list(map(lambda x:os.path.join(data_path,x),self.data_name)))
        self.fps = 60
        self.crop_size = [128,128]
        self.num_bins = 3
        self.width = 180
        self.height = 180
        self.device = 'cpu'
        self.cond_length = 3
        self.seq_length = 25
    
    def __len__(self):
        return len(self.data_ls)
    
    def __getname__(self): 
        return 'vot' 

    def __getitem__(self, index):
        event_pth = os.path.join(self.data_ls[index],self.data_name[index])+'_voxel_crop.npy'
        event_data = np.load(event_pth)
        max_idx = event_data.shape[0]
        event_seq_length = self.cond_length+self.seq_length
        event_start_idx = np.random.randint(0,max_idx-event_seq_length-1)


        crop_x = np.random.randint(0,self.width-self.crop_size[0])
        crop_y = np.random.randint(0,self.height-self.crop_size[1])

        events_voxel_cat = torch.from_numpy(event_data[event_start_idx:event_start_idx+event_seq_length])
        a = abs(events_voxel_cat.max())
        b = abs(events_voxel_cat.min())
        max_norm = a if a>b else b
        events_voxel_cat = events_voxel_cat/max_norm
        events_voxel_cat = events_voxel_cat[:,:,crop_y:crop_y+self.crop_size[0],crop_x:crop_x+self.crop_size[1]]

        event0 = (events_voxel_cat[:3] +1)/2 
        

        return {"pixel_values": events_voxel_cat[3:], "image": event0, 'dataset': self.__getname__()}

