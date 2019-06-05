import torch 
import torch.nn as nn
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, Dataset
import nrrd
import numpy as np
class CTDataset(Dataset):
  def __init__(self,patient,mode = 'train'):
    super().__init__()    
    image,image_header = nrrd.read('../data/LUNG1-' + patient + '/image.nrrd')
    if mode == 'train':
      images = [image]
      for i in range(3):
        #add gaussian noise
        noise = 400 * np.random.randn(*image_header['sizes'])
        images.append(image + noise)
      images = np.stack(images)
    else:
      images = np.expand_dims(image,axis = 0)
    padding_length = 320 - image_header['sizes'][-1]
    padded = np.zeros((images.shape[0], 512,512,padding_length)) - 1024
    print(images.shape)
    print(padded.shape)
    images = np.concatenate([images,padded],axis = -1)
    images = zoom(images, (1, .5, .5, .5),order = 1)
    images = ((images + 1024) / 4095).astype('float32')
    images[images < 0] = 0
    images[images > 1] = 1
    self.data = images

    
  def __len__(self):
    return len(self.data)
  def __getitem__(self,idx):
    data = self.data[idx]
    return {'data':data}


class RIDERTestDataset(Dataset):
  def __init__(self,patient,mode = 'train'):
    super().__init__()    
    #data_list = []
    counter = 0
    #for patient in ['1129164940','3152132495','2669524182','3160137230']:
    image,image_header = nrrd.read('../data/RIDER-' + patient + '/image_test.nrrd') 
    #images = [image]
    if mode == 'train':
      images = [image]
      for i in range(3):
        #add gaussian noise
        noise = 2 * np.random.randn(*image_header['sizes'])
        images.append(image + noise)
      images = np.stack(images)
    else:
      images = np.expand_dims(image,axis = 0)
    padding_length = 320 - image_header['sizes'][-1]
    padded = np.zeros((images.shape[0], 512,512,padding_length)) - 1024
    print(images.shape)
    print(padded.shape)
    images = np.concatenate([images,padded],axis = -1)
    images = zoom(images, (1, .5, .5, .5),order = 1)
    images = ((images + 1024) / 4095).astype('float32')
    images[images < 0] = 0
    images[images > 1] = 1
    self.data = images
    print(self.data.max())
    #self.labels = np.concatenate(label_list,axis = 0)
    
  def __len__(self):
    return len(self.data)
  def __getitem__(self,idx):
    data = self.data[idx]
    return {'data':data}


class RIDERReTestDataset(Dataset):
  def __init__(self,patient,mode = 'train'):
    super().__init__()    
    #data_list = []
    counter = 0
    #for patient in ['1129164940','3152132495','2669524182','3160137230']:
    image,image_header = nrrd.read('../data/My Drive/RIDER-' + patient + '/image_retest.nrrd') 
    #images = [image]
    if mode == 'train':
      images = [image]
      for i in range(3):
        #add gaussian noise
        noise = 400 * np.random.randn(*image_header['sizes'])
        images.append(image + noise)
      images = np.stack(images)
    else:
      images = np.expand_dims(image,axis = 0)
    padding_length = 320 - image_header['sizes'][-1]
    padded = np.zeros((images.shape[0], 512,512,padding_length)) - 1024
    print(images.shape)
    print(padded.shape)
    images = np.concatenate([images,padded],axis = -1)
    images = zoom(images, (1, .5, .5, .5),order = 1)
    images = ((images + 1024) / 4095).astype('float32')
    images[images < 0] = 0
    images[images > 1] = 1
    self.data = images
    print(self.data.max())
    #self.labels = np.concatenate(label_list,axis = 0)
    
  def __len__(self):
    return len(self.data)
  def __getitem__(self,idx):
    data = self.data[idx]
    return {'data':data}
