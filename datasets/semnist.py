import torch
import os
import numpy as np
from torchvision.datasets import VisionDataset
from typing import Any, Callable, List, Optional, Tuple
from PIL import Image

class SEMNIST(VisionDataset):
    def __init__(
        self,
        root: str,
        dataset: str = 'train',
        img_type: str = 'GrayScale',
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        assert os.path.isdir(self.root), 'Couldn`t find {} folder'.format(self.root)
        self._img_path = os.path.join(self.root, 'SEMNIST', dataset + 'set','img')
        self._labels_path = os.path.join(self.root, 'SEMNIST', dataset + 'set','labels')
        assert os.path.isdir(self._img_path), 'Couldn`t find {} folder'.format(self._img_path)
        assert os.path.isdir(self._labels_path), 'Couldn`t find {} folder'.format(self._labels_path)
        self.data_list = os.listdir(self._img_path)
        self.id = lambda id_extended: os.path.splitext(id_extended)[0]
        self.data_list = [self.id(path) for path in self.data_list]
        assert img_type in ['GrayScale', 'RGB'], 'Image type {} not supported'.format(img_type)
        self.img_type = img_type
    
    def __len__(self):
        return len(self.data_list)
   
    def __getitem__(self, index):
        img_path = os.path.join(self._img_path, self.data_list[index]+'.png')
        labels_path = os.path.join(self._labels_path, self.data_list[index]+'.txt')
        if self.img_type == 'RGB':
            img = Image.open(img_path).convert('RGB')
        elif self.img_type == 'GrayScale':
            img = Image.open(img_path)
        f = open(labels_path, "r")
        target = torch.tensor([int(x) for x in f.readline().split(' ')])
        f.close()
        if self.transform is not None:
            img = self.transform(img)
        return img, target

def collate_semnist(data, fixed_width = False, max_width = 280, max_length = 10):
    '''  
    We should build a custom collate_fn rather than using default collate_fn,
    as the width of every image is different and merging images (including padding) 
    is not supported by default. It also pad the labels sequences to have same length. 
    Args:
        data: list of tuple (image, label)
        fixed_width: bool, if the model needs images with fixed width
        max_width: max_width of the model if the model needs it
        max_length: max_length of the model if the model needs it
    Return:
        padded_img - Padded Image, tensor of shape (batch_size, channels, height, padded_width)
        width - Original width of each image(without padding), tensor of shape(batch_size)
        label - Padded sequence, tensor of shape (batch_size, padded_length)
        length - Original length of each sequence(without padding), tensor of shape(batch_size)
    '''

    # sorting is important for usage pack padded sequence when using lstm models. It should be in decreasing order.
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, sequences = zip(*data)  # unzip the images from sequences to edit them more comfortably.
    width = [img.shape[2] for img in images] # save all the original images width
    length = [len(seq) for seq in sequences] # save all the original sequences length
    max_width = max_width if fixed_width else max(width) # Fixed width if the models requires it
    padded_img = torch.zeros(len(images), images[0].shape[0], images[0].shape[1], max_width) # initialize the batch tensor with zeros
    max_length = max_length if fixed_width else max(length) # Fixed length if the model requires it 
    padded_seq = torch.zeros(len(sequences), max_length).long() # initialize the sequence tensor with zeros
    for i, img in enumerate(images):
        end = width[i]
        padded_img[i,:,:,:end] = img # insert the images in the first width part of each place 
    for i, seq in enumerate(sequences):
        end = length[i]
        padded_seq[i,:end] = seq  # insert the sequences in the first length part of each place 
    return padded_img, padded_seq, torch.from_numpy(np.array(width)), torch.from_numpy(np.array(length))

def collate_semnist_fixed_length(data):
    """
    Callable collate_semnist when fixed_lenght is neede by the model. See collate_semnist for arguments and functionality.
    """
    return collate_semnist(data, fixed_width = True)