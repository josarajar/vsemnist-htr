from types import SimpleNamespace
import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision import datasets, models
import os
import numpy as np
import copy
from tqdm import tqdm




def load_EMNIST(datapath = '~/.pytorch/EMNIST_data/', split ='bymerge', distort='True'):
    
    # Define a transform to load the data as Tensors
    if distort:
        transform = T.Compose([
            T.RandomAffine(degrees=20, translate=(0.2, 0.1), scale=(0.7, 1.15)),
            T.ToTensor(),])
    else:
        transform = T.ToTensor()

    # Download and load the EMNIST training  data
    trainset = datasets.EMNIST(datapath, split=split, download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

    # Download and load the EMNIST test data
    testset = datasets.EMNIST(datapath, split=split, download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

    validloader = copy.deepcopy(trainloader)  # Creates a copy of the object 

    #We take the first 550k images for training
    trainloader.dataset.data = trainloader.dataset.data[:550000,:,:]
    trainloader.dataset.targets = trainloader.dataset.targets[:550000]

    #And the rest for validation
    validloader.dataset.data = validloader.dataset.data[550000:,:,:]
    validloader.dataset.targets = validloader.dataset.targets[550000:]

    return trainloader, validloader, testloader

def generate_random_lengths(min_length=3, max_length=10, n_sequences=1):
    assert max_length>=min_length, 'max_length={} should be >= min_length={}'.format(max_length, min_length)       
    return np.random.randint(min_length, max_length+1, n_sequences) #max-length+1 so length 10 is also included

def create_folder(path, logger):
    if not os.path.isdir(path):
        if logger:
            logger.info('Creating root folder '+ path +'...')
        else:
            print('Creating root folder '+ path +'...') 
        try:
            os.mkdir(path)
        except OSError:
            msg = 'Creation of the directory {} failed'.format(path)             
            
        else:
            msg = 'Successfully created the directory {}'.format(path)
            logger.error(msg) if logger else print(msg)

def save_sample(root, image, seq, idx, dataset='demo'):
    transformToPIL = T.ToPILImage() # transformer for saving Images as PIL    
    if isinstance(root, torch._six.string_classes):
        root = os.path.expanduser(root) # In case we use '~/.pytorch' path for example
    create_folder(root)
    database_path = os.path.join(root,'SEMNIST')
    create_folder(database_path)

    set_path = os.path.join(database_path,dataset+'set') # Check if set is train or test
    create_folder(set_path)
    
    set_img_path = os.path.join(set_path, 'img')
    set_labels_path = os.path.join(set_path, 'labels')
    create_folder(set_img_path)
    create_folder(set_labels_path)
    
    img_path = os.path.join(set_img_path,'semnist_' + dataset + '_' + '{:0>6}'.format(idx) + '.png') 
    seq_path = os.path.join(set_labels_path,'semnist_' + dataset + '_' + '{:0>6}'.format(idx) + '.txt')
    imagePIL = transformToPIL(image.squeeze())  
    imagePIL.save(img_path) # Saving image into path
    seq = ' '.join(list(map(str, list(seq.numpy())))) # Convert seq tensor into string
    
    f = open(seq_path, "w")
    f.write(seq)
    f.close()
    
def generate_sequences(dataloader, sequence_lengths, save = False, root='~/.pytorch/SEMNIST_data/', dataset='demo'):
    num_seq = len(sequence_lengths)
    num_source_characters = len(dataloader)
    num_target_characters = np.sum(num_seq)
    assert num_source_characters>=num_target_characters, 'There are not enough characters in the source dataset to generate {} sequences.'.format(num_seq)
    dataiter = iter(dataloader)
    if not save:
        image_list = []
        label_list = []
    for seq, seq_length in enumerate(tqdm(sequence_lengths)):
        generated_image = []
        generated_seq = []
        for ind in range(seq_length):
            char_image, char_label = dataiter.next()
            generated_image.append(char_image.mT) # Transpose the image because the original are not visually align
            generated_seq.append(char_label)
        generated_image = torch.concat(generated_image,3) # Concatenate (w dim) the characters images to build the sequence image.
        generated_seq = torch.concat(generated_seq)
        if save:
            save_sample(root,generated_image, generated_seq, seq, dataset=dataset)
        else:   
            image_list.append(generated_image)
            label_list.append(generated_seq)
    if not save:
        return image_list, label_list

def generateSEMNIST(semnistpath = '~/.pytorch/SEMNIST_data/', emnistpath = '~/.pytorch/SEMNIST_data/', distort = True, split='bymerge', n_train_sequences = 80000, n_valid_sequences = 20000, n_test_sequences = 16000, logger=None):
    """
    Generates SEMNIST train, valid and test set from EMNIST dataset. The generation procedure generates random sequences of random sizes between 3 and 10 characters.
    """
    trainloader, validloader, testloader = load_EMNIST(emnistpath, split, distort)
    train_seq_lengths = generate_random_lengths(min_length=3, max_length=10, n_sequences=n_train_sequences)
    test_seq_lengths = generate_random_lengths(min_length=3, max_length=10, n_sequences=n_test_sequences)
    valid_seq_lengths = generate_random_lengths(min_length=3, max_length=10, n_sequences=n_valid_sequences)
    generate_sequences(trainloader, train_seq_lengths, save = True, root=semnistpath, dataset='train')
    generate_sequences(testloader, test_seq_lengths, save = True, root=semnistpath, dataset='test')
    generate_sequences(validloader, valid_seq_lengths, save = True, root=semnistpath, dataset='valid')

def check_semnist_dataset_existance(semnistpath, logger):
    """If semnist dataset is incomplete return False, if not dataset exists and is complete and return True
    """
    root = semnistpath
    if isinstance(root, torch._six.string_classes):
        root = os.path.expanduser(root) # In case we use '~/.pytorch' path for example  
    datapath = os.path.join(root,'SEMNIST')
    trainpath = os.path.join(datapath,'trainset')
    testpath = os.path.join(datapath,'testset')
    validpath = os.path.join(datapath,'validset')
    print(root, trainpath, testpath, validpath, datapath)

    
    if not os.path.isdir(root) or not os.path.isdir(datapath) or not os.path.isdir(trainpath) or not os.path.isdir(testpath) or not os.path.isdir(validpath):
        return False
    else:
        return True