from cmath import exp
from torch import nn
from torch import optim
import torch
import time
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import os

def eval_performance(model, dataloader, mode='seqCER', device='cpu'):
    """
    CER = eval_performance(model, dataloader, mode='seqCER')
    
    Computes the accuracy of a given model over a given dataset. There are two modes of computing the performance:
    
        1. `mode='seqCER'` in this case it will compute the Character Error Rate (CER) per sequence, and after that
            it will compute the mean between all the sequences in the dataset.
            
        2. `mode='totalCER'` in this case it will sum up the total number of character errors in the dataset and it
            will divide between the total number of characters in the dataset.
    
    Args:
    
        model: Pytorch model to be evaluated
        
        dataloader: Dataloader object wichi will feed our model.
        
        mode: str, posibilities 'seqCER' and 'totalCER' regarding mean CER per sequence or absolute CER of the
        dataset.
        
    """
        
    loss = 0
    CER = 0

    model.eval()
    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():

        for images,labels, widths, lengths in tqdm(dataloader):
            # Move input and label tensors to the default device
            images, labels = images.to(device), labels.to(device) 
            probs = model.forward(images)
            top_p, top_class = probs.topk(1, dim=1)
            errors = (top_class.squeeze() != labels)
            mask = torch.ones_like(errors)
            for ind, length in enumerate(lengths):
                mask[ind,length:] = 0
            errors*=mask
            if mode == 'totalCER':
                CER += torch.sum(errors.type(torch.FloatTensor))/torch.sum(lengths)
            else:
                CER += torch.mean(torch.sum(errors.type(torch.FloatTensor),1)/lengths)         
        model.train()
        return CER/len(dataloader)
    
class Train_Eval():
    """Training and evaluation method"""
    def __init__(self,model,epochs=100,lr=0.001, modelpath=None, logger=None):
        
        self.model = model
        
        self.lr = lr #Learning Rate
        
        self.optim = optim.Adam(self.model.parameters(), self.lr)
        
        self.epochs = epochs
        
        self.criterion = nn.NLLLoss(reduction='none')             
        
        # A list to store the loss evolution along training
        
        self.loss_during_training = [] 
        
        self.valid_loss_during_training = []
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.logger:
            self.logger.info('Training in device {}.'.format(self.device))

        self.model.to(self.device)

        self.modelpath = modelpath
        
    def trainloop(self,trainloader,validloader):
        
        # Optimization Loop
        
        for e in range(int(self.epochs)):
            
            start_time = time.time()
            
            # Random data permutation at each epoch
            
            running_loss = 0.
            
            self.model.train()
            
            for images, labels, widths, lengths in tqdm(trainloader):
                
                # Move input and label tensors to the default device
                images, labels = images.to(self.device), labels.to(self.device)  
        
                self.optim.zero_grad()  #TO RESET GRADIENTS!
            
                out = self.model.forward(images)

                #Your code here
                loss = self.criterion(out,labels)
                
                mask = torch.ones_like(loss)
                for ind, length in enumerate(lengths):
                    mask[ind,length:] = 0
                 
                # Apply a mask to solve different lengths sequences
                loss = torch.mean(loss*mask)
                
                running_loss += loss.item()

                #Your code here
                loss.backward()
                
                #Your code here
                self.optim.step()
                
                
            self.loss_during_training.append(running_loss/len(trainloader))
            
            # Validation Loss
            
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                
                self.model.eval()
                
                running_loss = 0.
                
                for images,labels, widths, lengths in tqdm(validloader):
                    
                    # Move input and label tensors to the default device
                    images, labels = images.to(self.device), labels.to(self.device)                    
                                   
                    out = self.model.forward(images)

                    #Your code here
                    loss = self.criterion(out,labels)
                    
                    mask = torch.ones_like(loss)
                    
                    for ind, length in enumerate(lengths):
                        mask[ind,length:] = 0
                        
                    # Apply a mask to solve different lengths sequences   
                    loss = torch.mean(loss*mask)  

                    running_loss += loss.item()   
                    
                self.valid_loss_during_training.append(running_loss/len(validloader))    

            # set model back to train mode
            self.model.train()

            if(e % 1 == 0): # Every epoch
                msg = 'Epoch {}. Training loss: {}, Validation loss: {}. Time per epoch: {} seconds'.format( 
                      e, self.loss_during_training[-1], self.valid_loss_during_training[-1], 
                    
                       (time.time() - start_time))
                if self.logger:
                    self.logger.info(msg)
                else:
                    print(msg)
                
            if(e % 10 ==0): # Every 10 epochs):
                if self.modelpath:
                    torch.save(self.model.state_dict(), self.modelpath)
                    msg = 'Model saved in {}'.format(self.modelpath)
                    if self.logger:
                        self.logger.info(msg)
                    else:
                        print(msg)
                msg = 'Epoch {}. Training CER: {}, Validation CER: {}.'.format(e, eval_performance(self. model, trainloader, self.device),
                                                                                      eval_performance(self.model,validloader, self.device))
                
                if self.logger:
                    self.logger.info(msg)
                else:
                    print(msg)


def create_logger(name, log_info_file, error_info_file, console_level=logging.INFO, formatter=None):
    if not formatter:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs info messages
    ifh = logging.FileHandler(log_info_file)
    ifh.setLevel(logging.INFO)
    # create file handler which logs from error messages
    efh = logging.FileHandler(error_info_file)
    efh.setLevel(logging.WARNING)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    ch.setFormatter(formatter)
    ifh.setFormatter(formatter)
    efh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(ifh)
    logger.addHandler(efh)

    return logger

def plot_loss_during_training(loss_during_training, valid_loss_during_training, experiment_root_path, logger):
    plt.plot(loss_during_training)
    plt.plot(valid_loss_during_training)
    plt.title('Loss during training')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    graphdir = os.path.join(experiment_root_path,'graphs')
    try:
        os.mkdir(graphdir)
    except:
        logger.error("Couldn't save loss during training figure in {}".format(graphdir))
    figdir = os.path.join(graphdir, 'loss_during_training')
    plt.savefig(figdir)
    logger.info('Loss during training saved in {}'.format(figdir))
    plt.show()
