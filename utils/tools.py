from torch import nn
from torch import optim
import torch
import time
from tqdm import tqdm


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
    def __init__(self,model,epochs=100,lr=0.001):
        
        self.model = model
        
        self.lr = lr #Learning Rate
        
        self.optim = optim.Adam(self.model.parameters(), self.lr)
        
        self.epochs = epochs
        
        self.criterion = nn.NLLLoss(reduction='none')             
        
        # A list to store the loss evolution along training
        
        self.loss_during_training = [] 
        
        self.valid_loss_during_training = []
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        
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

                print('Epoch {}. Training loss: {}, Validation loss: {}. Time per epoch: {} seconds'.format( 
                      e, self.loss_during_training[-1], self.valid_loss_during_training[-1], 
                    
                       (time.time() - start_time)))
                
            if(e % 10 ==0): # Every 10 epochs):
                print('Epoch {}. Training CER: {}, Validation CER: {}.'.format(e, eval_performance(self. model, trainloader, self.device),
                                                                                      eval_performance(self.model,validloader, self.device)))