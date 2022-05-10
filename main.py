from cgi import test
from cmath import log
from pyexpat import model
import sys
import os
import argparse
from syslog import LOG_INFO
from datasets.generate_semnist import generateSEMNIST, check_semnist_dataset_existance
from utils.tools import create_logger, eval_performance
import datetime
from datasets.semnist import get_SEMNIST_loaders
from models import htrModels
import torch
from utils.tools import Train_Eval, plot_loss_during_training

def main():
    """
    Main function of vsemnist-htr project.
    """

    # 1. Let's check the command line arguments, if something is missing or misspeled, it will finish the execution.
    try:
        args = cmdline_args()
        print(args)
    except:
        print('Try $python  main.py --mode test --arch Basic_CNN --model ./models/pretrained/cnn_v1 --executionid my_experiment')
        sys.exit(1)

    # 2. Initialize the log files (info and error). If something is wrong, the code will stop.
    try:
        if not os.path.isdir(args.outputpath):
            os.mkdir(args.outputpath)
        executionfilesdir = os.path.join(args.outputpath, args.executionid)
        os.mkdir(executionfilesdir)
        log_info_file = os.path.join(executionfilesdir,'info.log')
        error_info_file = os.path.join(executionfilesdir, 'error.log')
        logger = create_logger(args.executionid, log_info_file, error_info_file)
    except:
        print("Can't acces to experiment folder {}.".format(executionfilesdir))
        sys.exit(1)

    # 3. Load database in case it doesn't exist in path given in args.semnistpath.
    logger.info('Checking if SEMNIST dataset is in {} path'.format(args.semnistpath))
    if not check_semnist_dataset_existance(semnistpath = args.semnistpath, logger=logger):
        logger.info("SEMNIST dataset doesn't exists at path {}. Creating dataset...".format(args.semnistpath))
        generateSEMNIST(semnistpath = args.semnistpath, emnistpath = args.emnistpath, logger=logger)
    else:
        logger.info("SEMNIST has been already created in {}. Loading datasset...".format(args.semnistpath))
    
    # Instantiating dataloaders
    logger.info("Initializing dataloaders.")
    trainloader, validloader, testloader = get_SEMNIST_loaders(args.semnistpath, args.batch)

    # 4. Instantiate the model we want to train, evaluate or get predictions.
    arch = args.arch 
    if arch == 'Basic_CNN':
        network = htrModels.Basic_CNN(height=28, nlabels=47,prob=args.dropout)
    elif arch == 'CNN_6':
        network = htrModels.CNN_6(height=28, nlabels=47, prob=args.dropout)
    elif arch == 'Basic_CNN_STN':
        network == htrModels.Basic_CNN_STN(height=28, nlabels=47, prob=args.dropout)
    else:
        logger.info("Model {} doesn't implemented yet".format(arch))
        logger.error("Aborting since model {} doesn't implemented yet... ".format(arch))
        sys.exit(0)
    
    # Let's load the model in case it has been already pretrained 
    if args.pretrained or args.mode in ['eval', 'test', 'predict']:
        try:
            modelpath = args.model
            network.load_state_dict(torch.load(modelpath))
        except:
            logger.error('Model {} not found. You should train the model from scratch. Set --pretrained False Aborting...'.format(modelpath))
            sys.exit(1)

    if args.mode == 'train':
        modelpath = args.model
        trainablemodel = Train_Eval(network, args.epochs, args.lr, modelpath = modelpath, logger=logger)
        try:
            trainablemodel.trainloop(trainloader, validloader)
            
            torch.save(trainablemodel.model.state_dict(), modelpath)
            plot_loss_during_training(trainablemodel.loss_during_training, trainablemodel.valid_loss_during_training, executionfilesdir, logger)
        except Exception as e:
            logger.error('Some error occur during training. Aborting...\n' + e)
            sys.exit(1)
        

    if args.mode in ['eval', 'test']:
        try:
            testcer = eval_performance(network, testloader)
            logger.info('Test CER computed over model {} with architecture {}: {}'.format(modelpath, arch, testcer))
        except Exception as e:
            logger.error("Couldn't eval performance in model {}. Something wrong: \n".format(modelpath) + e)
            sys.exit(1)


def cmdline_args():
    # Make parser object

    parser = argparse.ArgumentParser(description=__doc__, 
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", help="Mode of execution", choices=['train', 'eval', 'test', 'predict'], required=True)
    parser.add_argument("--arch", help="Model architecture to use", choices=['Basic_CNN', 'CNN_6', 'Basic_CNN_STN', 'CNN_6_STN'], required=True)
    parser.add_argument("--model", help="Model name for saving in training and loading in evaluation or prediction", required=True)
    parser.add_argument("--executionid", help="Identifier of the current execution", type=str, required=True)
    parser.add_argument("--pretrained", help="Indicate if load pretrained model", type=bool, default=False)
    parser.add_argument("--outputpath", help="Root path for outputs (logs, graphs...) of the experiments", type=str, default='experiments')
    parser.add_argument("--semnistpath", help="SEMNIST database root directory", default='~/.pytorch/SEMNIST_data/')
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=20)
    parser.add_argument("--batch", help="Batch size", type=int, default=64)
    parser.add_argument("--dropout", help="Dropout probability",type=float, default=0.2)
    parser.add_argument("--lr", help="Static learning rate",type=float, default=0.001)
    parser.add_argument("--emnistpath", help="EMNIST database root directory", default='~/.pytorch/EMNIST_data/')
    
    parser.add_argument("-v", "--verbosity", type=int, choices=[0,1,2], default=0,
                   help="increase output verbosity (default: %(default)s)")
    args = parser.parse_args()        

    print('Setting an unique identifier for the execution')
    now = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")      
    args.executionid = args.executionid + '-' + str(now)    
    print(args.executionid)
    return(args)



# Try running with these args
#
# "Hello" 123 --enable
if __name__ == '__main__':
    main()