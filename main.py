from statistics import mode
import sys
import argparse
from generi


def main():
    try:
        args = cmdline_args()
        print(args)
    except:
        print('Try $python  main.py --mode test --arch basic_cnn --model ./models/pretrained/cnn_v1')

    # Load database
    generateSEMNIST(semnistpath = '~/.pytorch/SEMNIST_data/', emnistpath = '~/.pytorch/SEMNIST_data/'  


def cmdline_args():
    # Make parser object
    desc = [
    'Executes training, evaluation or prediction of a given model'
    'over the SEMNIST dataset given set (training, test)'
  
  ]
    descstr = "".join(desc)
    parser = argparse.ArgumentParser(description=__doc__, 
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", help="Mode of execution", choices=['train', 'eval', 'test', 'predict'], required=True)
    parser.add_argument("--arch", help="Model architecture to use", choices=['basic_cnn', 'six_cnn', 'basic_cnn_stn', 'six_cnn_stn'], required=True)
    parser.add_argument("--model", help="Model name for saving in training and loading in evaluation or prediction", required=True)
    parser.add_argument("--semnistpath", help="SEMNIST database root directory", default='~/.pytorch/SEMNIST_data/')
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=20)
    parser.add_argument("--batch", help="Batch size", type=int, default=64)
    parser.add_argument("--lr", help="Static learning rate",type=float, default=0.001)
    parser.add_argument("--emnistpath", help="EMNIST database root directory", default='~/.pytorch/EMNIST_data/')
    
    parser.add_argument("-v", "--verbosity", type=int, choices=[0,1,2], default=0,
                   help="increase output verbosity (default: %(default)s)")
    args = parser.parse_args()              

    return(args)



# Try running with these args
#
# "Hello" 123 --enable
if __name__ == '__main__':
    main()