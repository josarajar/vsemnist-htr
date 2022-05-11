# vsemnist-htr
This is repository is based on the sequence EMNIST dataset. This dataset is buield by the code using the EMNIST dataset, form which it concatenate random charcaters to build sequences of characters of lengthbetween 3 and 10. 

It contains a main code from which you can train different models implemented in the models folder. Some models have been pretrained so you can use to evaluate the test data.

Thera are also two notebooks where there are some explanations about all the procedure, you can take a look to it to better understand the code.

For trying the code, you can execute the following command:

python  main.py --mode test --arch Basic_CNN --model ./models/pretrained/cnn_v1 --executionid my_experiment

The previous command will test the Basic_CNN model saved in models/pretrained. If it is the first time you use the code, the SEMNIST database will be created.

To execute the code, you can build a conda environment using the .yml file in the repository. In this file you can find all the dependencies.

