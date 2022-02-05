# Temporal Alignment Prediction for Supervised Representation Learning and Few-Shot Sequence Classification

Pytorch implementation of the supervised learning of TAP for sequence data.


# Acknowledgments
Part of the coda is adapted from the PyTorch implementation of MoCo which is publicly available at https://github.com/facebookresearch/moco.
Please also check the license and usage there if you want to make use of this code. 


###############################################################################
## Quick Start

### 1. Requirements

Python3, Pytorch, numpy, scipy, sklearn

### 2. Prepare Data

We reorganize sequence data into the ".mat" format by Matlab. 
Training sequence samples and statistics are stored in "trainseqfull.mat";
Testing sequence samples and statistics are stored in "testseqfull.mat".

#### "trainseqfull.mat" contains the following three variables:

     "trainseqfull" is a matrix of size N*T*d, where N is the number of training sequences, T is the length of the longest sequence in the dataset, d the dimension of the element-wise features. It contains all training sequences, sequences shorter than T are padded.

     “trainlengthseqfull” is a matrix of size N*1, it contains the lengths of the corresponding sequences in "trainseqfull".

     "trainlabelseqfull" is a matrix of size N*1, it contains the class labels of the corresponding sequences in "trainseqfull".

#### "testseqfull.mat" contains the following three variables:

     "testseqfull" is a matrix of size N*T*d, where N is the number of testing sequences, T is the length of the longest sequence in the dataset, d the dimension of the element-wise features. It contains all testing sequences, sequences shorter than T are padded.

     “testlengthseqfull” is a matrix of size N*1, it contains the lengths of the corresponding sequences in "testseqfull".

     "testlabelseqfull" is a matrix of size N*1, it contains the class labels of the corresponding sequences in "testseqfull".

For example, if the processed data of the ChaLearn dataset can be saved at

```
/datamat
    /ChaLearn
        trainseqfull.mat
        testseqfull.mat
```

### 3. Usage
Please modify to your own data and checkpoint paths. Please set the arguments according to your own data.

main_supervised_hm.py: the code for training TAP using the triplet loss with hard minming
OPA_supervised_hm.py: the code for the TAP model with the triplet loss, where the prediction network is implemented by a 3-layer CNN

main_supervised_lifted.py: the code for training TAP using the lifted structured loss
OPA_supervised_lifted.py: the code for the TAP model with the lifted structured loss, where the prediction network is implemented by a 3-layer CNN

main_supervised_binomial.py: the code for training TAP using the binomial deviance loss
OPA_supervised_binomial.py: the code for the TAP model with the binomial deviance loss, where the prediction network is implemented by a 3-layer CNN


#### Supervised training
Please refer to the beginning of main_supervised_binomial.py for the arguments.

```bash
python main_supervised_binomial.py -data ./datamat/ChaLearn/ -savedir ./checkpoints/ChaLearn --inputdim 100 -b 64
```

#### Continue training
```bash
python main_supervised_binomial.py -data ./datamat/ChaLearn/ -savedir ./checkpoints/ChaLearn --inputdim 100 --resume ./checkpoints/ChaLearn/checkpoint_0030.pth.tar
```