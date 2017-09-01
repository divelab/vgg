# VGG Networks

Created by [Hongyang Gao](http://eecs.wsu.edu/~hgao/) and [Shuiwang Ji](http://www.eecs.wsu.edu/~sji/) at Washington State University.

## Introduction

## System requirement

#### Programming language
Python 3.5+

#### Python Packages
tensorflow (CPU) or tensorflow-gpu (GPU), numpy, h5py

## Prepare data

In this project, we provided a set of sample datasets for training, validation, and testing.
Please read the code in utils/cifar_util.py and make sure you understand the code.

## Configure the network

All network hyperparameters are configured in main.py.

#### Training

max_step: how many iterations or steps to train

test_interval: how many steps to perform a mini test or validation

save_interval: how many steps to save the model

summary_interval: how many steps to save the summary

learning_rate: learning rate of model

#### Data

data_dir: data directory

train_data: h5 file for training

valid_data: h5 file for validation

test_data: h5 file for testing

batch: batch size

channel: input image channel number

height, width: height and width of input image

#### Debug

logdir: where to store log

modeldir: where to store saved models

sampledir: where to store predicted samples, please add a / at the end for convinience

model_name: the name prefix of saved models

reload_step: where to return training

test_step: which step to test or predict

random_seed: random seed for tensorflow

#### Network architecture

network_depth: how deep of the U-Net including the bottom layer

class_num: how many classes. Usually number of classes plus one for background

start_channel_num: the number of channel for the first conv layer


## Training and Testing

#### Start training

After configure the network, we can start to train. Run
```
python main.py
```
The training of a VGG for classification will start.

#### Training process visualization

We employ tensorboard to visualize the training process.

```
tensorboard --logdir=logdir/ --port=6066
```

The classification results including training and validation accuracies, and the prediction outputs are all available in tensorboard.

#### Testing and prediction

Select a good point to test your model based on validation or other measures.

Fill the test_step in main.py with the checkpoint you want to test, run

```
python main.py --action=test
```

The final output include accuracy and loss.

