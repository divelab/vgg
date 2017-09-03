# VGG Networks

This is the first tutorial project for new students in our DIVE lab. The whole project includes two parts. If you finish this one earlier, you can ask for the second one.

Created by [Hongyang Gao](http://eecs.wsu.edu/~hgao/), [Lei Cai](http://www.eecs.wsu.edu/~lcai/), [Zhengyang Wang](http://www.eecs.wsu.edu/~zwang6/) and [Shuiwang Ji](http://www.eecs.wsu.edu/~sji/) at Washington State University.

## Introduction

The project aims to help you learn basic convolutional neural networks and study how to use our servers for implementing them. It is based on a real-world dataset named CIFAR-10, which includes images with 10-class labels. You will use CNNs to do image classification task. This repository includes tensorflow code of the VGG networks along with data processing utils.

You are supposed to complete all the requirements and do a presentation by the end of September. In the presentation, you should demonstrate your learning process and will be questioned about your understanding by others.

During the project, you should at least study the first half of the popular [CS231n](http://cs231n.stanford.edu/) course on your own.

It is not an easy project for starters. Feel free to ask any of us for help. Good luck!

## Project requirements

1. Setup your account on the servers. Each of you will use one GPU on the DIVE1 server for this project.

- Build your own Python environment with tensorflow(GPU) and other packages installed according to the system requirement. Since you do not have the admin access to the server, this step is necessary. I suggest you build your environment under the tempspace folders so later you can directly use it on the other two servers.

- Download this repository to your folder on the server. If you are familiar with Github, you can use the git commands. But this is not mandatory.

2. Download the CIFAR-10 dataset. As only the training subset and the validation subset are available, you should use the validation subset as the testing set. For development/validation, divide the training set into a new training set and a validation set. The testing set is used only in testing and must not be used in any step of training. You will always report the performance on the testing set.

- To develop a good habit, organize all your folders with appropriate names.

- The data link is provided in cifar_util.py. You can try the code in cifar_util.py or download manually.

- As the code will read h5 files as input, you should pre-process the dataset into h5 files. The h5 files in the dataset folder are just samples for testing the code. You must do this step by yourself. Refer to the code in cifar_util.py as an example.

- (Optional) The h5 files will be directly fed into the neural networks. You can explore data augmentation and normalization when you generate the h5 files. These techniques will help improve the performance of the models.

3. Run the VGG networks.

- Read all the code and make sure that you understand it thoroughly. You will be asked questions about every detail in your presentation.

- Read the next section for how to configure the network. Note that the default setting is not fully correct. For example, the valid_data should not be the same as test_data.

- Run the code, change hyperparameters for training and testing and report your results. Use tensorboard for development. 

- Run testing/prediction. The higher the testing accuracy is, the better!

4. (Optional) Explore other convolutional neural networks for this image classification task.

- Try the famous ResNet and report the performance. You can find code for ResNet online. But be prepared for questions about every piece of code that you use.

5. Presentation. You are not required to write a report.

- Show how you do every step.

- Report your results.

- Q&A.

## System requirement

#### Programming language
Python 3.5+

#### Python Packages
tensorflow (CPU) or tensorflow-gpu (GPU), numpy, h5py

## Configure the network

All network hyperparameters are configured in main.py.

#### Training

max_step: how many iterations or steps to train

test_interval: how many steps to perform a mini validation

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

sampledir: where to store predicted samples, please add a / at the end for convenience

model_name: the name prefix of saved models

reload_step: where to return training

test_step: which step to test or predict

random_seed: random seed for tensorflow

#### Network architecture

network_depth: how deep of the network including the bottom layer

class_num: how many classes. Usually number of classes plus one for background

start_channel_num: the number of channel for the first conv layer


## Training and Testing

#### Start training

After configuring the network, we can start to train. Run
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

The final output includes accuracy and loss.

