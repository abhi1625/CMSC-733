# CMSC-733
Projects for the course CMSC733 - Classical and Deep Learning Approaches for
Geometric Computer Vision

## HW0: Alolhamora!

### Running Phase 1:
- From the terminal cd into the folder `Abhi1625_hw0/Phase1/Code/`
- Run the file Wrapper.py 
The file should run smoothly. In case the file doesn't run check the `Maps` folder in the above directory if its empty, run the file as follows:
```
python Wrapper.py --Maps_flag=True
```
This will take some time as the texton, brightness and color maps will be generated again. The default value of this flag is False. The above step should also be followed in case the filter banks are changed in any way.


### Running Phase 2:
-In the file `Train.py` change the line
```
from Network.*** import CIFAR10Model
```
to 
```
from Network.ResNext import CIFAR10Model
```
inorder to run the ResNext Model. Change this lines for all the models you want to run: 
Network1, ResNet, ResNext, DenseNet. Do the same for `Test.py` file.

## Project 1: MyAutoPano! 

### Running Phase 1:
- Run the file Wrapper.py 
The file should run smoothly to generate the results for the Train Set 2. In case the file doesn't run check that the relative path `../Data/Train/Set2` exists. 

- To run the Wrapper.py for a different data set, use command line argument `BasePath`
```
python Wrapper.py --BasePath= Path/To/Files
```

