# CMSC-733
Project 1: MyAutoPano! for the course CMSC-733

##Running Phase 1:
- Run the file Wrapper.py 
The file should run smoothly to generate the results for the Train Set 2. In case the file doesn't run check that the relative path `../Data/Train/Set2` exists. 

- To run the Wrapper.py for a different data set, use command line argument `BasePath`
```
python Wrapper.py --BasePath= Path/To/Files
```

##Running Phase 2:
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
