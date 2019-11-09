## HW0: Alolhamora!
In this homework, a simplified version of [probability of boundary detection algorithm](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/papers/amfm_pami2010.pdf) was developed, which finds boundaries by examining brightness, color, and texture information across multiple scales (different sizes of objects/image). The output of the algorithm is a per-pixel probability of the boundary detected. The simplified algorithm performs much better when compared to classical edge detection algorithms like [Canny](https://ieeexplore.ieee.org/document/4767851) and [Sobel](https://en.wikipedia.org/wiki/Sobel_operator). The original image and the output of the implemented pipeline is shown below:
<!--![original](Abhi1625_hw0/Phase1/BSDS500/Images/8.jpg)![pblite](Abhi1625_hw0/Phase1/Code/8/PbLite_8canny=0.1.png) -->

<img src="Phase1/BSDS500/Images/8.jpg" align="center" alt="Your image title" width="400"/> <img src="Phase1/Code/8/PbLite_8canny=0.1.png" align="center" alt="Your image title" width="400"/>
### Running Phase 1:
- From the terminal cd into the folder `Abhi1625_hw0/Phase1/Code/`
- Run the file Wrapper.py 
The file should run smoothly. In case the file doesn't run check the `Maps` folder in the above directory if its empty, run the file as follows:
```
python Wrapper.py --Maps_flag=True
```
This will take some time as the texton, brightness and color maps will be generated again. The default value of this flag is False. The above step should also be followed in case the filter banks are changed in any way.


### Running Phase 2:
For phase 2 of this homework, multiple neural network architectures were implemented. Various criterion like number of parameters, train and test set accuracies were compared for each network architecture and detailed analysis of why one architecture works better than another one was provided.

- In the file `Train.py` change the line
```
from Network.*** import CIFAR10Model
```
to 
```
from Network.ResNext import CIFAR10Model
```
inorder to run the ResNext Model. Change this lines for all the models you want to run: 
Network1, ResNet, ResNext, DenseNet. Do the same for `Test.py` file.


