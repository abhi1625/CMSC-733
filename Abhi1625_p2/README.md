# CMSC-733
Projects for the course CMSC733 - Classical and Deep Learning Approaches for
Geometric Computer Vision

##Project 2: Face Swap
The description and steps used in each section for swapping faces is briefed in the CMSC733_P2.pdf.The swapping is tested for 3 different transforming techniques: Delaunay Triangulation, Thin Plate Splines and a deep learning approach using PR Net(Position map regression Network).
### Face Swap using Delaunay Triangulation:
- From the terminal cd into the folder `Abhi1625_p2/Code/`
- Run the file Wrapper.py with the following command line arguements:
```
python Wrapper.py --video=Test1.mp4 --swap_img=Rambo.jpg --mode=deln
```
This will run the code for delaunay triangulation to swap the face in `Test1.mp4` with the one in `Rambo.jpg`. 

### Face Swap using Thin Plate Splines:
- From the terminal cd into the folder `Abhi1625_p2/Code/`
- Run the file Wrapper.py with the following command line arguements:
```
python Wrapper.py --video=Test1.mp4 --swap_img=Rambo.jpg --mode=tps
```
This will run the code for Thin Plate Splines to swap the face in `Test1.mp4` with the one in `Rambo.jpg`.

Note: Make sure that all the videos and images ,you want to run the code for, are present in the TestSet2_P2 folder present in the current directory.

### Face Swap using PR Net:
Move to the PRNet folder in the current directory and follow the commands below to run the faceswap pipeline using PR Net.

- Download the PRN trained model at BaiduDrive or GoogleDrive, and put it into Data/net-data

- Convert the video for faceswapping into frames by using ffmpeg:
```	
ffmpeg -r 1 Test1.mp4 -r  "test%0d.png
```
- Feed the path to the swapping files by using command:
```	
python demo_texture.py -i image_path_1 -r image_path_2 -o output_path
```
