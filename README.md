# Badminton-Players-Tracking--YOLOv11
## Project Overview
The project uses Ultralytics' YOLOv11 to track the positions of the athletes in a badminton match video, plot their movement trajectories, and calculate the actual distance traveled by the athletes.
## Features
· Player Detection and tracking  
· Player Trajectory Mapping  
· Running Distance Calculation  
## Technologies Used
· Python  
· OpenCV  
· YOLOv11(for object detection and tracking)  
· NumPy  
· Pandas  
· Pytorch(pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116)
## Installation
git clone https://github.com/cxxmarvinbest/Badminton-Players-Tracking--YOLOv11.git  
git clone https://github.com/ultralytics/ultralytics.git
## Usage
1.Install Miniconda on the C drive, and make sure the installation directory does not contain any Chinese characters(https://mirroes.tuna.tsinghua.edu.cn/anaconda/miniconda/)  
2.Create a virtual environment(conda create -n yolov11 python=3.8)  
3.Configure domestic mirrors(https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)  
4.Use `extract.ipynb` to split `test.mp4` into individual images based on a specified number of frames  
5.Use `Labelimg` to annotate the image, dividing it into `player` and `center`  
6.Split the `images` and `Annotation` into `train` and `val`, and place them into `datasets`  
7.Configure and specify the paths of the dataset and category information:`datasets/data.yaml`  
8.Train model:`train_model.py`  


