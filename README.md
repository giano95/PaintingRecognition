# Painting Recognition

[report_1](./foto_presentazione/report_1.png)

## Prerequisities
- Python 3.6.5
- OpenCV
- Numpy

## Installation
Create a new virtual enviroment called 'env' and with a 3.6.5 version of python (we used pyenv to handle multiple python version):
```bash
virtualenv -p /home/marco/.pyenv/versions/3.6.5/bin/python env
```
Activate the virtual enviroment:
```bash
source env/bin/activate
```
Install the opencv-contrib wheel provided by Lorenzo:
```bash
pip3 install resources/opencv_contrib_python_headless-4.1.0.25-cp36-cp36m-linux_x86_64.whl
```
Install the requirements:
```bash
pip3 install -r requirements.txt
```
Clone the SRN-Deblur project from: https://github.com/jiangsutx/SRN-Deblur 


## Run
```bash
python3 main.py --input GOPR5826.MP4 --output GOPR5826_with_draw
```

The output video is created inside a directory called 'output' istead the input video must be located inside a directory called 'input.

### Resources
The 'resources' folder contains:
- haarcascade_frontalface_default.xml for face detection
- haarcascade_eye.xml for eyes detection
- map.png picture of the museum map
- data.csv info about the paintings
- coordinates.json coordinates used for people localization
- paintings_db the folder containings all the paintings images
- calibration_data.npz a file containing the data for the undistortion of the GOPRO video
- opencv_contrib_python_headless-4.1.0.25-cp36-cp36m-linux_x86_64.whl opencv-contrib wheel

### External Resources
The 'yolo-coco' folder contains:
- yolov3.weights.txt link for the yolov3.weights download.
The 'SRN-Deblur' folder contain:
- deblur-project.txt link of the deblur project we use in image-prerocessing

### Videos used in the Testing phase
In both the Painting Detection and Retrivial we do not use the videos taht give us the best result but we try to use simple and difficult ones in order to have a realistic testing scenario. For the Painting Detection Testing we use:
- GOPR5826
- 20180206_113800
- VIRB0400
- 20180206_11360
- VIRB0395
- IMG_9626
For the Painting Retrival instead:
- GOPR5826
- 20180206_114604
- 20180206_113600
- VIRB0395
- IMG_2653
- VIRB0392
