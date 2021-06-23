# Webcam-Based Gaze-Tracker #

The objective is to create a webcam-based eye tracker which detects point of gaze for a person sittign at a monitor.
The project uses OpenVINO, a toolkit for inference and neural network optimisation. The project uses four different models.
- [Face Detection](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
- [Facial Landmark Detection](https://docs.openvinotoolkit.org/2018_R5/_docs_Retail_object_attributes_landmarks_regression_0009_onnx_desc_landmarks_regression_retail_0009.html)
- [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/omz_models_model_head_pose_estimation_adas_0001.html)
- [Gaze Estimation](https://docs.openvinotoolkit.org/latest/omz_models_model_gaze_estimation_adas_0002.html)

The application outputs a live video feed of where a person is looking at, on their monitor. Here is a screenshot from a video capture, with gaze overlayed on the webcam feed.
![image1](references/video-screen.png)

Here is a screenshot from a video capture, with gaze overlayed on the an image.
![image2](references/video-screen-picture.png)


### The Pipeline

The pipeline is shown as follows:

![image3](references/pipeline.png)

An in-depth description of these stages can be found over at the [Wiki](https://github.com/healyr4/Gaze-Tracking/wiki/Pipeline-Overview)

## Project Set Up and Installation

### Requirements

#### Hardware

- 6th to 10th generation Intel® Core™ processor with Iris® Pro graphics or Intel® HD Graphics.
- OR use of Intel® Neural Compute Stick 2 (NCS2)
- Webcam (optional)

#### Software

* Intel® Distribution of OpenVINO™ toolkit 2020.3 release
* Python 3.7
* OpenCV 3.4.2
* pyautogui 0.9.48

### Set up development environment

##### 1. Install OpenVINO Toolkit

Dwonlaod OpenVINO. The tutorial is located[here](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/get-started.html).

##### 2. Install dependancies

Use Anaconda to install dependencies. [Anaconda](https://docs.conda.io/en/latest/miniconda.html):

```
conda env create -f environment.yml
```

### Download the models

Use OpenVINO's `model downloader` to download the models. CD into the main directory and run the following commands, depending on your environment

```
#Linux
python3 $INTEL_OPENVINO_DIR/deployment_tools/tools/model_downloader/downloader.py --list model.lst -o models

#Windows
python "%INTEL_OPENVINO_DIR%\deployment_tools\tools\model_downloader\downloader.py" --list model.lst -o models
```

## Demo

Run the application using a video of your choice. Ensure you're in te main directory. The command should be of this format

```
# Windows
python src\main.py -o <folder_to_put_results> -i <video path> -it video --show_input --show_video
For example: python src\main.py -o <my_results> -i <bin/my_video.mp4> -it video --show_input --show_video
```

Using your Webcam:

```
# Windows
python src\main.py  -o results -it cam 
For example:python src\main.py  -o results -it cam  --show_input --show_video
```

The results will be in the directory you specify, in the above example it's my_results
