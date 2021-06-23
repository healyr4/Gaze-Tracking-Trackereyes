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
