# Computer Pointer Controller

## Introduction
### What it Does

This project demonstrate to use the InferenceEngine API from Intel's OpenVino ToolKit to build application that control the mouse of computer using eye movement.
In this project we use Gaze Estimation that you can find [here](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html) to estimate the user's eyes and change the mouse position.

### How it Works

The Computer Pointer Controller uses the Inference Engine included in the Intel® Distribution of OpenVINO™ toolkit and the Python auto GUI [pyautogui](https://pyautogui.readthedocs.io/en/latest/) library to move the computer pointer. 

The gaze estimation model requires three inputs(the head pose, the eye left and eye right) as the outputs of theses models :  

- [Face Detection](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
- [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
- [Facial Landmarks Detection](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)

## Project Set Up and Installation

### Setup

#### Install Intel® Distribution of OpenVINO™ toolkit

Refer to [Intel® Distribution of OpenVINO™](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux) for more information about how to install and setup the Intel® Distribution of OpenVINO™ toolkit.

You will need the OpenCL™ Runtime Package if you plan to run inference on the GPU. It is not mandatory for CPU inference. 

#### Create a virtual environment
Actions will be similar to the one below:`
- Install virtual environment by following instructions in this [Documentation](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
- Create a virtual environment ```$ python3 -m venv /path/to/new/virtual/env``
- Activate the new virtual environment ```$ source /path/to/new/virtual/env/bin/activate```

### Install Requirements
- Install project dependencies
Here is the structure of my project 

![Project directory structure](./resources/architecture.png)

If you’re going to share the project with the rest of the world you will need to install dependencies by running at directory root  
```$pip install -r requirements.txt```

### Download Models
Refer to the official documentation of [Intel® Distribution of OpenVINO™](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux) for installation of tools.
-  Download theses models and place them in `./resources/models` folder

##### What model to use

By default, this application uses four Intel® model: 
- the **face-detection-adas-binary-0001**, 
- **landmarks-regression-retail-0009**, 
- **head-pose-estimation-adas-0001**, 
- and finally **gaze-estimation-adas-0002**, that can be accessed using the **model downloader**. The **model downloader** downloads the __.xml__ and __.bin__ files that will be used by the application.

##### Download the __.xml__ and __.bin__ files

Go to the **model downloader** directory present inside Intel® Distribution of OpenVINO™ toolkit:

  ```
  cd /opt/intel/openvino/deployment_tools/tools/model_downloader
  ```

Specify which model to download with `--name`.
- To download the different  model, run the following command:


  ```
  sudo ./downloader.py --name gaze-estimation-adas-0002
  sudo ./downloader.py --name landmarks-regression-retail-0009
  sudo ./downloader.py --name face-detection-adas-binary-0001
  sudo ./downloader.py --name head-pose-estimation-adas-0001

  ```


**Note:** The application uses a configuration file  `models.json` in resources folder. Update the paths of different models in this file if they are not the default.<br><br>


## Run the application

#### Running on the CPU

This project was tested on CPU using `python 3.7.
When running Intel® Distribution of OpenVINO™ toolkit Python applications on the CPU, the CPU extension library is required. This can be found at /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/

Though by default application runs on CPU, this can also be explicitly specified by ```-d CPU``` command-line argument:

```
python3 main.py -i resources/demo.mp4  -d  CPU 
```

## Demo

To run basic demo, use `demo.mp4` video   in `./resources/video.mp4`  and `-d CPU` option.

```
python3 main.py  --model ./resources/models.json  --video ./resources/demo.mp4  --stats true --toggle false

```

## Documentation

- Model downloader tool:  [OpenVino Tools Documentation](https://docs.openvinotoolkit.org/latest/_tools_downloader_README.html) 

- Python Auto Gui : [pyautogui](https://pyautogui.readthedocs.io/en/latest/)

- Commands Lines arguments :

    `-m, --model`, is the path to model config file default `./resource/models.json` it is required
    
    `-d, --device`, default='CPU'  Specify the target device to infer on `example : CPU, GPU etc...`
    
    `-v, --video`, default=None  Specify the video path, not required for camera 
    
    `-i, --input_type`, default='video', Specify the input pipeline video or camera input
    
    `-t, --toggle`, default='false'  Toggle the camera or video feed  when enable and show just statistics  `example : true`
    
    `-s, --stats`, default='false',  Enable statistics output of application

## Benchmarks

Here are the results of running multiple model precisions on CPU devise.

The benchmarks  include: model loading time, input/output processing time, model inference time etc.

### Model landmarks-regression-retail-0009

 `INFO ] Device info  
         CPU  
         MKLDNNPlugin............ version 2.1  
         Build................... 42025`  
         
#####- FP16

    Count:      89500 iterations
    Duration:   60003.71 ms
    Latency:    1.78 ms
    Throughput: 1491.57 FPS
    

#####- FP16-INT8

    Count:      115196 iterations
    Duration:   60015.98 ms
    Latency:    1.45 ms
    Throughput: 1919.42 FPS

    
####- FP32

    Count:      83532 iterations
    Duration:   60010.79 ms
    Latency:    1.79 ms
    Throughput: 1391.95 FPS


### Model head-pose-estimation-adas-0001

####- FP16

    Count:      16948 iterations
    Duration:   60009.26 ms
    Latency:    10.90 ms
    Throughput: 282.42 FPS
    
    
####- FP32

    Count:      14788 iterations
    Duration:   60028.53 ms
    Latency:    11.62 ms
    Throughput: 246.35 FPS

### Model gaze-estimation-adas-0002

####- FP16
 
    Count:      11784 iterations
    Duration:   60044.69 ms
    Latency:    15.46 ms
    Throughput: 196.25 FPS
    

####- FP32

    Count:      10964 iterations
    Duration:   60038.59 ms
    Latency:    17.36 ms
    Throughput: 182.62 FPS

### Model face-detection-adas-0001

####- FP16

    Count:      572 iterations
    Duration:   60754.75 ms
    Latency:    335.76 ms
    Throughput: 9.41 FPS

####- FP16-INT8

    Count:      940 iterations
    Duration:   60278.97 ms
    Latency:    210.88 ms
    Throughput: 15.59 FPS
    
####- FP32

    Count:      476 iterations
    Duration:   60568.63 ms
    Latency:    441.17 ms
    Throughput: 7.86 FPS
    
    
## Results
    

 
 - There is difference in running time of FP32 , FP16 and INT8 depending of model. 
 
 For example for  face-detection-adas-0001 model the throughput  of FP16-INT8 > FP16 > FP32.
 
 I observe that FP16-INT8 offer better performance for FPS compared to the results of inference in higher precision (FP32), because they allow loading more data into a single processor instruction. 
 
 The consequence is that the accuracy is reduced for lower precisions.
 
 
## Stand Out Suggestions

- Benchmark the running times of different parts of pre-processing and inference pipeline  by specifying `stats` argument in CLI.

- Edge cases : Multiple faces in the same input frame are taken in account.

- Add a CLI command to toggle video or camera output, statistics are presented using `-t, --toggle` command option.
 
- Allow user to select their input option in the command line argument : `-i, --input` with values of `video` or `cam`
 


### Edge Cases

- Multiple people in the frame:

Multiple people in the frame will return many  cropped eyes to feed to gaze model. 
To solve this problem  I loop on different faces and move the pointer accordingly.

- Lighting change :

 Result on bad performance, faces are not detected at all.


