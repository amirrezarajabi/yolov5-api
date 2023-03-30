# YOLOV5-api

The main structure of this code is taken from [Yolo's repository](https://github.com/ultralytics/yolov5) but it is prepared for deploying on server by adding this two files :

### [api.py](./api.py)
* using:  `python3 api.py`
* This code receives a zip file of samples by http post method and responds to it by zip file of the results
* The structure of directories in the zip file of samples
```
-   content
    -   output
        -    samples No.1 name
            -    0.jpg
            -    1.jpg
            -    ...
        -    samples No.2 name
            -    0.jpg
            -    1.jpg
            -    ...
        -    ...
```
* The structure of directories in the zip file of the results
```
-   output
    -    samples No.1 name
        -    0.jpg
        -    1.jpg
        -    ...
    -    samples No.2 name
        -    0.jpg
        -    1.jpg
        -    ...
    -    ...
```
* Models are loadded in this file
* This works on `port=5000`

### [run.py](./run.py)
* This consists of two functions:
  * `prepare_model` : This function loads model
    * argument :
      * `weights` : path to model's weight
    * return :
      * model & device that model is loadded on it
  * `run` : This function used to run topk for each sample by getting:
    * argument:
      * `model` : name of the model for dogs or horses
      * `device` : the device that model is load on it
      * `save_directory` : directory for saving the result ofrun
      * `source` : path to one of the samples

## Excutation

First step is installing pip :
*     sudo apt install python3-pip
after that, it's Dependencies` turn:
*     pip install -r requirements.txt
Then it will be ready by running `api.py`
*     python3 api.py