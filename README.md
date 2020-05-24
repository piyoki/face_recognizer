# Face Recognizer on Jetson

*** Copy Right 2020 Kevin Yu. All rights reserved.

*** Author: Kevin Yu

*** Update Time: 2020/05/24

*** Contact: kevinyu211@yahoo.com

This repo demonstrates how to use Dlib, a powerful Object Detection library, to detect and identify faces in real-time. Dlib is able to compile with CUDA, which means, the detector can run with GPU on the Jetson. All the demos below are tested on my Jetson AGX Xavier DevKit, it should also work on other Jetson Platforms such as the Jetson Nano and the Jetson TX2.

*** Notes: You may find more detailed description of the project [HERE](https://hikariai.net)

Table of Contents
-----------------

* [Demo](#demo)
* [Prerequisite](#prerequisite)
* [Setup](#setup)
* [How To Use](#how-to-use)

Demo
----

#### Live-Demo

<p align="center">
  <img src="https://media.giphy.com/media/SuIyc6hmQXsKondhRo/giphy.gif" | width="640" height="480">
</p>

<a name="demo"></a>

Prerequisite
------------

#### Dependencies

The Face Detection API depends on the following libraries:

- Dlib
- Face_recognition
- Pickle
- OpenCV

**Notes:**

OpenCV and Pickle are pre-installed with JetPack, so do not need to reinstalled.

For detailed steps to install Dlib and Face_recognition module, follow the Dlib installation instructions in the [Setup](#setup) section.

<a name="prerequisite"></a>

Setup
-----

```shell script
$ cd ~
$ git clone https://github.com/yqlbu/face_recognizer
$ cd face_recognizer/
$ sudo chmod +x setup.sh
$ ./setup.sh
```

<a name="setup"></a>

How To Use
----------

#### Train the custom dataset

Notes: you may customize the dataset inside /images based on your own need. To do so, you need to correctly name the image file for each image inside /images.

```shell script
$ python3 training.py
```

#### Recognize unknown faces

```shell script
$ python3 recognizer.py
```

#### Run the detector to identify faces of an input image

```shell script
$ python3 detector.py
```

#### Run the detector in real-time to identify faces of an input stream

Notes: the input sources are not limited to Camera stream, but any form of MJPEG stream such as Video, RTSP, and HTTP Stream

```shell script
$ python3 live-demo.py
```

<a name="how-to-use"></a>
