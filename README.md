# Face Recognizer on Jetson

*** Copy Right 2020 Kevin Yu. All rights reserved.

*** Author: Kevin Yu

*** Update Time: 2020/05/19

This repo aims to give you clear instructions on how to install packages in AArch64(ARM) Platform, especially in Jetson family. All the packages have been tested on Jetson AGX Xavier and Jetson Nano.

*** Notes: the instructions below are for manual installation. For auto installation, you may find the installation script [HERE](https://github.com/yqlbu/jetson-install)

Table of Contents
-----------------

* [Demo](#demo)
* [Prerequisite](#prerequisite)
* [Setup](#setup)
* [How To Use](#how-to-use)

Demo
----

<a name="demo"></a>

Prerequisite
------------

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

```shell script
$ python3 training.py
```

```shell script
$ python3 recognizer.py
```

```shell script
$ python3 detector.py
```

```shell script
$ python3 live-demo.py
```

<a name="how-to-use"></a>


