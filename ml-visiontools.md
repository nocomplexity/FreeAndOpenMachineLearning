libfacedetection
================

This is an open source library for CNN-based face detection in images.
The CNN model has been converted to static variables in C source files.
The source code does not depend on any other libraries. What you need is
just a C++ compiler. You can compile the source code under Windows,
Linux, ARM and any platform with a C++ compiler.

SIMD instructions are used to speed up the detection. You can enable
AVX2 if you use Intel CPU or NEON for ARM.

| 

  --------------------- -----------------------------------------------
  **SBB License**       GNU General Public License (GPL) 2.0
  **Core Technology**   CPP
  **Project URL**       <https://github.com/ShiqiYu/libfacedetection>
  **Source Location**   <https://github.com/ShiqiYu/libfacedetection>
  **Tag(s)**            Computer vision
  --------------------- -----------------------------------------------

YOLOv3
======

A minimal PyTorch implementation of YOLOv3, with support for training,
inference and evaluation.

You only look once (YOLO) is a state-of-the-art, real-time object
detection system. In depth paper on YOLOv3 is on:
<https://pjreddie.com/media/files/papers/YOLOv3.pdf>

| 

  --------------------- -----------------------------------------------------
  **SBB License**       GNU General Public License (GPL) 2.0
  **Core Technology**   Python
  **Project URL**       <https://pjreddie.com/darknet/yolo/>
  **Source Location**   <https://github.com/eriklindernoren/PyTorch-YOLOv3>
  **Tag(s)**            Computer vision, ML
  --------------------- -----------------------------------------------------

Raster Vision
=============

Raster Vision is an open source Python framework for building computer
vision models on satellite, aerial, and other large imagery sets
(including oblique drone imagery).

It allows users (who don't need to be experts in deep learning!) to
quickly and repeatably configure experiments that execute a machine
learning workflow including: analyzing training data, creating training
chips, training models, creating predictions, evaluating models, and
bundling the model files and configuration for easy deployment.

Some features:

-   There is built-in support for chip classification, object detection,
    and semantic segmentation with backends using PyTorch and
    Tensorflow.
-   Experiments can be executed on CPUs and GPUs with built-in support
    for running in the cloud using AWS Batch. The framework is
    extensible to new data sources, tasks (eg. object detection),
    backends (eg. TF Object Detection API), and cloud providers.

Documentation on: <https://docs.rastervision.io/>

| 

  --------------------- -------------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Python
  **Project URL**       <https://rastervision.io/>
  **Source Location**   <https://github.com/azavea/raster-vision>
  **Tag(s)**            Computer vision, ML
  --------------------- -------------------------------------------

DeOldify
========

A Deep Learning based project for colorizing and restoring old images
(and video!)

| 

  --------------------- --------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Python
  **Project URL**       <https://github.com/jantic/DeOldify>
  **Source Location**   <https://github.com/jantic/DeOldify>
  **Tag(s)**            Computer vision, ML
  --------------------- --------------------------------------

SOD
===

SOD is an embedded, modern cross-platform computer vision and machine
learning software library that expose a set of APIs for deep-learning,
advanced media analysis & processing including real-time, multi-class
object detection and model training on embedded systems with limited
computational resource and IoT devices.

SOD was built to provide a common infrastructure for computer vision
applications and to accelerate the use of machine perception in open
source as well commercial products.

Designed for computational efficiency and with a strong focus on
real-time applications. SOD includes a comprehensive set of both classic
and state-of-the-art deep-neural networks with their [pre-trained
models](https://pixlab.io/downloads).

| 

  --------------------- --------------------------------------
  **SBB License**       GNU General Public License (GPL) 3.0
  **Core Technology**   C
  **Project URL**       <https://sod.pixlab.io/>
  **Source Location**   <https://github.com/symisc/sod>
  **Tag(s)**            Computer vision, ML
  --------------------- --------------------------------------

makesense.ai
============

makesense.ai is a free to use online tool for labelling photos. Thanks
to the use of a browser it does not require any complicated installation
-- just visit the website and you are ready to go. It also doesn't
matter which operating system you're running on -- we do our best to be
truly cross-platform. It is perfect for small computer vision
deeplearning projects, making the process of preparing a dataset much
easier and faster.

| 

  --------------------- ------------------------------------------
  **SBB License**       GNU General Public License (GPL) 3.0
  **Core Technology**   Typescript
  **Project URL**       <https://www.makesense.ai/>
  **Source Location**   <https://github.com/SkalskiP/make-sense>
  **Tag(s)**            Computer vision, ML, ML Tool, Photos
  --------------------- ------------------------------------------

DeepPrivacy
===========

DeepPrivacy is a fully automatic anonymization technique for images.

The DeepPrivacy GAN never sees any privacy sensitive information,
ensuring a fully anonymized image. It utilizes bounding box annotation
to identify the privacy-sensitive area, and sparse pose information to
guide the network in difficult scenarios.

DeepPrivacy detects faces with state-of-the-art detection methods. [Mask
R-CNN](https://arxiv.org/abs/1703.06870) is used to generate a sparse
pose information of the face, and
[DSFD](https://arxiv.org/abs/1810.10220) is used to detect faces in the
image.

The Github repository contains the source code for the paper
["DeepPrivacy: A Generative Adversarial Network for Face
Anonymization"](https://arxiv.org/abs/1909.04538), published at ISVC
2019.

| 

  --------------------- -------------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Python
  **Project URL**       <https://github.com/hukkelas/DeepPrivacy>
  **Source Location**   <https://github.com/hukkelas/DeepPrivacy>
  **Tag(s)**            Computer vision, ML, Privacy, Python
  --------------------- -------------------------------------------

Face\_recognition
=================

The world's simplest facial recognition api for Python and the command
line.

Recognize and manipulate faces from Python or from the command line with
the world's simplest face recognition library.

Built using [dlib](http://dlib.net/)'s state-of-the-art face recognition
built with deep learning. The model has an accuracy of 99.38% on the
[Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) benchmark.

This also provides a simple `face_recognition` command line tool that
lets you do face recognition on a folder of images from the command
line!

Full API documentation can be found here:
<https://face-recognition.readthedocs.io/en/latest/>

Git quick-scan report:

-   Date of git statics quick-scan report: 2019/12/19
-   Number of files in the git repository: 96
-   Total Lines of Code (of all files): 70415 total
-   Most recent commit in this repository: Tue Dec 3 16:53:45 2019 +0530
-   Number of authors:33

First commit info:

-   Author: Adam Geitgey
-   Date: Fri Mar 3 16:29:23 2017 -0800

| 

  --------------------- ------------------------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Python
  **Project URL**       <https://github.com/ageitgey/face_recognition>
  **Source Location**   <https://github.com/ageitgey/face_recognition>
  **Tag(s)**            Computer vision, face detection, ML, ML Tool, Python
  --------------------- ------------------------------------------------------

DeepFaceLab
===========

DeepFaceLab is a tool that utilizes machine learning to replace faces in
videos.

More than 95% of deepfake videos are created with DeepFaceLab.

| 

  --------------------- -----------------------------------------
  **SBB License**       GNU General Public License (GPL) 3.0
  **Core Technology**   Python
  **Project URL**       <https://github.com/iperov/DeepFaceLab>
  **Source Location**   <https://github.com/iperov/DeepFaceLab>
  **Tag(s)**            Computer vision, Deepfakes, ML, Python
  --------------------- -----------------------------------------

FaceSwap
========

FaceSwap is a tool that utilizes deep learning to recognize and swap
faces in pictures and videos.

When faceswapping was first developed and published, the technology was
groundbreaking, it was a huge step in AI development. It was also
completely ignored outside of academia because the code was confusing
and fragmentary. It required a thorough understanding of complicated AI
techniques and took a lot of effort to figure it out. Until one
individual brought it together into a single, cohesive collection.
Before "deepfakes" these techniques were like black magic, only
practiced by those who could understand all of the inner workings as
described in esoteric and endlessly complicated books and papers.

Powered by Tensorflow, Keras and Python; Faceswap will run on Windows,
macOS and Linux. And GPL licensed!

| 

  --------------------- -----------------------------------------
  **SBB License**       GNU General Public License (GPL) 3.0
  **Core Technology**   Python
  **Project URL**       <https://www.faceswap.dev/>
  **Source Location**   <https://github.com/deepfakes/faceswap>
  **Tag(s)**            Computer vision, Deepfakes, ML, Python
  --------------------- -----------------------------------------

JeelizFaceFilter
================

Javascript/WebGL lightweight face tracking library designed for
augmented reality webcam filters. Features : multiple faces detection,
rotation, mouth opening. Various integration examples are provided
(Three.js, Babylon.js, FaceSwap, Canvas2D, CSS3D...).

Enables developers to solve computer-vision problems directly from the
browser.

Features:

-   face detection,
-   face tracking,
-   face rotation detection,
-   mouth opening detection,
-   multiple faces detection and tracking,
-   very robust for all lighting conditions,
-   video acquisition with HD video ability,
-   interfaced with 3D engines like THREE.JS, BABYLON.JS, A-FRAME,
-   interfaced with more accessible APIs like CANVAS, CSS3D.

| 

  --------------------- -------------------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Javascript
  **Project URL**       <https://jeeliz.com/>
  **Source Location**   <https://github.com/jeeliz/jeelizFaceFilter>
  **Tag(s)**            Computer vision, face detection, Javascript, ML
  --------------------- -------------------------------------------------

OpenCV: Open Source Computer Vision Library
===========================================

OpenCV (Open Source Computer Vision Library) is an open source computer
vision and machine learning software library. OpenCV was built to
provide a common infrastructure for computer vision applications and to
accelerate the use of machine perception in the commercial products.
Being a BSD-licensed product, OpenCV makes it easy for businesses to
utilize and modify the code.

The library has more than 2500 optimized algorithms, which includes a
comprehensive set of both classic and state-of-the-art computer vision
and machine learning algorithms. These algorithms can be used to detect
and recognize faces, identify objects, classify human actions in videos,
track camera movements, track moving objects, extract 3D models of
objects, produce 3D point clouds from stereo cameras, stitch images
together to produce a high resolution image of an entire scene, find
similar images from an image database, remove red eyes from images taken
using flash, follow eye movements, recognize scenery and establish
markers to overlay it with augmented reality, etc.

| 

  --------------------- ----------------------------------------------------
  **SBB License**       BSD License 2.0 (3-clause, New or Revised) License
  **Core Technology**   C
  **Project URL**       <https://opencv.org/>
  **Source Location**   <https://github.com/opencv/opencv>
  **Tag(s)**            Computer vision, ML
  --------------------- ----------------------------------------------------

Luminoth
========

Luminoth is an open source toolkit for computer vision. Currently, we
support object detection and image classification, but we are aiming for
much more. It is built in Python, using TensorFlow and Sonnet.

Note: No longer maintained.

| 

  --------------------- ----------------------------------------------------
  **SBB License**       BSD License 2.0 (3-clause, New or Revised) License
  **Core Technology**   Python
  **Project URL**       <https://luminoth.ai>
  **Source Location**   <https://github.com/tryolabs/luminoth>
  **Tag(s)**            Computer vision, ML
  --------------------- ----------------------------------------------------
