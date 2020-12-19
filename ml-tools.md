AI Explainability 360
=====================

The AI Explainability 360 toolkit is an open-source library that
supports interpretability and explainability of datasets and machine
learning models. The AI Explainability 360 Python package includes a
comprehensive set of algorithms that cover different dimensions of
explanations along with proxy explainability metrics.

It is OSS from IBM (so apache2.0) so mind the history of openness IBM
has regarding OSS product development. The documentation can be found
here: <https://aix360.readthedocs.io/en/latest/>

| 

  --------------------- -------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Python
  **Project URL**       <http://aix360.mybluemix.net/>
  **Source Location**   <https://github.com/IBM/AIX360>
  **Tag(s)**            Data analytics, ML, ML Tool, Python
  --------------------- -------------------------------------

Apollo
======

Apollo is a high performance, flexible architecture which accelerates
the development, testing, and deployment of Autonomous Vehicles.

Apollo 2.0 supports vehicles autonomously driving on simple urban roads.
Vehicles are able to cruise on roads safely, avoid collisions with
obstacles, stop at traffic lights, and change lanes if needed to reach
their destination.

Apollo 5.5 enhances the complex urban road autonomous driving
capabilities of previous Apollo releases, by introducing curb-to-curb
driving support. With this new addition, Apollo is now a leap closer to
fully autonomous urban road driving. The car has complete 360-degree
visibility, along with upgraded perception deep learning model and a
brand new prediction model to handle the changing conditions of complex
road and junction scenarios, making the car more secure and aware.

| 

  --------------------- ----------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   C++
  **Project URL**       <http://apollo.auto/>
  **Source Location**   <https://github.com/ApolloAuto/apollo>
  **Tag(s)**            ML, ML Tool
  --------------------- ----------------------------------------

Data Science Version Control (DVC)
==================================

**Data Science Version Control** or **DVC** is an **open-source** tool
for data science and machine learning projects. With a simple and
flexible Git-like architecture and interface it helps data scientists:

1.  manage **machine learning models** -- versioning, including data
    sets and transformations (scripts) that were used to generate
    models;
2.  make projects **reproducible**;
3.  make projects **shareable**;
4.  manage experiments with branching and **metrics** tracking;

It aims to replace tools like Excel and Docs that are being commonly
used as a knowledge repo and a ledger for the team, ad-hoc scripts to
track and move deploy different model versions, ad-hoc data file
suffixes and prefixes.

| 

  --------------------- ------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Python
  **Project URL**       <https://dvc.org/>
  **Source Location**   <https://github.com/iterative/dvc>
  **Tag(s)**            ML, ML Tool, Python
  --------------------- ------------------------------------

Espresso
========

Espresso is an open-source, modular, extensible end-to-end neural
automatic speech recognition (ASR) toolkit based on the deep learning
library [PyTorch](https://github.com/pytorch/pytorch) and the popular
neural machine translation toolkit `` `fairseq ``
\<<https://github.com/pytorch/fairseq>\>\`\_\_. Espresso supports
distributed training across GPUs and computing nodes, and features
various decoding approaches commonly employed in ASR, including
look-ahead word-based language model fusion, for which a fast,
parallelized decoder is implemented.

Research paper can be found at <https://arxiv.org/pdf/1909.08723.pdf>

| 

  --------------------- -----------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Python
  **Project URL**       <https://github.com/freewym/espresso>
  **Source Location**   <https://github.com/freewym/espresso>
  **Tag(s)**            ML, ML Tool, Python, speech recognition
  --------------------- -----------------------------------------

EuclidesDB
==========

EuclidesDB is a multi-model machine learning feature database that is
tight coupled with PyTorch and provides a backend for including and
querying data on the model feature space. Some features of EuclidesDB
are listed below:

-   Written in C++ for performance;
-   Uses protobuf for data serialization;
-   Uses gRPC for communication;
-   LevelDB integration for database serialization;
-   Many indexing methods implemented
    ([Annoy](https://github.com/spotify/annoy),
    [Faiss](https://github.com/facebookresearch/faiss), etc);
-   Tight PyTorch integration through libtorch;
-   Easy integration for new custom fine-tuned models;
-   Easy client language binding generation;
-   Free and open-source with permissive license;

| 

  --------------------- -----------------------------------------------
  **SBB License**       Apache License 2.0

  **Core Technology**   CPP

  **Project URL**       <https://e>
                        uclidesdb.readthedocs.io/en/latest/index.html

  **Source Location**   <https://github.com/perone/euclidesdb>

  **Tag(s)**            ML, ML Tool
  --------------------- -----------------------------------------------

Fabrik
======

Fabrik is an online collaborative platform to build, visualize and train
deep learning models via a simple drag-and-drop interface. It allows
researchers to collaboratively develop and debug models using a web GUI
that supports importing, editing and exporting networks written in
widely popular frameworks like Caffe, Keras, and TensorFlow.

| 

  --------------------- --------------------------------------
  **SBB License**       GNU General Public License (GPL) 3.0
  **Core Technology**   Javascript, Python
  **Project URL**       <http://fabrik.cloudcv.org/>
  **Source Location**   <https://github.com/Cloud-CV/Fabrik>
  **Tag(s)**            Data Visualization, ML, ML Tool
  --------------------- --------------------------------------

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

Kedro
=====

Kedro is a workflow development tool that helps you build data pipelines
that are robust, scalable, deployable, reproducible and versioned. We
provide a standard approach so that you can:

-   spend more time building your data pipeline,
-   worry less about how to write production-ready code,
-   standardise the way that your team collaborates across your project,
-   work more efficiently.

Features:

-   A standard and easy-to-use project template, allowing your
    collaborators to spend less time understanding how you've set up
    your analytics project
-   Data abstraction, managing how you load and save data so that you
    don't have to worry about the reproducibility of your code in
    different environments
-   Configuration management, helping you keep credentials out of your
    code base
-   Pipeline visualisation with
    Kedro-Viz:(<https://github.com/quantumblacklabs/kedro-viz>) making
    it easy to see how your data pipeline is constructed
-   Seamless packaging, allowing you to ship your projects to
    production, e.g. using Docker
    (<https://github.com/quantumblacklabs/kedro-docker>) or
    Kedro-Airflow (<https://github.com/quantumblacklabs/kedro-airflow>)
-   Versioning for your datasets and machine learning models whenever
    your pipeline runs

Features:

-   A standard and easy-to-use project template, allowing your
    collaborators to spend less time understanding how you've set up
    your analytics project
-   Data abstraction, managing how you load and save data so that you
    don't have to worry about the reproducibility of your code in
    different environments
-   Configuration management, helping you keep credentials out of your
    code base
-   Pipeline visualisation with
    \[Kedro-Viz\](<https://github.com/quantumblacklabs/kedro-viz>)
    making it easy to see how your data pipeline is constructed
-   Seamless packaging, allowing you to ship your projects to
    production, e.g. using
    \[Kedro-Docker\](<https://github.com/quantumblacklabs/kedro-docker>)
    or
    \[Kedro-Airflow\](<https://github.com/quantumblacklabs/kedro-airflow>)
-   Versioning for your data sets and machine learning models whenever
    your pipeline runs

Documentation on: <https://kedro.readthedocs.io/>

The REACT visualization for Kedro is on:
[https://github.com/quantumblacklabs/kedro-viz](http://%20https://github.com/quantumblacklabs/kedro-viz%20)

| 

  --------------------- ---------------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Python
  **Project URL**       <https://github.com/quantumblacklabs/kedro>
  **Source Location**   <https://github.com/quantumblacklabs/kedro>
  **Tag(s)**            ML, ML Tool, Python
  --------------------- ---------------------------------------------

Ludwig
======

Ludwig is a toolbox built on top of TensorFlow that allows to train and
test deep learning models without the need to write code. Ludwig
provides two main functionalities: training models and using them to
predict. It is based on datatype abstraction, so that the same data
preprocessing and postprocessing will be performed on different datasets
that share data types and the same encoding and decoding models
developed for one task can be reused for different tasks.

All you need to provide is a CSV file containing your data, a list of
columns to use as inputs, and a list of columns to use as outputs,
Ludwig will do the rest. Simple commands can be used to train models
both locally and in a distributed way, and to use them to predict on new
data.

A programmatic API is also available in order to use Ludwig from your
python code. A suite of visualization tools allows you to analyze
models' training and test performance and to compare them.

Ludwig is built with extensibility principles in mind and is based on
data type abstractions, making it easy to add support for new data types
as well as new model architectures.

It can be used by practitioners to quickly train and test deep learning
models as well as by researchers to obtain strong baselines to compare
against and have an experimentation setting that ensures comparability
by performing standard data preprocessing and visualization.

| 

  --------------------- ----------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Python
  **Project URL**       <https://uber.github.io/ludwig/>
  **Source Location**   <https://github.com/uber/ludwig>
  **Tag(s)**            ML, ML Tool
  --------------------- ----------------------------------

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

MLflow
======

MLflow offers a way to simplify ML development by making it easy to
track, reproduce, manage, and deploy models. MLflow (currently in alpha)
is an open source platform designed to manage the entire machine
learning lifecycle and work with any machine learning library. It
offers:

-   Record and query experiments: code, data, config, results
-   Packaging format for reproducible runs on any platform
-   General format for sending models to diverse deploy tools

| 

  --------------------- ------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Python
  **Project URL**       <https://mlflow.org/>
  **Source Location**   <https://github.com/mlflow/mlflow>
  **Tag(s)**            ML, ML Tool, Python
  --------------------- ------------------------------------

MLPerf
======

A broad ML benchmark suite for measuring performance of ML software
frameworks, ML hardware accelerators, and ML cloud platforms.

The MLPerf effort aims to build a common set of benchmarks that enables
the machine learning (ML) field to measure system performance for both
training and inference from mobile devices to cloud services. We believe
that a widely accepted benchmark suite will benefit the entire
community, including researchers, developers, builders of machine
learning frameworks, cloud service providers, hardware manufacturers,
application providers, and end users.

| 

  --------------------- ---------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Python
  **Project URL**       <https://mlperf.org/>
  **Source Location**   <https://github.com/mlperf/reference>
  **Tag(s)**            ML, ML Tool, Performance
  --------------------- ---------------------------------------

ModelDB
=======

A system to manage machine learning models.

ModelDB is an end-to-end system to manage machine learning models. It
ingests models and associated metadata as models are being trained,
stores model data in a structured format, and surfaces it through a
web-frontend for rich querying. ModelDB can be used with any ML
environment via the ModelDB Light API. ModelDB native clients can be
used for advanced support in spark.ml and scikit-learn.

The ModelDB frontend provides rich summaries and graphs showing model
data. The frontend provides functionality to slice and dice this data
along various attributes (e.g. operations like filter by hyperparameter,
group by datasets) and to build custom charts showing model performance.

| 

  --------------------- -------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Python, Javascript
  **Project URL**       <https://mitdbg.github.io/modeldb/>
  **Source Location**   <https://github.com/mitdbg/modeldb>
  **Tag(s)**            Administration, ML, ML Tool
  --------------------- -------------------------------------

Netron
======

Netron is a viewer for neural network, deep learning and machine
learning models.

Netron supports [ONNX](http://onnx.ai) (`.onnx`, `.pb`), Keras (`.h5`,
`.keras`), CoreML (`.mlmodel`) and TensorFlow Lite (`.tflite`). Netron
has experimental support for Caffe (`.caffemodel`), Caffe2
(`predict_net.pb`), MXNet (`-symbol.json`), TensorFlow.js (`model.json`,
`.pb`) and TensorFlow (`.pb`, `.meta`).

| 

  --------------------- ----------------------------------------
  **SBB License**       GNU General Public License (GPL) 2.0
  **Core Technology**   Python, Javascript
  **Project URL**       <https://www.lutzroeder.com/ai/>
  **Source Location**   <https://github.com/lutzroeder/Netron>
  **Tag(s)**            Data viewer, ML, ML Tool
  --------------------- ----------------------------------------

NLP Architect
=============

NLP Architect is an open-source Python library for exploring the
state-of-the-art deep learning topologies and techniques for natural
language processing and natural language understanding. It is intended
to be a platform for future research and collaboration.

Features:

-   Core NLP models used in many NLP tasks and useful in many NLP
    applications
-   Novel NLU models showcasing novel topologies and techniques
-   Optimized NLP/NLU models showcasing different optimization
    algorithms on neural NLP/NLU models
-   Model-oriented design:
    -   Train and run models from command-line.
    -   API for using models for inference in python.
    -   Procedures to define custom processes for training, inference or
        anything related to processing.
    -   CLI sub-system for running procedures
-   Based on optimized Deep Learning frameworks:
    -   [TensorFlow](https://www.tensorflow.org/)
    -   [PyTorch](https://pytorch.org/)
    -   [Dynet](https://dynet.readthedocs.io/en/latest/)
-   Essential utilities for working with NLP models -- Text/String
    pre-processing, IO, data-manipulation, metrics, embeddings.

| 

  --------------------- ---------------------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Python
  **Project URL**       <http://nlp_architect.nervanasys.com/>
  **Source Location**   <https://github.com/NervanaSystems/nlp-architect>
  **Tag(s)**            ML, ML Tool, NLP, Python
  --------------------- ---------------------------------------------------

ONNX
====

ONNX provides an open source format for AI models. It defines an
extensible computation graph model, as well as definitions of built-in
operators and standard data types. Initially we focus on the
capabilities needed for inferencing (evaluation).

Open Neural Network Exchange (ONNX) is an open standard format for
representing machine learning models. ONNX is supported by a community
of partners who have implemented it in many frameworks and tools.

Caffe2, PyTorch, Microsoft Cognitive Toolkit, Apache MXNet and other
tools are developing ONNX support. Enabling interoperability between
different frameworks and streamlining the path from research to
production will increase the speed of innovation in the AI community. We
are an early stage and we invite the community to submit feedback and
help us further evolve ONNX.

Companies behind ONNX are AWS, Facebook and Microsoft Corporation and
more.

| 

  --------------------- --------------------------------
  **SBB License**       MIT License
  **Core Technology**   Python
  **Project URL**       <http://onnx.ai/>
  **Source Location**   <https://github.com/onnx/onnx>
  **Tag(s)**            ML, ML Tool
  --------------------- --------------------------------

OpenML
======

OpenML is an on-line machine learning platform for sharing and
organizing data, machine learning algorithms and experiments. It claims
to be designed to create a frictionless, networked ecosystem, so that
you can readily integrate into your existing
processes/code/environments. It also allows people from all over the
world to collaborate and build directly on each other's latest ideas,
data and results, irrespective of the tools and infrastructure they
happen to use. So nice ideas to build an open science movement. The
people behind OpemML are mostly (data)scientist. So using this product
for real world business use cases will take some extra effort.

Altrhough OpenML is exposed as an foundation based on openness, a quick
inspection learned that the OpenML platform  is not as open as you want.
Also the OSS software is not created to be run on premise. So be aware
when doing large (time) investments into this OpenML platform.

| 

  --------------------- ----------------------------------------------------
  **SBB License**       BSD License 2.0 (3-clause, New or Revised) License
  **Core Technology**   Java
  **Project URL**       <https://openml.org>
  **Source Location**   <https://github.com/openml/OpenML>
  **Tag(s)**            ML, ML Tool
  --------------------- ----------------------------------------------------

Orange
======

Orange is a comprehensive, component-based software suite for machine
learning and data mining, developed at Bioinformatics Laboratory.

Orange is available by default on Anaconda Navigator dashboard.
[Orange](http://orange.biolab.si/) is a component-based data mining
software. It includes a range of data visualization, exploration,
preprocessing and modeling techniques. It can be used through a nice and
intuitive user interface or, for more advanced users, as a module for
the Python programming language.

One of the nice features is the option for visual programming. Can you
do visual interactive data exploration for rapid qualitative analysis
with clean visualizations. The graphic user interface allows you to
focus on exploratory data analysis instead of coding, while clever
defaults make fast prototyping of a data analysis workflow extremely
easy.

 

 

| 

  ------------------------------------- -----------------------------------------
  **SBB License** **Core Technology**   GNU General Public License (GPL) 3.0
  **Project URL**                       <https://orange.biolab.si/>
  **Source Location**                   <https://github.com/biolab/orange3>
  **Tag(s)**                            Data Visualization, ML, ML Tool, Python
  ------------------------------------- -----------------------------------------

PySyft
======

| A library for encrypted, privacy preserving deep learning. PySyft is a
  Python library for secure, private Deep Learning. PySyft decouples
  private data from model training, using [Multi-Party Computation
  (MPC)](https://en.wikipedia.org/wiki/Secure_multi-party_computation)
  within PyTorch. View the paper on
  [Arxiv](https://arxiv.org/abs/1811.04017).

| 

  --------------------- ---------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Python
  **Project URL**       <https://github.com/OpenMined/PySyft>
  **Source Location**   <https://github.com/OpenMined/PySyft>
  **Tag(s)**            ML, ML Tool, Python, Security
  --------------------- ---------------------------------------

RAPIDS
======

The RAPIDS suite of software libraries gives you the freedom to execute
end-to-end data science and analytics pipelines entirely on GPUs. It
relies on [NVIDIA® CUDA®](https://developer.nvidia.com/cuda-toolkit)
primitives for low-level compute optimization, but exposes that GPU
parallelism and high-bandwidth memory speed through user-friendly Python
interfaces.

RAPIDS also focuses on common data preparation tasks for analytics and
data science. This includes a familiar DataFrame API that integrates
with a variety of machine learning algorithms for end-to-end pipeline
accelerations without paying typical serialization costs--. RAPIDS also
includes support for multi-node, multi-GPU deployments, enabling vastly
accelerated processing and training on much larger dataset sizes.

| 

  --------------------- --------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   C++
  **Project URL**       <http://rapids.ai/>
  **Source Location**   <https://github.com/rapidsai/>
  **Tag(s)**            ML, ML Hosting, ML Tool
  --------------------- --------------------------------

SHAP
====

**SHAP (SHapley Additive exPlanations)** is a unified approach to
explain the output of any machine learning model. SHAP connects game
theory with local explanations, uniting several previous methods \[1-7\]
and representing the only possible consistent and locally accurate
additive feature attribution method based on expectations (see our
[papers](https://github.com/slundberg/shap#citations) for details and
citations).

There are also sample notebooks that demonstrate different use cases for
SHAP in the github repro.

| 

  --------------------- -------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Python
  **Project URL**       <https://github.com/slundberg/shap>
  **Source Location**   <https://github.com/slundberg/shap>
  **Tag(s)**            ML, ML Tool
  --------------------- -------------------------------------

Skater
======

Skater is a python package for model agnostic interpretation of
predictive models. With Skater, you can unpack the internal mechanics of
arbitrary models; as long as you can obtain inputs, and use a function
to obtain outputs, you can use Skater to learn about the models internal
decision policies.

The project was started as a research idea to find ways to enable better
interpretability(preferably human interpretability) to predictive "black
boxes" both for researchers and practioners.

Documentation at:
<https://datascienceinc.github.io/Skater/overview.html>

| 

  --------------------- ------------------------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Python
  **Project URL**       <https://www.datascience.com/resources/tools/skater>
  **Source Location**   <https://github.com/datascienceinc/Skater>
  **Tag(s)**            ML, ML Tool
  --------------------- ------------------------------------------------------

Snorkel
=======

Snorkel is a system for rapidly **creating, modeling, and managing
training data**, currently focused on accelerating the development of
*structured or "dark" data extraction applications* for domains in which
large labeled training sets are not available or easy to obtain.

| 

  --------------------- -------------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Python
  **Project URL**       <https://www.snorkel.org/>
  **Source Location**   <https://github.com/HazyResearch/snorkel>
  **Tag(s)**            ML, ML Tool
  --------------------- -------------------------------------------

Streamlit
=========

The fastest way to build custom ML tools. Streamlit lets you create apps
for your machine learning projects with deceptively simple Python
scripts. It supports hot-reloading, so your app updates live as you edit
and save your file. No need to mess with HTTP requests, HTML,
JavaScript, etc. All you need is your favorite editor and a browser.

Documentation on: <https://streamlit.io/docs/>

| 

  --------------------- -----------------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Javascipt, Python
  **Project URL**       <https://streamlit.io/>
  **Source Location**   <https://github.com/streamlit/streamlit>
  **Tag(s)**            ML, ML Framework, ML Hosting, ML Tool, Python
  --------------------- -----------------------------------------------

TensorWatch
===========

TensorWatch is a debugging and visualization tool designed for data
science, deep learning and reinforcement learning from Microsoft
Research. It works in Jupyter Notebook to show real-time visualizations
of your machine learning training and perform several other key analysis
tasks for your models and data.

TensorWatch is designed to be flexible and extensible so you can also
build your own custom visualizations, UIs, and dashboards. Besides
traditional "what-you-see-is-what-you-log" approach, it also has a
unique capability to execute arbitrary queries against your live ML
training process, return a stream as a result of the query and view this
stream using your choice of a visualizer (we call this [Lazy Logging
Mode](https://github.com/microsoft/tensorwatch#lazy-logging-mode%5D)).

TensorWatch is under heavy development with a goal of providing a
platform for debugging machine learning in one easy to use, extensible,
and hackable package.

| 

  --------------------- --------------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Python
  **Project URL**       <https://github.com/microsoft/tensorwatch>
  **Source Location**   <https://github.com/microsoft/tensorwatch>
  **Tag(s)**            ML, ML Tool
  --------------------- --------------------------------------------

VisualDL
========

VisualDL is an open-source cross-framework web dashboard that richly
visualizes the performance and data flowing through your neural network
training. VisualDL is a deep learning visualization tool that can help
design deep learning jobs. It includes features such as scalar,
parameter distribution, model structure and image visualization.

| 

  --------------------- --------------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   C++
  **Project URL**       <http://visualdl.paddlepaddle.org/>
  **Source Location**   <https://github.com/PaddlePaddle/VisualDL>
  **Tag(s)**            ML, ML Tool
  --------------------- --------------------------------------------

What-If Tool
============

The [What-If Tool](https://pair-code.github.io/what-if-tool) (WIT)
provides an easy-to-use interface for expanding understanding of a
black-box ML model. With the plugin, you can perform inference on a
large set of examples and immediately visualize the results in a variety
of ways. Additionally, examples can be edited manually or
programatically and re-run through the model in order to see the results
of the changes. It contains tooling for investigating model performance
and fairness over subsets of a dataset.

The purpose of the tool is that give people a simple, intuitive, and
powerful way to play with a trained ML model on a set of data through a
visual interface with absolutely no code required.

| 

  --------------------- ------------------------------------------------
  **SBB License**       Apache License 2.0

  **Core Technology**   Python

  **Project URL**       <https://pair-code.github.io/what-if-tool/>

  **Source Location**   https
                        ://github.com/tensorflow/tensorboard/tree/mas
                        ter/tensorboard/plugins/interactive\_inference

  **Tag(s)**            ML, ML Tool
  --------------------- ------------------------------------------------
