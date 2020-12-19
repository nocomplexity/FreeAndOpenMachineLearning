Acme
====

Acme is a library of reinforcement learning (RL) agents and agent
building blocks. Acme strives to expose simple, efficient, and readable
agents, that serve both as reference implementations of popular
algorithms and as strong baselines, while still providing enough
flexibility to do novel research. The design of Acme also attempts to
provide multiple points of entry to the RL problem at differing levels
of complexity.

Overall Acme strives to expose simple, efficient, and readable agent
baselines while still providing enough flexibility to create novel
implementations.

| 

  --------------------- ------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Python
  **Project URL**       <https://github.com/deepmind/acme>
  **Source Location**   <https://github.com/deepmind/acme>
  **Tag(s)**            ML Framework
  --------------------- ------------------------------------

AdaNet
======

AdaNet is a lightweight TensorFlow-based framework for automatically
learning high-quality models with minimal expert intervention. AdaNet
builds on recent AutoML efforts to be fast and flexible while providing
learning guarantees. Importantly, AdaNet provides a general framework
for not only learning a neural network architecture, but also for
learning to ensemble to obtain even better models.

This project is based on the *AdaNet algorithm*, presented in "[AdaNet:
Adaptive Structural Learning of Artificial Neural
Networks](http://proceedings.mlr.press/v70/cortes17a.html)" at [ICML
2017](https://icml.cc/Conferences/2017), for learning the structure of a
neural network as an ensemble of subnetworks.

AdaNet has the following goals:

-   *Ease of use*: Provide familiar APIs (e.g. Keras, Estimator) for
    training, evaluating, and serving models.
-   *Speed*: Scale with available compute and quickly produce high
    quality models.
-   *Flexibility*: Allow researchers and practitioners to extend AdaNet
    to novel subnetwork architectures, search spaces, and tasks.
-   *Learning guarantees*: Optimize an objective that offers theoretical
    learning guarantees.

Documentation at <https://adanet.readthedocs.io/en/latest/>

| 

  --------------------- --------------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Python
  **Project URL**       <https://adanet.readthedocs.io/en/latest/>
  **Source Location**   <https://github.com/tensorflow/adanet>
  **Tag(s)**            ML, ML Framework
  --------------------- --------------------------------------------

Analytics Zoo
=============

Analytics Zoo provides a unified analytics + AI platform that seamlessly
unites *Spark, TensorFlow, Keras and BigDL* programs into an integrated
pipeline; the entire pipeline can then transparently scale out to a
large Hadoop/Spark cluster for distributed training or inference.

-   *Data wrangling and analysis using PySpark*
-   *Deep learning model development using TensorFlow or Keras*
-   *Distributed training/inference on Spark and BigDL*
-   *All within a single unified pipeline and in a user-transparent
    fashion!*

| 

  --------------------- ----------------------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Python
  **Project URL**       <https://analytics-zoo.github.io/master/>
  **Source Location**   <https://github.com/intel-analytics/analytics-zoo>
  **Tag(s)**            ML, ML Framework, Python
  --------------------- ----------------------------------------------------

Apache MXNet
============

Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with
Dynamic, Mutation-aware Dataflow Dep Scheduler; for Python, R, Julia,
Scala, Go, Javascript and more.

All major GPU and CPU vendors support this project, but also the real
giants like Amazon, Microsoft, Wolfram and a number of very respected
universities. So watch this project or play with it to see if it fits
your use case.

Apache MXNet (incubating) is a deep learning framework designed for both
*efficiency* and *flexibility*. It allows you to **mix** [symbolic and
imperative
programming](https://mxnet.incubator.apache.org/architecture/index.html#deep-learning-system-design-concepts)
to **maximize** efficiency and productivity. At its core, MXNet contains
a dynamic dependency scheduler that automatically parallelizes both
symbolic and imperative operations on the fly. A graph optimization
layer on top of that makes symbolic execution fast and memory efficient.
MXNet is portable and lightweight, scaling effectively to multiple GPUs
and multiple machines.

MXNet is also more than a deep learning project. It is also a collection
of [blue prints and
guidelines](https://mxnet.incubator.apache.org/architecture/index.html#deep-learning-system-design-concepts)
for building deep learning systems, and interesting insights of DL
systems for hackers.

Gluon is the high-level interface for MXNet. It is more intuitive and
easier to use than the lower level interface. Gluon supports dynamic
(define-by-run) graphs with JIT-compilation to achieve both flexibility
and efficiency. The perfect starters documentation with a great crash
course on deep learning can be found here: <https://d2l.ai/index.html> 
An earlier version of this documentation is still available on:[ 
http://gluon.mxnet.io/](http://gluon.mxnet.io/)

Part of the project is also the the Gluon API specification (see
<https://github.com/gluon-api/gluon-api>)

The Gluon API specification (Python based) is an effort to improve
speed, flexibility, and accessibility of deep learning technology for
all developers, regardless of their deep learning framework of choice.
The Gluon API offers a flexible interface that simplifies the process of
prototyping, building, and training deep learning models without
sacrificing training speed.

| 

  --------------------- ---------------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   CPP
  **Project URL**       <https://mxnet.apache.org/>
  **Source Location**   <https://github.com/apache/incubator-mxnet>
  **Tag(s)**            ML, ML Framework
  --------------------- ---------------------------------------------

Apache Spark MLlib
==================

Apache Spark MLlib. MLlib is Apache Spark's scalable machine learning
library. MLlib is a Spark subproject providing machine learning
primitives. MLlib is a standard component of Spark providing machine
learning primitives on top of Spark platform.

Apache Spark is a FOSS platform for large-scale data processing. The
Spark engine is written in Scala and is well suited for applications
that reuse a working set of data across multiple parallel operations.
It's designed to work as a standalone cluster or as part of Hadoop YARN
cluster. It can access data from sources such as HDFS, Cassandra or
Amazon S3.

MLlib can be seen as a core Spark's APIs and interoperates with NumPy in
Python and R libraries. And Spark is very fast! MLlib ships with Spark
as a standard component.

MLlib library contains many algorithms and utilities, e.g.:

-   Classification: logistic regression, naive Bayes.
-   Regression: generalized linear regression, survival regression.
-   Decision trees, random forests, and gradient-boosted trees.
-   Recommendation: alternating least squares (ALS).
-   Clustering: K-means, Gaussian mixtures (GMMs).
-   Topic modeling: latent Dirichlet allocation (LDA).
-   Frequent item sets, association rules, and sequential pattern
    mining.

Using Spark MLlib gives the following advantages:

-   Excellent scalability options
-   Performance
-   User-friendly APIs
-   Integration with Spark and its other components

But using Spark means that also the Spark platform must be used.

| 

  --------------------- -----------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Java
  **Project URL**       <https://spark.apache.org/mllib/>
  **Source Location**   <https://github.com/apache/spark>
  **Tag(s)**            ML, ML Framework
  --------------------- -----------------------------------

auto\_ml
========

Automated machine learning for analytics & production.

Automates the whole machine learning process, making it super easy to
use for both analytics, and getting real-time predictions in production.

Unfortunate unmaintained currently, but still worth playing with.

| 

  --------------------- ------------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Python
  **Project URL**       <http://auto-ml.readthedocs.io>
  **Source Location**   <https://github.com/ClimbsRocks/auto_ml>
  **Tag(s)**            ML, ML Framework
  --------------------- ------------------------------------------

BigDL
=====

BigDL is a distributed deep learning library for Apache Spark; with
BigDL, users can write their deep learning applications as standard
Spark programs, which can directly run on top of existing Spark or
Hadoop clusters.

-   **Rich deep learning support.** Modeled after
    [Torch](http://torch.ch/), BigDL provides comprehensive support for
    deep learning, including numeric computing (via
    [Tensor](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/tensor))
    and high level [neural
    networks](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/nn);
    in addition, users can load pre-trained
    [Caffe](http://caffe.berkeleyvision.org/) or
    [Torch](http://torch.ch/) or
    [Keras](https://faroit.github.io/keras-docs/1.2.2/) models into
    Spark programs using BigDL.
-   **Extremely high performance.** To achieve high performance, BigDL
    uses [Intel MKL](https://software.intel.com/en-us/intel-mkl) and
    multi-threaded programming in each Spark task. Consequently, it is
    orders of magnitude faster than out-of-box open source
    [Caffe](http://caffe.berkeleyvision.org/), [Torch](http://torch.ch/)
    or [TensorFlow](https://www.tensorflow.org/) on a single-node Xeon
    (i.e., comparable with mainstream GPU).
-   **Efficiently scale-out.** BigDL can efficiently scale out to
    perform data analytics at "Big Data scale", by leveraging [Apache
    Spark](http://spark.apache.org/) (a lightning fast distributed data
    processing framework), as well as efficient implementations of
    synchronous SGD and all-reduce communications on Spark.

| 

  --------------------- --------------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Java
  **Project URL**       <https://bigdl-project.github.io/master/>
  **Source Location**   <https://github.com/intel-analytics/BigDL>
  **Tag(s)**            ML, ML Framework
  --------------------- --------------------------------------------

Blocks
======

Blocks is a framework that is supposed to make it easier to build
complicated neural network models on top of
[Theano](http://www.deeplearning.net/software/theano/).

Blocks is a framework that helps you build neural network models on top
of Theano. Currently it supports and provides:

-   Constructing parametrized Theano operations, called "bricks"
-   Pattern matching to select variables and bricks in large models
-   Algorithms to optimize your model
-   Saving and resuming of training
-   Monitoring and analyzing values during training progress (on the
    training set as well as on test sets)
-   Application of graph transformations, such as dropout

| 

  --------------------- -------------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Python
  **Project URL**       <http://blocks.readthedocs.io/en/latest/>
  **Source Location**   <https://github.com/mila-udem/blocks>
  **Tag(s)**            ML, ML Framework
  --------------------- -------------------------------------------

Caffe
=====

Caffe is a deep learning framework made with expression, speed, and
modularity in mind. It is developed by Berkeley AI Research
([BAIR](http://bair.berkeley.edu))/The Berkeley Vision and Learning
Center (BVLC) and community contributors.

Caffe is an Open framework, models, and worked examples for deep
learning:

-   4.5 years old
-   7,000+ citations, 250+ contributors, 24,000+ stars
-   15,000+ forks, \>1 pull request / day average at peak

Focus has been vision, but also handles , reinforcement learning, speech
and text.

Why Caffe?

-   **Expressive architecture** encourages application and innovation.
    Models and optimization are defined by configuration without
    hard-coding. Switch between CPU and GPU by setting a single flag to
    train on a GPU machine then deploy to commodity clusters or mobile
    devices.
-   **Extensible code** fosters active development. In Caffe's first
    year, it has been forked by over 1,000 developers and had many
    significant changes contributed back. Thanks to these contributors
    the framework tracks the state-of-the-art in both code and models.
-   **Speed** makes Caffe perfect for research experiments and industry
    deployment. Caffe can process **over 60M images per day** with a
    single NVIDIA K40 GPU\*. That's 1 ms/image for inference and 4
    ms/image for learning and more recent library versions and hardware
    are faster still. We believe that Caffe is among the fastest convnet
    implementations available.

| 

  --------------------- ----------------------------------------------------
  **SBB License**       BSD License 2.0 (3-clause, New or Revised) License
  **Core Technology**   CPP
  **Project URL**       <http://caffe.berkeleyvision.org/>
  **Source Location**   <https://github.com/BVLC/caffe>
  **Tag(s)**            ML, ML Framework
  --------------------- ----------------------------------------------------

ConvNetJS
=========

ConvNetJS is a Javascript library for training Deep Learning models
(Neural Networks) entirely in your browser. Open a tab and you're
training. No software requirements, no compilers, no installations, no
GPUs, no sweat.

ConvNetJS is a Javascript implementation of Neural networks, together
with nice browser-based demos. It currently supports:

-   Common **Neural Network modules** (fully connected layers,
    non-linearities)
-   Classification (SVM/Softmax) and Regression (L2) **cost functions**
-   Ability to specify and train **Convolutional Networks** that process
    images
-   An experimental **Reinforcement Learning** module, based on Deep Q
    Learning

For much more information, see the main page at
[convnetjs.com](http://convnetjs.com)

Note: Not actively maintained, but still useful to prevent reinventing
the wheel.

| 

  --------------------- ------------------------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Javascript
  **Project URL**       <https://cs.stanford.edu/people/karpathy/convnetjs/>
  **Source Location**   <https://github.com/karpathy/convnetjs>
  **Tag(s)**            Javascript, ML, ML Framework
  --------------------- ------------------------------------------------------

Datumbox
========

The Datumbox Machine Learning Framework is an open-source framework
written in Java which allows the rapid development Machine Learning and
Statistical applications. The main focus of the framework is to include
a large number of machine learning algorithms & statistical methods and
to be able to handle large sized datasets.

Datumbox comes with a large number of pre-trained models which allow you
to perform Sentiment Analysis (Document & Twitter), Subjectivity
Analysis, Topic Classification, Spam Detection, Adult Content Detection,
Language Detection, Commercial Detection, Educational Detection and
Gender Detection.

Datumbox is not supported by a large team of commercial developers or
large group of FOSS developers. Basically one developer maintains it as
a side project. So review this FOSS project before you make large
investments building applications on top of it.

| 

  --------------------- --------------------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Java
  **Project URL**       <http://www.datumbox.com/>
  **Source Location**   <https://github.com/datumbox/datumbox-framework>
  **Tag(s)**            ML, ML Framework
  --------------------- --------------------------------------------------

DeepDetect
==========

DeepDetect implements support for supervised and unsupervised deep
learning of images, text and other data, with focus on simplicity and
ease of use, test and connection into existing applications. It supports
classification, object detection, segmentation, regression, autoencoders
and more.

It has Python and other client libraries.

Deep Detect has also a REST API for Deep Learning with:

-   JSON communication format
-   Pre-trained models
-   Neural architecture templates
-   Python, Java, C\# clients
-   Output templating

| 

  --------------------- ---------------------------------------
  **SBB License**       MIT License
  **Core Technology**   C++
  **Project URL**       <https://deepdetect.com>
  **Source Location**   <https://github.com/beniz/deepdetect>
  **Tag(s)**            ML, ML Framework
  --------------------- ---------------------------------------

Deeplearning4j
==============

Deep Learning for Java, Scala & Clojure on Hadoop & Spark With GPUs.

Eclipse Deeplearning4J is an distributed neural net library written in
Java and Scala.

Eclipse Deeplearning4j a commercial-grade, open-source, distributed
deep-learning library written for Java and Scala. DL4J is designed to be
used in business environments on distributed GPUs and CPUs.

Deeplearning4J integrates with Hadoop and Spark and runs on several
backends that enable use of CPUs and GPUs. The aim of this project is to
create a plug-and-play solution that is more convention than
configuration, and which allows for fast prototyping. This project is
created by Skymind who delivers support and offers also the option for
machine learning models to be hosted with Skymind's model server on a
cloud environment

| 

  --------------------- ----------------------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Java
  **Project URL**       <https://deeplearning4j.org>
  **Source Location**   <https://github.com/deeplearning4j/deeplearning4j>
  **Tag(s)**            ML, ML Framework
  --------------------- ----------------------------------------------------

Detectron2
==========

Detectron is Facebook AI Research's software system that implements
state-of-the-art object detection algorithms, including [Mask
R-CNN](https://arxiv.org/abs/1703.06870). Detectron2 is a ground-up
rewrite of Detectron that started with maskrcnn-benchmark. The platform
is now implemented in [PyTorch](https://pytorch.org/). With a new, more
modular design. Detectron2 is flexible and extensible, and able to
provide fast training on single or multiple GPU servers. Detectron2
includes high-quality implementations of state-of-the-art object
detection algorithms,

New in Detctron 2:

-   It is powered by the [PyTorch](https://pytorch.org) deep learning
    framework.
-   Includes more features such as panoptic segmentation, densepose,
    Cascade R-CNN, rotated bounding boxes, etc.
-   Can be used as a library to support [different
    projects](https://github.com/facebookresearch/detectron2/blob/master/projects)
    on top of it. We'll open source more research projects in this way.
-   It [trains much
    faster](https://detectron2.readthedocs.io/notes/benchmarks.html).

The goal of Detectron is to provide a high-quality, high-performance
codebase for object detection *research*. It is designed to be flexible
in order to support rapid implementation and evaluation of novel
research.

A number of Facebook teams use this platform to train custom models for
a variety of applications including augmented reality and community
integrity. Once trained, these models can be deployed in the cloud and
on mobile devices, powered by the highly efficient Caffe2 runtime.

Documentation on: <https://detectron2.readthedocs.io/index.html>

| 

  --------------------- --------------------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Python
  **Project URL**       <https://github.com/facebookresearch/Detectron2>
  **Source Location**   <https://github.com/facebookresearch/detectron2>
  **Tag(s)**            ML, ML Framework, Python
  --------------------- --------------------------------------------------

Dopamine
========

Dopamine is a research framework for fast prototyping of reinforcement
learning algorithms. It aims to fill the need for a small, easily
grokked codebase in which users can freely experiment with wild ideas
(speculative research).

Our design principles are:

-   *Easy experimentation*: Make it easy for new users to run benchmark
    experiments.
-   *Flexible development*: Make it easy for new users to try out
    research ideas.
-   *Compact and reliable*: Provide implementations for a few,
    battle-tested algorithms.
-   *Reproducible*: Facilitate reproducibility in results.

| 

  --------------------- ------------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Python
  **Project URL**       <https://github.com/google/dopamine>
  **Source Location**   <https://github.com/google/dopamine>
  **Tag(s)**            ML, ML Framework, Reinforcement Learning
  --------------------- ------------------------------------------

Fastai
======

The fastai library simplifies training fast and accurate neural nets
using modern best practices. Fast.ai's mission is to make the power of
state of the art deep learning available to anyone. fastai sits on top
of [PyTorch](https://pytorch.org/), which provides the foundation.

fastai is a deep learning library which provides high-level components
that can quickly and easily provide state-of-the-art results in standard
deep learning domains, and provides researchers with low-level
components that can be mixed and matched to build new approaches. It
aims to do both things without substantial compromises in ease of use,
flexibility, or performance.

Docs can be found on: <http://docs.fast.ai/>

| 

  --------------------- -------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Python
  **Project URL**       <http://www.fast.ai/>
  **Source Location**   <https://github.com/fastai/fastai/>
  **Tag(s)**            ML, ML Framework
  --------------------- -------------------------------------

Featuretools
============

*One of the holy grails of machine learning is to automate more and more
of the feature engineering process."* ― Pedro

[Featuretools](https://www.featuretools.com) is a python library for
automated feature engineering. Featuretools automatically creates
features from temporal and relational datasets. Featuretools works
alongside tools you already use to build machine learning pipelines. You
can load in pandas dataframes and automatically create meaningful
features in a fraction of the time it would take to do manually.

Featuretools is a python library for automated feature engineering.
Featuretools can automatically create a single table of features for any
"target entity".

Featuretools is a framework to perform automated feature engineering. It
excels at transforming transactional and relational datasets into
feature matrices for machine learning.

| 

  --------------------- ----------------------------------------------------
  **SBB License**       BSD License 2.0 (3-clause, New or Revised) License
  **Core Technology**   Python
  **Project URL**       <https://www.featuretools.com/>
  **Source Location**   <https://github.com/Featuretools/featuretools>
  **Tag(s)**            ML, ML Framework, Python
  --------------------- ----------------------------------------------------

FlyingSquid
===========

FlyingSquid is a ML framework for automatically building models from
multiple noisy label sources. Users write functions that generate noisy
labels for data, and FlyingSquid uses the agreements and disagreements
between them to learn a *label model* of how accurate the *labeling
functions* are. The label model can be used directly for downstream
applications, or it can be used to train a powerful end model.

| 

  --------------------- ------------------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Python
  **Project URL**       <http://hazyresearch.stanford.edu/flyingsquid>
  **Source Location**   <https://github.com/HazyResearch/flyingsquid>
  **Tag(s)**            ML Framework, Python
  --------------------- ------------------------------------------------

Karate Club
===========

Karate Club is an unsupervised machine learning extension library for
[NetworkX](https://networkx.github.io/).

*Karate Club* consists of state-of-the-art methods to do unsupervised
learning on graph structured data. To put it simply it is a Swiss Army
knife for small-scale graph mining research. First, it provides network
embedding techniques at the node and graph level. Second, it includes a
variety of overlapping and non-overlapping community detection methods.
Implemented methods cover a wide range of network science (NetSci,
Complenet), data mining ([ICDM](http://icdm2019.bigke.org/),
[CIKM](http://www.cikm2019.net/), [KDD](https://www.kdd.org/kdd2020/)),
artificial intelligence
([AAAI](http://www.aaai.org/Conferences/conferences.php),
[IJCAI](https://www.ijcai.org/)) and machine learning
([NeurIPS](https://nips.cc/), [ICML](https://icml.cc/),
[ICLR](https://iclr.cc/)) conferences, workshops, and pieces from
prominent journals.

The documentation can be found at:
<https://karateclub.readthedocs.io/en/latest/>

The Karate ClubAPI draws heavily from the ideas of scikit-learn and
theoutput generated is suitable as input for scikit-learn's
machinelearning procedures.

The paper can be found at: <https://arxiv.org/pdf/2003.04819.pdf>

| 

  --------------------- -----------------------------------------------------
  **SBB License**       GNU General Public License (GPL) 3.0
  **Core Technology**   Python
  **Project URL**       <https://karateclub.readthedocs.io/en/latest/>
  **Source Location**   <https://github.com/benedekrozemberczki/karatecluB>
  **Tag(s)**            ML Framework
  --------------------- -----------------------------------------------------

Keras
=====

Keras is a high-level neural networks API, written in Python and capable
of running on top of TensorFlow, CNTK, or Theano. It was developed with
a focus on enabling fast experimentation. Being able to go from idea to
result with the least possible delay is key to doing good research.

Use Keras if you need a deep learning library that:

-   Allows for easy and fast prototyping (through user friendliness,
    modularity, and extensibility).
-   Supports both convolutional networks and recurrent networks, as well
    as combinations of the two.
-   Runs seamlessly on CPU and GPU.

| 

  --------------------- ---------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Python
  **Project URL**       <https://keras.io/>
  **Source Location**   <https://github.com/keras-team/keras>
  **Tag(s)**            ML, ML Framework
  --------------------- ---------------------------------------

learn2learn
===========

learn2learn is a PyTorch library for meta-learning implementations.

The goal of meta-learning is to enable agents to *learn how to learn*.
That is, we would like our agents to become better learners as they
solve more and more tasks.

Features:

learn2learn provides high- and low-level utilities for meta-learning.
The high-level utilities allow arbitrary users to take advantage of
exisiting meta-learning algorithms. The low-level utilities enable
researchers to develop new and better meta-learning algorithms.

Some features of learn2learn include:

-   Modular API: implement your own training loops with our low-level
    utilities.
-   Provides various meta-learning algorithms (e.g. MAML, FOMAML,
    MetaSGD, ProtoNets, DiCE)
-   Task generator with unified API, compatible with torchvision,
    torchtext, torchaudio, and cherry.
-   Provides standardized meta-learning tasks for vision (Omniglot,
    mini-ImageNet), reinforcement learning (Particles, Mujoco), and even
    text (news classification).
-   100% compatible with PyTorch --- use your own modules, datasets, or
    libraries!

| 

  --------------------- ----------------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Python
  **Project URL**       <http://learn2learn.net/>
  **Source Location**   <https://github.com/learnables/learn2learn/>
  **Tag(s)**            ML Framework
  --------------------- ----------------------------------------------

Lore
====

Lore is a python framework to make machine learning approachable for
Engineers and maintainable for Data Scientists.

Features

-   Models support hyper parameter search over estimators with a data
    pipeline. They will efficiently utilize multiple GPUs (if available)
    with a couple different strategies, and can be saved and distributed
    for horizontal scalability.
-   Estimators from multiple packages are supported:
    [Keras](https://keras.io/) (TensorFlow/Theano/CNTK),
    [XGBoost](https://xgboost.readthedocs.io/) and [SciKit
    Learn](http://scikit-learn.org/stable/). They can all be subclassed
    with build, fit or predict overridden to completely customize your
    algorithm and architecture, while still benefiting from everything
    else.
-   Pipelines avoid information leaks between train and test sets, and
    one pipeline allows experimentation with many different estimators.
    A disk based pipeline is available if you exceed your machines
    available RAM.
-   Transformers standardize advanced feature engineering. For example,
    convert an American first name to its statistical age or gender
    using US Census data. Extract the geographic area code from a free
    form phone number string. Common date, time and string operations
    are supported efficiently through pandas.
-   Encoders offer robust input to your estimators, and avoid common
    problems with missing and long tail values. They are well tested to
    save you from garbage in/garbage out.
-   IO connections are configured and pooled in a standard way across
    the app for popular (no)sql databases, with transaction management
    and read write optimizations for bulk data, rather than typical ORM
    single row operations. Connections share a configurable query cache,
    in addition to encrypted S3 buckets for distributing models and
    datasets.
-   Dependency Management for each individual app in development, that
    can be 100% replicated to production. No manual activation, or magic
    env vars, or hidden files that break python for everything else. No
    knowledge required of venv, pyenv, pyvenv, virtualenv,
    virtualenvwrapper, pipenv, conda. Ain't nobody got time for that.
-   Tests for your models can be run in your Continuous Integration
    environment, allowing Continuous Deployment for code and training
    updates, without increased work for your infrastructure team.
-   Workflow Support whether you prefer the command line, a python
    console, jupyter notebook, or IDE. Every environment gets readable
    logging and timing statements configured for both production and
    development.

| 

  --------------------- --------------------------------------
  **SBB License**       GNU General Public License (GPL) 2.0
  **Core Technology**   Python
  **Project URL**       <https://github.com/instacart/lore>
  **Source Location**   <https://github.com/instacart/lore>
  **Tag(s)**            ML, ML Framework, Python
  --------------------- --------------------------------------

Microsoft Cognitive Toolkit (CNTK)
==================================

The Microsoft Cognitive Toolkit (<https://cntk.ai>) is a unified deep
learning toolkit that describes neural networks as a series of
computational steps via a directed graph. In this directed graph, leaf
nodes represent input values or network parameters, while other nodes
represent matrix operations upon their inputs. CNTK allows users to
easily realize and combine popular model types such as feed-forward
DNNs, convolutional nets (CNNs), and recurrent networks (RNNs/LSTMs). It
implements stochastic gradient descent (SGD, error backpropagation)
learning with automatic differentiation and parallelization across
multiple GPUs and servers. CNTK has been available under an open-source
license since April 2015.

Docs on: <https://docs.microsoft.com/en-us/cognitive-toolkit/>

| 

  --------------------- -------------------------------------------------------
  **SBB License**       MIT License
  **Core Technology**   C++
  **Project URL**       <https://docs.microsoft.com/en-us/cognitive-toolkit/>
  **Source Location**   <https://github.com/Microsoft/CNTK>
  **Tag(s)**            ML, ML Framework
  --------------------- -------------------------------------------------------

ml5.js
======

ml5.js aims to make machine learning approachable for a broad audience
of artists, creative coders, and students. The library provides access
to machine learning algorithms and models in the browser, building on
top of [TensorFlow.js](https://js.tensorflow.org/) with no other
external dependencies.

The library is supported by code examples, tutorials, and sample data
sets with an emphasis on ethical computing. Bias in data, stereotypical
harms, and responsible crowdsourcing are part of the documentation
around data collection and usage.

ml5.js is heavily inspired by [Processing](https://processing.org/) and
[p5.js](https://p5js.org/).

| 

  --------------------- ----------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Javascript
  **Project URL**       <https://ml5js.org/>
  **Source Location**   <https://github.com/ml5js/ml5-library>
  **Tag(s)**            Javascript, ML, ML Framework
  --------------------- ----------------------------------------

Mljar
=====

MLJAR is a platform for rapid prototyping, developing and deploying
machine learning models.

MLJAR makes algorithm search and tuning painless. It checks many
different algorithms for you. For each algorithm hyper-parameters are
separately tuned. All computations run in parallel in MLJAR cloud, so
you get your results very quickly. At the end the ensemble of models is
created, so your predictive model will be super accurate.

There are two types of interface available in MLJAR:

-   you can run Machine Learning models in your browser, you don't need
    to code anything. Just upload dataset, click which attributes to
    use, which algorithms to use and go! This makes Machine Learning
    super easy for everyone and make it possible to get really useful
    models,
-   there is a python wrapper over MLJAR API, so you don't need to open
    any browser or click on any button, just write fancy python code! We
    like it and hope you will like it too! To start using MLJAR python
    package please go to our
    [github](https://github.com/mljar/mljar-api-python).

| 

  --------------------- ---------------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Python
  **Project URL**       <https://mljar.com/>
  **Source Location**   <https://github.com/mljar/mljar-supervised>
  **Tag(s)**            ML, ML Framework, Python
  --------------------- ---------------------------------------------

MLsquare
========

\[ML\]² -- ML Square is a python library that utilises deep learning
techniques to:

-   Enable interoperability between existing standard machine learning
    frameworks.
-   Provide explainability as a first-class function.
-   Make ML self learnable.

The following are the design goals:

-   Bring Your Own Spec First.
-   Bring Your Own Experience First.
-   Consistent.
-   Compositional.
-   Modular.
-   Extensible

See <https://arxiv.org/pdf/2001.00818.pdf> for a in depth explanation.

| 

  --------------------- ----------------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Python
  **Project URL**       <https://mlsquare.readthedocs.io/en/latest/>
  **Source Location**   <https://github.com/mlsquare/mlsquare>
  **Tag(s)**            ML Framework
  --------------------- ----------------------------------------------

NeuralStructuredLearning
========================

Neural Structured Learning (NSL) is a new learning paradigm to train
neural networks by leveraging structured signals in addition to feature
inputs. Structure can be explicit as represented by a graph or implicit
as induced by adversarial perturbation.

Structured signals are commonly used to represent relations or
similarity among samples that may be labeled or unlabeled. Leveraging
these signals during neural network training harnesses both labeled and
unlabeled data, which can improve model accuracy, particularly when the
amount of labeled data is relatively small. Additionally, models trained
with samples that are generated by adversarial perturbation have been
shown to be robust against malicious attacks, which are designed to
mislead a model's prediction or classification.

NSL generalizes to Neural Graph Learning as well as to Adversarial
Learning. The NSL framework in TensorFlow provides the following
easy-to-use APIs and tools for developers to train models with
structured signals:

-   **Keras APIs** to enable training with graphs (explicit structure)
    and adversarial pertubations (implicit structure).
-   **TF ops and functions** to enable training with structure when
    using lower-level TensorFlow APIs
-   **Tools** to build graphs and construct graph inputs for training

NSL is part of the TensorFlow framework. More info on:
<https://www.tensorflow.org/neural_structured_learning/>

| 

  --------------------- -------------------------------------------------
  **SBB License**       Apache License 2.0

  **Core Technology**   Python

  **Project URL**       <https://w>
                        ww.tensorflow.org/neural\_structured\_learning/

  **Source Location**   <https://git>
                        hub.com/tensorflow/neural-structured-learning

  **Tag(s)**            ML, ML Framework, Python
  --------------------- -------------------------------------------------

NNI (Neural Network Intelligence)
=================================

NNI (Neural Network Intelligence) is a toolkit to help users run
automated machine learning (AutoML) experiments. The tool dispatches and
runs trial jobs generated by tuning algorithms to search the best neural
architecture and/or hyper-parameters in different environments like
local machine, remote servers and cloud. (Microsoft ML project)

Who should consider using NNI:

-   Those who want to try different AutoML algorithms in their training
    code (model) at their local machine.
-   Those who want to run AutoML trial jobs in different environments to
    speed up search (e.g. remote servers and cloud).
-   Researchers and data scientists who want to implement their own
    AutoML algorithms and compare it with other algorithms.
-   ML Platform owners who want to support AutoML in their platform.

| 

  --------------------- -----------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Python
  **Project URL**       <https://nni.readthedocs.io/en/latest/>
  **Source Location**   <https://github.com/Microsoft/nni>
  **Tag(s)**            ML, ML Framework
  --------------------- -----------------------------------------

NuPIC
=====

The Numenta Platform for Intelligent Computing (**NuPIC**) is a machine
intelligence platform that implements the [HTM learning
algorithms](https://numenta.com/resources/papers-videos-and-more/). HTM
is a detailed computational theory of the neocortex. At the core of HTM
are time-based continuous learning algorithms that store and recall
spatial and temporal patterns. NuPIC is suited to a variety of problems,
particularly anomaly detection and prediction of streaming data sources.

Note: This project is in Maintenance Mode.

| 

  --------------------- ---------------------------------------------
  **SBB License**       GNU Affero General Public License Version 3
  **Core Technology**   Python
  **Project URL**       <https://numenta.org/>
  **Source Location**   <https://github.com/numenta/nupic>
  **Tag(s)**            ML Framework, Python
  --------------------- ---------------------------------------------

Plato
=====

The Plato Research Dialogue System is a flexible framework that can be
used to create, train, and evaluate conversational AI agents in various
environments. It supports interactions through speech, text, or dialogue
acts and each conversational agent can interact with data, human users,
or other conversational agents (in a multi-agent setting). Every
component of every agent can be trained independently online or offline
and Plato provides an easy way of wrapping around virtually any existing
model, as long as Plato's interface is adhered to.

OSS by Uber.

| 

  --------------------- -----------------------------------------------
  **SBB License**       MIT License

  **Core Technology**   Python

  **Project URL**       <https://github.com>
                        /uber-research/plato-research-dialogue-system

  **Source Location**   <https://github.com>
                        /uber-research/plato-research-dialogue-system

  **Tag(s)**            ML, ML Framework
  --------------------- -----------------------------------------------

Polyaxon
========

A platform for reproducible and scalable machine learning and deep
learning on kubernetes

Polyaxon is a platform for building, training, and monitoring large
scale deep learning applications.

Polyaxon deploys into any data center, cloud provider, or can be hosted
and managed by Polyaxon, and it supports all the major deep learning
frameworks such as Tensorflow, MXNet, Caffe, Torch, etc.

Polyaxon makes it faster, easier, and more efficient to develop deep
learning applications by managing workloads with smart container and
node management. And it turns GPU servers into shared, self-service
resources for your team or organization.

| 

  --------------------- ----------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Python
  **Project URL**       <https://polyaxon.com/>
  **Source Location**   <https://github.com/polyaxon/polyaxon>
  **Tag(s)**            ML, ML Framework
  --------------------- ----------------------------------------

PyCaret
=======

PyCaret is an open source `low-code` machine learning library in Python
that aims to reduce the hypothesis to insights cycle time in a ML
experiment. It enables data scientists to perform end-to-end experiments
quickly and efficiently. In comparison with the other open source
machine learning libraries, PyCaret is an alternate low-code library
that can be used to perform complex machine learning tasks with only few
lines of code. PyCaret is essentially a Python wrapper around several
machine learning libraries and frameworks such as `scikit-learn`,
`XGBoost`, `Microsoft LightGBM`, `spaCy` and many more.

The design and simplicity of PyCaret is inspired by the emerging role of
`citizen data scientists`, a term first used by Gartner. Citizen Data
Scientists are `power users` who can perform both simple and moderately
sophisticated analytical tasks that would previously have required more
expertise. Seasoned data scientists are often difficult to find and
expensive to hire but citizen data scientists can be an effective way to
mitigate this gap and address data related challenges in business
setting.

PyCaret claims to be `imple`, `easy to use` and `deployment ready`. All
the steps performed in a ML experiment can be reproduced using a
pipeline that is automatically developed and orchestrated in PyCaret as
you progress through the experiment. A `pipeline` can be saved in a
binary file format that is transferable across environments.

| 

  --------------------- --------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Python
  **Project URL**       <https://www.pycaret.org>
  **Source Location**   <https://github.com/pycaret/pycaret>
  **Tag(s)**            ML Framework
  --------------------- --------------------------------------

Pylearn2
========

Pylearn2 is a library designed to make machine learning research easy.

This project does not have any current developer

| 

  --------------------- ----------------------------------------------------
  **SBB License**       BSD License 2.0 (3-clause, New or Revised) License
  **Core Technology**   Python
  **Project URL**       <http://deeplearning.net/software/pylearn2/>
  **Source Location**   <https://github.com/lisa-lab/pylearn2>
  **Tag(s)**            ML, ML Framework
  --------------------- ----------------------------------------------------

Pyro
====

Deep universal probabilistic programming with Python and PyTorch. Pyro
is in an alpha release. It is developed and used by[Uber AI
Labs](http://uber.ai).

Pyro is a universal probabilistic programming language (PPL) written in
Python and supported by [PyTorch](http://pytorch.org) on the backend.
Pyro enables flexible and expressive deep probabilistic modeling,
unifying the best of modern deep learning and Bayesian modeling. It was
designed with these key principles:

-   Universal: Pyro can represent any computable probability
    distribution.
-   Scalable: Pyro scales to large data sets with little overhead.
-   Minimal: Pyro is implemented with a small core of powerful,
    composable abstractions.
-   Flexible: Pyro aims for automation when you want it, control when
    you need it.

Documentation on: <http://docs.pyro.ai/>

| 

  --------------------- --------------------------------------
  **SBB License**       GNU General Public License (GPL) 2.0
  **Core Technology**   Python
  **Project URL**       <http://pyro.ai/>
  **Source Location**   <https://github.com/uber/pyro>
  **Tag(s)**            ML, ML Framework, Python
  --------------------- --------------------------------------

Pythia
======

Pythia is a modular framework for supercharging vision and language
research built on top of PyTorch created by Facebook.

You can use Pythia to bootstrap for your next vision and language
multimodal research project. Pythia can also act as starter codebase for
challenges around vision and language datasets (TextVQA challenge, VQA
challenge).

It features:

-   **Model Zoo**: Reference implementations for state-of-the-art vision
    and language model including
    [LoRRA](https://arxiv.org/abs/1904.08920) (SoTA on VQA and TextVQA),
    [Pythia](https://arxiv.org/abs/1807.09956) model (VQA 2018 challenge
    winner) and
    [BAN](https://github.com/facebookresearch/pythia/blob/master).
-   **Multi-Tasking**: Support for multi-tasking which allows training
    on multiple dataset together.
-   **Datasets**: Includes support for various datasets built-in
    including VQA, VizWiz, TextVQA and VisualDialog.
-   **Modules**: Provides implementations for many commonly used layers
    in vision and language domain
-   **Distributed**: Support for distributed training based on
    DataParallel as well as DistributedDataParallel.
-   **Unopinionated**: Unopinionated about the dataset and model
    implementations built on top of it.
-   **Customization**: Custom losses, metrics, scheduling, optimizers,
    tensorboard; suits all your custom needs.

| 

  --------------------- -----------------------------------------------
  **SBB License**       BSD License 2.0 (3-clause, New or Revised)
                        License

  **Core Technology**   Python

  **Project URL**       <https://le>
                        arnpythia.readthedocs.io/en/latest/index.html

  **Source Location**   <https://github.com/facebookresearch/pythia>

  **Tag(s)**            ML, ML Framework, Python
  --------------------- -----------------------------------------------

PyTorch
=======

PyTorch is a Python-first machine learning framework that is utilized
heavily towards deep learning. It supports CUDA technology (From NVIDIA)
to fully use the the power of the dedicated GPUs in training, analyzing
and validating neural networks models.

Deep learning frameworks have often focused on either usability or
speed, but not both. PyTorch is a machine learning library that shows
that these two goals are in fact compatible: it provides an imperative
and Pythonic programming style that supports code as a model, makes
debugging easy and is consistent with other popular scientific computing
libraries, while remaining efficient and supporting hardware
accelerators such as GPUs.

PyTorch is very widely used, and is under active development and
support. PyTorch is:

-   a deep learning framework that puts Python first.
-    a research-focused framework.
-   Python package that provides two high-level features:

Pytorch uses tensor computation (like NumPy) with strong GPU
acceleration. It can use deep neural networks built on a tape-based
autograd system.

PyTorch is a Python package that provides two high-level features:

-   Tensor computation (like NumPy) with strong GPU acceleration
-   Deep neural networks built on a tape-based autograd system

You can reuse your favorite Python packages such as NumPy, SciPy and
Cython to extend PyTorch when needed. PyTorch has become a popular tool
in the deep learning research community by combining a focus on
usability with careful performance considerations.

A very good overview of the design principles and architecture of
PyTorch can be found in this paper
<https://arxiv.org/pdf/1912.01703.pdf> .

| 

  --------------------- --------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Python
  **Project URL**       <http://pytorch.org/>
  **Source Location**   <https://github.com/pytorch/pytorch>
  **Tag(s)**            ML, ML Framework
  --------------------- --------------------------------------

ReAgent
=======

ReAgent is an open source end-to-end platform for applied reinforcement
learning (RL) developed and used at Facebook. ReAgent is built in Python
and uses PyTorch for modeling and training and TorchScript for model
serving. The platform contains workflows to train popular deep RL
algorithms and includes data preprocessing, feature transformation,
distributed training, counterfactual policy evaluation, and optimized
serving. For more detailed information about ReAgent see the white paper
[here](https://research.fb.com/publications/horizon-facebooks-open-source-applied-reinforcement-learning-platform/).

The platform was once named "Horizon" but we have adopted the name
"ReAgent" recently to emphasize its broader scope in decision making and
reasoning.

| 

  --------------------- -------------------------------------------------------
  **SBB License**       BSD License 2.0 (3-clause, New or Revised) License
  **Core Technology**   Python
  **Project URL**       <https://engineering.fb.com/ml-applications/horizon/>
  **Source Location**   <https://github.com/facebookresearch/ReAgent>
  **Tag(s)**            ML, ML Framework, Python
  --------------------- -------------------------------------------------------

RLCard
======

RLCard is a toolkit for Reinforcement Learning (RL) in card games. It
supports multiple card environments with easy-to-use interfaces. The
goal of RLCard is to bridge reinforcement learning and imperfect
information games, and push forward the research of reinforcement
learning in domains with multiple agents, large state and action space,
and sparse reward. RLCard is developed by [DATA
Lab](http://faculty.cs.tamu.edu/xiahu/) at Texas A&M University.

-   Paper: <https://arxiv.org/abs/1910.04376>

| 

  --------------------- ---------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Python
  **Project URL**       <http://rlcard.org/>
  **Source Location**   <https://github.com/datamllab/rlcard>
  **Tag(s)**            ML Framework, Python
  --------------------- ---------------------------------------

Scikit-learn
============

scikit-learn is a Python module for machine learning. s cikit-learn is a
Python module for machine learning built on top of SciPy and is
distributed under the 3-Clause BSD license.

Key features:

-   Simple and efficient tools for predictive data analysis
-   Accessible to everybody, and reusable in various contexts
-   Built on NumPy, SciPy, and matplotlib
-   Open source, commercially usable -- BSD license

| 

  --------------------- ----------------------------------------------------
  **SBB License**       BSD License 2.0 (3-clause, New or Revised) License
  **Core Technology**   Python
  **Project URL**       <http://scikit-learn.org>
  **Source Location**   <https://github.com/scikit-learn/scikit-learn>
  **Tag(s)**            ML, ML Framework
  --------------------- ----------------------------------------------------

SINGA
=====

Distributed deep learning system.

SINGA was initiated by the DB System Group at National University of
Singapore in 2014, in collaboration with the database group of Zhejiang
University.

SINGA's software stack includes three major components, namely, core, IO
and model:

1.  The core component provides memory management and tensor operations.
2.  IO has classes for reading (and writing) data from (to) disk and
    network.
3.  The model component provides data structures and algorithms for
    machine learning models, e.g., layers for neural network models,
    optimizers/initializer/metric/loss for general machine learning
    models.

| 

  --------------------- -----------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Java
  **Project URL**       <http://singa.apache.org/>
  **Source Location**   <https://github.com/apache/singa>
  **Tag(s)**            ML Framework
  --------------------- -----------------------------------

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

Tensorflow
==========

TensorFlow is an Open Source Software Library for Machine Intelligence.
TensorFlow is by far the most used and popular ML open source project.
And since the first initial release was only just in November 2015 it is
expected that the impact of this OSS package will expand even more.

TensorFlow™ is an open source software library for numerical computation
using data flow graphs. Nodes in the graph represent mathematical
operations, while the graph edges represent the multidimensional data
arrays (tensors) communicated between them. The flexible architecture
allows you to deploy computation to one or more CPUs or GPUs in a
desktop, server, or mobile device with a single API. TensorFlow was
originally developed by researchers and engineers working on the Google
Brain Team within Google's Machine Intelligence research organization
for the purposes of conducting machine learning and deep neural networks
research, but the system is general enough to be applicable in a wide
variety of other domains as well.

TensorFlow comes with a tool called
[TensorBoard](https://www.tensorflow.org/versions/r0.11/how_tos/graph_viz/index.html)
which you can use to get some insight into what is happening.
TensorBoard is a suite of web applications for inspecting and
understanding your TensorFlow runs and graphs.

There is also a version of TensorFlow that runs in a browser. This is
TensorFlow.js (<https://js.tensorflow.org/> ). TensorFlow.js is a WebGL
accelerated, browser based JavaScript library for training and deploying
ML models.

Since privacy is a contentious fight TensorFlow has now (2020) also a
library called 'TensorFlow Privacy' . This is a python library that
includes implementations of TensorFlow optimizers for training machine
learning models with differential privacy. The library comes with
tutorials and analysis tools for computing the privacy guarantees
provided. See: <https://github.com/tensorflow/privacy>

| 

  --------------------- --------------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   C
  **Project URL**       <https://www.tensorflow.org/>
  **Source Location**   <https://github.com/tensorflow/tensorflow>
  **Tag(s)**            ML, ML Framework
  --------------------- --------------------------------------------

TF Encrypted
============

TF Encrypted is a framework for encrypted machine learning in
TensorFlow. It looks and feels like TensorFlow, taking advantage of the
ease-of-use of the Keras API while enabling training and prediction over
encrypted data via secure multi-party computation and homomorphic
encryption. TF Encrypted aims to make privacy-preserving machine
learning readily available, without requiring expertise in cryptography,
distributed systems, or high performance computing.

| 

  --------------------- ------------------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Python
  **Project URL**       <https://tf-encrypted.io/>
  **Source Location**   <https://github.com/tf-encrypted/tf-encrypted>
  **Tag(s)**            ML, ML Framework, Privacy
  --------------------- ------------------------------------------------

Theano
======

Theano is a Python library that allows you to define, optimize, and
evaluate mathematical expressions involving multi-dimensional arrays
efficiently. It can use GPUs and perform efficient symbolic
differentiation.

Note: After almost ten years of development the company behind Theano
has stopped development and support(Q4-2017). But this library has been
an innovation driver for many other OSS ML packages!

Since a lot of ML libraries and packages use Theano you should check (as
always) the health of your ML stack.

| 

  --------------------- ------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Python
  **Project URL**       <http://www.deeplearning.net/>
  **Source Location**   <https://github.com/Theano/Theano>
  **Tag(s)**            ML, ML Framework, Python
  --------------------- ------------------------------------

Thinc
=====

Thinc is the machine learning library powering spaCy. It features a
battle-tested linear model designed for large sparse learning problems,
and a flexible neural network model under development for spaCy v2.0.

Thinc is a lightweight deep learning library that offers an elegant,
type-checked, functional-programming API for composing models, with
support for layers defined in other frameworks such as PyTorch,
TensorFlow and MXNet. You can use Thinc as an interface layer, a
standalone toolkit or a flexible way to develop new models.

Thinc is a practical toolkit for implementing models that follow the
"Embed, encode, attend, predict" architecture. It's designed to be easy
to install, efficient for CPU usage and optimised for NLP and deep
learning with text -- in particular, hierarchically structured input and
variable-length sequences.

| 

  --------------------- --------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Python
  **Project URL**       <https://thinc.ai/>
  **Source Location**   <https://github.com/explosion/thinc>
  **Tag(s)**            ML, ML Framework, NLP, Python
  --------------------- --------------------------------------

Turi
====

Turi Create simplifies the development of custom machine learning
models.Turi is OSS machine learning from Apple.

Turi Create simplifies the development of custom machine learning
models. You don't have to be a machine learning expert to add
recommendations, object detection, image classification, image
similarity or activity classification to your app.

| 

  --------------------- ----------------------------------------------------
  **SBB License**       BSD License 2.0 (3-clause, New or Revised) License
  **Core Technology**   Python
  **Project URL**       <https://github.com/apple/turicreate>
  **Source Location**   <https://github.com/apple/turicreate>
  **Tag(s)**            ML, ML Framework, ML Hosting
  --------------------- ----------------------------------------------------

TuriCreate
==========

This SBB is from Apple. Apple, is with Siri already for a long time
active in machine learning. But even Apple is releasing building blocks
under OSS licenses now.

Turi Create simplifies the development of custom machine learning
models. You don't have to be a machine learning expert to add
recommendations, object detection, image classification, image
similarity or activity classification to your app.

-   **Easy-to-use:** Focus on tasks instead of algorithms
-   **Visual:** Built-in, streaming visualizations to explore your data
-   **Flexible:** Supports text, images, audio, video and sensor data
-   **Fast and Scalable:** Work with large datasets on a single machine
-   **Ready To Deploy:** Export models to Core ML for use in iOS, macOS,
    watchOS, and tvOS apps

| 

  --------------------- ----------------------------------------------------
  **SBB License**       BSD License 2.0 (3-clause, New or Revised) License
  **Core Technology**   Python
  **Project URL**       <https://turi.com/index.html>
  **Source Location**   <https://github.com/apple/turicreate>
  **Tag(s)**            ML, ML Framework, Python
  --------------------- ----------------------------------------------------

Vowpal Wabbit
=============

Vowpal Wabbit is a machine learning system which pushes the frontier of
machine learning with techniques such as online, hashing, allreduce,
reductions, learning2search, active, and interactive learning. There is
a specific focus on reinforcement learning with several contextual
bandit algorithms implemented and the online nature lending to the
problem well. Vowpal Wabbit is a destination for implementing and
maturing state of the art algorithms with performance in mind.

-   **Input Format.** The input format for the learning algorithm is
    substantially more flexible than might be expected. Examples can
    have features consisting of free form text, which is interpreted in
    a bag-of-words way. There can even be multiple sets of free form
    text in different namespaces.
-   **Speed.** The learning algorithm is fast --- similar to the few
    other online algorithm implementations out there. There are several
    optimization algorithms available with the baseline being sparse
    gradient descent (GD) on a loss function.
-   **Scalability.** This is not the same as fast. Instead, the
    important characteristic here is that the memory footprint of the
    program is bounded independent of data. This means the training set
    is not loaded into main memory before learning starts. In addition,
    the size of the set of features is bounded independent of the amount
    of training data using the hashing trick.
-   **Feature Interaction.** Subsets of features can be internally
    paired so that the algorithm is linear in the cross-product of the
    subsets. This is useful for ranking problems. The alternative of
    explicitly expanding the features before feeding them into the
    learning algorithm can be both computation and space intensive,
    depending on how it's handled.

Microsoft Research is a major contributor to Vowpal Wabbit.

| 

  --------------------- -------------------------------------------------
  **SBB License**       MIT License
  **Core Technology**   CPP
  **Project URL**       <https://vowpalwabbit.org/>
  **Source Location**   <https://github.com/VowpalWabbit/vowpal_wabbit>
  **Tag(s)**            ML, ML Framework
  --------------------- -------------------------------------------------

XAI
===

XAI is a Machine Learning library that is designed with AI
explainability in its core. XAI contains various tools that enable for
analysis and evaluation of data and models. The XAI library is
maintained by [The Institute for Ethical AI &
ML](http://ethical.institute/), and it was developed based on the [8
principles for Responsible Machine
Learning](http://ethical.institute/principles.html).

You can find the documentation at
<https://ethicalml.github.io/xai/index.html>.

| 

  --------------------- ----------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Python
  **Project URL**       <https://ethical.institute/index.html>
  **Source Location**   <https://github.com/EthicalML/xai>
  **Tag(s)**            ML, ML Framework, Python
  --------------------- ----------------------------------------
