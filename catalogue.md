Building Blocks for FOSS ML
===========================

This section presents the most widespread, mature and promising open
source ML software available. The purpose of this section is just to
make you curious to maybe try something that suits you.

ML software comes in many different forms. A lot can be written on the
differences on all packages below, the quality or the usability. Truth
is however there is never one best solution. Depending your practical
use case you should make a motivated choice for what package to use.

As with many other evolving technologies in heavy development: Standards
are still lacking, so you must ensure that you can switch to another
application with minimal pain involved. By using a real open source
solution you already have taken the best step! Using OSS makes you are
far more independent than using ML cloud solutions. This because these
work as 'black-box' solutions and by using OSS you can always build your
own migration interfaces if needed. Lock-in for ML is primarily in the
data and your data cleansing process. So always make sure that you keep
full control of all your data and steps involved in the data preparation
steps you follow. The true value for all ML solutions are of course
always the initial data sources used.

Open Machine Learning Frameworks
================================

There are a number of stable and production ready ML frameworks. But
choosing which framework to use depends on the use case. If you want to
experiment with the latest research insights implemented you will make
another choice than if you need to implement your solution in production
into a critical environment. For business use: So doing innovation
experiments and creating machine learning application most of the time
you want a framework that is stable and widely used.

If you have an edge use case experimenting with different frameworks can
be a valid choice.

PyTorch is dominating the research, but is now extending this success to
industry applications. TensorFlow is already used for many production
business cases. But as it is with all software: Transitions from major
versions (from TensorFlow 1.0 to 2.0) is difficult. Interoperability
standards to easily switch from ML framework are not mature for
production use yet.

ML Frameworks
=============

Choosing a machine learning (ML) framework or library to solve your use
case is easier said than done. Selecting a ML Framework involves making
an assessment to decide what is right for your use case. Several factors
are important for this assessment for your use case. E.g.:

-   Easy of use;
-   Support in the market. Some major FOSS ML Frameworks are supported
    by many consultancy firms. But maybe community support using mailing
    lists or internet forums is sufficient to start.
-   Short goal versus long term strategy. Doing fast innovation tracks
    means the cost for starting from scratch again should be low. But if
    you directly focus on a possible production deployment, whether on
    premise or using cloud hosting this can significantly delay startup
    time. Often it is recommended to experiment fast and in a later
    phase take new requirements like maintenance and production
    deployment into account.
-   Research of business use case. Some ML frameworks are focussed on
    innovation and research. If your company is not trying to develop a
    better ML algorithms this may not be the best ML framework for
    experimenting for business use cases.
-   Closed (Commercial) dependencies. Some FOSS frameworks have a
    dependency with a commercial data collection. E.g. many translation
    frameworks need an API key of Google or AWS to function. All costs
    aspects of these dependencies should be taken into account before
    starting. There is nothing wrong with using commercial software, but
    transparency on used data sets and models can be crucial for
    acceptance of your machine learning application.

A special-purpose framework may be better at one aspect than a
general-purpose. But the cost of context switching is high:

-   different languages or APIs
-   different data formats
-   different tuning tricks

Your first model for experimenting should be about getting the
infrastructure and development tools right. Simple models are usually
interpretable. Interpretable models are easier to debug. Complex model
erode boundaries beware of the CACE principle (CACE principle: Changing
Anything Changes Everything)

Acme
----

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
------

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
-------------

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
------------

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
------------------

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
--------

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
-----

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
------

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
-----

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
---------

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
--------

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
----------

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
--------------

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
----------

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
--------

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
------

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
------------

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
-----------

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
-----------

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
-----

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
-----------

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
----

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
----------------------------------

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
------

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
-----

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
--------

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
------------------------

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
---------------------------------

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
-----

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
-----

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
--------

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
-------

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
--------

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
----

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
------

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
-------

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
-------

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
------

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
------------

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
-----

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
---------

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
----------

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
------------

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
------

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
-----

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
----

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
----------

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
-------------

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
---

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

Computer vision
===============

Computer vision is a field that deals with how computers can be made to
gain high-level understanding of digital images and videos. Machine
learning is a good match for image classification.

libfacedetection
----------------

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
------

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
-------------

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
--------

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
---

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
------------

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
-----------

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
-----------------

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
-----------

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
--------

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
----------------

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
-------------------------------------------

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
--------

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

ML Tools
========

Besides FOSS machine learning frameworks there are special tools that
save you time when creating ML applications. This section is a
opinionated collection of FOSS ML tools that can make creating
applications easier.

AI Explainability 360
---------------------

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
------

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
----------------------------------

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
--------

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
----------

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
------

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
-----------------

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
-----

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
------

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
------------

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
------

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
------

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
-------

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
------

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
-------------

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
----

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
------

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
------

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
------

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
------

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
----

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
------

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
-------

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
---------

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
-----------

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
--------

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
------------

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

ML hosting
==========

Machine learning needs an infrastructure stack and various components to
run. For training and for production. Some open frameworks makes
creating ML solutions easier and faster.

BentoML
-------

BentoML makes it easy to serve and deploy machine learning models in the
cloud.

It is an open source framework for machine learning teams to build
cloud-native prediction API services that are ready for production.
BentoML supports most popular ML training frameworks and common
deployment platforms including major cloud providers and
docker/kubernetes.

Documentation on: <https://bentoml.readthedocs.io/en/latest/index.html>

| 

  --------------------- --------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Python
  **Project URL**       <http://BentoML.ai>
  **Source Location**   <https://github.com/bentoml/BentoML>
  **Tag(s)**            ML, ML Hosting, Python
  --------------------- --------------------------------------

Streamlit
---------

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

RAPIDS
------

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

Acumos AI
---------

Acumos AI is a platform and open source framework that makes it easy to
build, share, and deploy AI apps. Acumos standardizes the infrastructure
stack and components required to run an out-of-the-box general AI
environment.

Acumos is a platform which enhances the development, training and
deployment of AI models. Its purpose is to scale up the introduction of
AI-based software across a wide range of industrial and commercial
problems in order to reach a critical mass of applications. In this way,
Acumos will drive toward a data-centric process for producing software
based upon machine learning as the central paradigm. The platform seeks
to empower data scientists to publish more adaptive AI models and shield
them from the task of custom development of fully integrated solutions.
Ideally, software developers will use Acumos to change the process of
software development from a code-writing and editing exercise into a
classroom-like code training process in which models will be trained and
graded on their ability to successfully analyze datasets that they are
fed. Then, the best model can be selected for the job and integrated
into a complete application.

Acumos is part of the LF Deep Learning Foundation, an umbrella
organization within The Linux Foundation that supports and sustains open
source innovation in artificial intelligence, machine learning, and deep
learning while striving to make these critical new technologies
available to developers and data scientists everywhere.

| 

  --------------------- -------------------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Java
  **Project URL**       <https://www.acumos.org/>
  **Source Location**   <https://gerrit.acumos.org/r/#/admin/projects/>
  **Tag(s)**            ML, ML Hosting
  --------------------- -------------------------------------------------

Ray
---

Ray is a flexible, high-performance distributed execution framework for
AI applications. Ray is currently under heavy development. But Ray has
already a good start, with good documentation
(<http://ray.readthedocs.io/en/latest/index.html>) and a tutorial. Also
Ray is backed by scientific researchers and published papers.

Ray comes with libraries that accelerate deep learning and reinforcement
learning development:

-   [Ray Tune](http://ray.readthedocs.io/en/latest/tune.html):
    Hyperparameter Optimization Framework
-   [Ray RLlib](http://ray.readthedocs.io/en/latest/rllib.html): A
    Scalable Reinforcement Learning Library

| 

  --------------------- --------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Python
  **Project URL**       <https://ray-project.github.io/>
  **Source Location**   <https://github.com/ray-project/ray>
  **Tag(s)**            ML, ML Hosting
  --------------------- --------------------------------------

Turi
----

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

NLP Frameworks
==============

Natural language processing (NLP) is a field located at the intersection
of data science and machine learning (ML). It is focussed on teaching
machines how to understand human languages and extract meaning from
text. Using good open FOSS NLP software saves you time and has major
benefits above using closed solutions.

NLP tools make it simple to handle NLP-related tasks such as document
classification, topic modeling, part-of-speech (POS) tagging, word
vectors, and sentiment analysis.

AllenNLP
--------

An open-source NLP research library, built on PyTorch. AllenNLP is a NLP
research library, built on PyTorch, for developing state-of-the-art deep
learning models on a wide variety of linguistic tasks. AllenNLP makes it
easy to design and evaluate new deep learning models for nearly any NLP
problem, along with the infrastructure to easily run them in the cloud
or on your laptop.

AllenNLP was designed with the following principles:

-   *Hyper-modular and lightweight.* Use the parts which you like
    seamlessly with PyTorch.
-   *Extensively tested and easy to extend.* Test coverage is above 90%
    and the example models provide a template for contributions.
-   *Take padding and masking seriously*, making it easy to implement
    correct models without the pain.
-   *Experiment friendly.* Run reproducible experiments from a json
    specification with comprehensive logging.

| 

  --------------------- ---------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Python
  **Project URL**       <http://allennlp.org/>
  **Source Location**   <https://github.com/allenai/allennlp>
  **Tag(s)**            ML, NLP, Python
  --------------------- ---------------------------------------

Apache OpenNLP
--------------

The Apache OpenNLP library is a machine learning based toolkit for the
processing of natural language text.

The Apache OpenNLP library is a machine learning based toolkit for the
processing of natural language text. It supports the most common NLP
tasks, such as tokenization, sentence segmentation, part-of-speech
tagging, named entity extraction, chunking, parsing, and coreference
resolution. These tasks are usually required to build more advanced text
processing services. OpenNLP also included maximum entropy and
perceptron based machine learning.

The goal of the OpenNLP project will be to create a mature toolkit for
the abovementioned tasks. An additional goal is to provide a large
number of pre-built models for a variety of languages, as well as the
annotated text resources that those models are derived from.

| 

  --------------------- ----------------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Java
  **Project URL**       <http://opennlp.apache.org/>
  **Source Location**   <http://opennlp.apache.org/source-code.html>
  **Tag(s)**            NLP
  --------------------- ----------------------------------------------

Apache Tika
-----------

The Apache Tika™ toolkit detects and extracts metadata and text from
over a thousand different file types (such as PPT, XLS, and PDF). All of
these file types can be parsed through a single interface, making Tika
useful for search engine indexing, content analysis, translation, and
much more.

Several wrappers are available to use Tika in another programming
language, such as [Julia](https://github.com/aviks/Taro.jl) or
[Python](https://github.com/chrismattmann/tika-python)

| 

  --------------------- ----------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Java
  **Project URL**       <https://tika.apache.org/>
  **Source Location**   <https://tika.apache.org/>
  **Tag(s)**            NLP
  --------------------- ----------------------------

BERT
----

**BERT**, or **B**idirectional **E**ncoder **R**epresentations from
**T**ransformers, is a new method of pre-training language
representations which obtains state-of-the-art results on a wide array
of Natural Language Processing (NLP) tasks.

Our academic paper which describes BERT in detail and provides full
results on a number of tasks can be found here:
<https://arxiv.org/abs/1810.04805>.

OSS NLP training models from Google Research.

| 

  --------------------- -------------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Python
  **Project URL**       <https://github.com/google-research/bert>
  **Source Location**   <https://github.com/google-research/bert>
  **Tag(s)**            NLP
  --------------------- -------------------------------------------

Bling Fire
----------

A lightning fast Finite State machine and REgular expression
manipulation library. Bling Fire Tokenizer is a tokenizer designed for
fast-speed and quality tokenization of Natural Language text. It mostly
follows the tokenization logic of NLTK, except hyphenated words are
split and a few errors are fixed.

| 

  --------------------- ------------------------------------------
  **SBB License**       MIT License
  **Core Technology**   CPP
  **Project URL**       <https://github.com/Microsoft/BlingFire>
  **Source Location**   <https://github.com/Microsoft/BlingFire>
  **Tag(s)**            NLP
  --------------------- ------------------------------------------

ERNIE
-----

An Implementation of ERNIE For Language Understanding (including
Pre-training models and Fine-tuning tools)

[ERNIE 2.0](https://arxiv.org/abs/1907.12412v1)**is a continual
pre-training framework for language understanding** in which
pre-training tasks can be incrementally built and learned through
multi-task learning. In this framework, different customized tasks can
be incrementally introduced at any time. For example, the tasks
including named entity prediction, discourse relation recognition,
sentence order prediction are leveraged in order to enable the models to
learn language representations.

| 

  --------------------- -----------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Python
  **Project URL**       <https://github.com/PaddlePaddle/ERNIE>
  **Source Location**   <https://github.com/PaddlePaddle/ERNIE>
  **Tag(s)**            NLP, Python
  --------------------- -----------------------------------------

fastText
--------

[fastText](https://fasttext.cc/) is a library for efficient learning of
word representations and sentence classification. Models can later be
reduced in size to even fit on mobile devices.

Created by Facebook Opensource, now available for us all. Also used for
the new search on StackOverflow, see
<https://stackoverflow.blog/2019/08/14/crokage-a-new-way-to-search-stack-overflow/>

| 

  --------------------- ------------------------------------------------
  **SBB License**       MIT License
  **Core Technology**   CPP, Python
  **Project URL**       <https://fasttext.cc/>
  **Source Location**   <https://github.com/facebookresearch/fastText>
  **Tag(s)**            NLP
  --------------------- ------------------------------------------------

Flair
-----

A very simple framework for **state-of-the-art NLP**. Developed by
[Zalando Research](https://research.zalando.com/).

Flair is:

-   **A powerful NLP library.** Flair allows you to apply our
    state-of-the-art natural language processing (NLP) models to your
    text, such as named entity recognition (NER), part-of-speech tagging
    (PoS), sense disambiguation and classification.
-   **Multilingual.** Thanks to the Flair community, we support a
    rapidly growing number of languages. We also now include '*one
    model, many languages*' taggers, i.e. single models that predict PoS
    or NER tags for input text in various languages.
-   **A text embedding library.** Flair has simple interfaces that allow
    you to use and combine different word and document embeddings,
    including our proposed [Flair
    embeddings](https://drive.google.com/file/d/17yVpFA7MmXaQFTe-HDpZuqw9fJlmzg56/view?usp=sharing),
    BERT embeddings and ELMo embeddings.
-   **A Pytorch NLP framework.** Our framework builds directly on
    [Pytorch](https://pytorch.org/), making it easy to train your own
    models and experiment with new approaches using Flair embeddings and
    classes.

| 

  --------------------- --------------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Python
  **Project URL**       <https://github.com/zalandoresearch/flair>
  **Source Location**   <https://github.com/zalandoresearch/flair>
  **Tag(s)**            ML, NLP, Python
  --------------------- --------------------------------------------

Gensim
------

Gensim is a Python library for *topic modelling*, *document indexing*
and *similarity retrieval* with large corpora. Target audience is the
*natural language processing* (NLP) and *information retrieval* (IR)
community.

 

| 

  --------------------- -----------------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Python
  **Project URL**       <https://github.com/RaRe-Technologies/gensim>
  **Source Location**   <https://github.com/RaRe-Technologies/gensim>
  **Tag(s)**            ML, NLP, Python
  --------------------- -----------------------------------------------

Icecaps
-------

Microsoft Icecaps is an open-source toolkit for building neural
conversational systems. Icecaps provides an array of tools from recent
conversation modeling and general NLP literature within a flexible
paradigm that enables complex multi-task learning setups.

Background information can be found here
<https://www.aclweb.org/anthology/P19-3021>

| 

  --------------------- -----------------------------------------------
  **SBB License**       MIT License

  **Core Technology**   Python

  **Project URL**       <https://www.microsoft>.
                        com/en-us/research/project/microsoft-icecaps/

  **Source Location**   <https://github.com/microsoft/icecaps>

  **Tag(s)**            NLP, Python
  --------------------- -----------------------------------------------

jiant
-----

`jiant` is a software toolkit for natural language processing research,
designed to facilitate work on multitask learning and transfer learning
for sentence understanding tasks.

New software for the The General Language Understanding Evaluation
(GLUE) benchmark. This software can be used for evaluating, and
analyzing natural language understanding systems.

See also: <https://super.gluebenchmark.com/>

| 

  --------------------- ------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Python
  **Project URL**       <https://jiant.info/>
  **Source Location**   <https://github.com/nyu-mll/jiant>
  **Tag(s)**            NLP, Python, Research
  --------------------- ------------------------------------

Klassify
--------

Redis based text classification service with real-time web interface.

What is Text Classification: Text classification, document
classification or document categorization is a problem in library
science, information science and computer science. The task is to assign
a document to one or more classes or categories.

| 

  --------------------- -------------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Python
  **Project URL**       <https://github.com/fatiherikli/klassify>
  **Source Location**   <https://github.com/fatiherikli/klassify>
  **Tag(s)**            ML, NLP, Text classification
  --------------------- -------------------------------------------

Neuralcoref
-----------

State-of-the-art coreference resolution based on neural nets and spaCy.

NeuralCoref is a pipeline extension for spaCy 2.0 that annotates and
resolves coreference clusters using a neural network. NeuralCoref is
production-ready, integrated in spaCy's NLP pipeline and easily
extensible to new training datasets.

| 

  --------------------- ----------------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Python
  **Project URL**       <https://huggingface.co/coref/>
  **Source Location**   <https://github.com/huggingface/neuralcoref>
  **Tag(s)**            ML, NLP, Python
  --------------------- ----------------------------------------------

NLP Architect
-------------

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

NLTK (Natural Language Toolkit)
-------------------------------

NLTK is a leading platform for building Python programs to work with
human language data. It provides easy-to-use interfaces to [over 50
corpora and lexical resources](http://nltk.org/nltk_data/) such as
WordNet, along with a suite of text processing libraries for
classification, tokenization, stemming, tagging, parsing, and semantic
reasoning, wrappers for industrial-strength NLP libraries.

Check also the (free) online Book (OReily published)

| 

  --------------------- --------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Python
  **Project URL**       <http://www.nltk.org>
  **Source Location**   <https://github.com/nltk/nltk>
  **Tag(s)**            NLP
  --------------------- --------------------------------

Pattern
-------

Pattern is a web mining module for Python. It has tools for:

-   Data Mining: web services (Google, Twitter, Wikipedia), web crawler,
    HTML DOM parser
-   Natural Language Processing: part-of-speech taggers, n-gram search,
    sentiment analysis, WordNet
-   Machine Learning: vector space model, clustering, classification
    (KNN, SVM, Perceptron)
-   Network Analysis: graph centrality and visualization.

| 

  --------------------- ----------------------------------------------------
  **SBB License**       BSD License 2.0 (3-clause, New or Revised) License
  **Core Technology**   Python
  **Project URL**       <https://www.clips.uantwerpen.be/pages/pattern>
  **Source Location**   <https://github.com/clips/pattern>
  **Tag(s)**            ML, NLP, Web scraping
  --------------------- ----------------------------------------------------

Rant
----

Rant is an all-purpose procedural text engine that is most simply
described as the opposite of Regex. It has been refined to include a
dizzying array of features for handling everything from the most basic
of string generation tasks to advanced dialogue generation, code
templating, automatic formatting, and more.

The goal of the project is to enable developers of all kinds to automate
repetitive writing tasks with a high degree of creative freedom.

Features:

-   Recursive, weighted branching with several selection modes
-   Queryable dictionaries
-   Automatic capitalization, rhyming, English indefinite articles, and
    multi-lingual number verbalization
-   Print to multiple separate outputs
-   Probability modifiers for pattern elements
-   Loops, conditional statements, and subroutines
-   Fully-functional object model
-   Import/Export resources easily with the .rantpkg format
-   Compatible with Unity 2017

| 

  --------------------- -------------------------------------
  **SBB License**       MIT License
  **Core Technology**   .NET
  **Project URL**       <https://berkin.me/rant/>
  **Source Location**   <https://github.com/TheBerkin/rant>
  **Tag(s)**            .NET, ML, NLP, text generation
  --------------------- -------------------------------------

SpaCy
-----

::: {.container .o-grid__col .o-grid__col--third}
Industrial-strength Natural Language Processing (NLP) with Python and
Cython

Features:

-   Non-destructive **tokenization**
-   **Named entity** recognition
-   Support for **26+ languages**
-   **13 statistical models** for 8 languages
-   Pre-trained **word vectors**
-   Easy **deep learning** integration
-   Part-of-speech tagging
-   Labelled dependency parsing
-   Syntax-driven sentence segmentation
-   Built in **visualizers** for syntax and NER
-   Convenient string-to-hash mapping
-   Export to numpy data arrays
-   Efficient binary serialization
-   Easy **model packaging** and deployment
-   State-of-the-art speed
-   Robust, rigorously evaluated accuracy
:::

| 

  --------------------- --------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Python
  **Project URL**       <https://spacy.io/>
  **Source Location**   <https://github.com/explosion/spaCy>
  **Tag(s)**            NLP
  --------------------- --------------------------------------

Stanford CoreNLP
----------------

Stanford CoreNLP provides a set of human language technology tools. It
can give the base forms of words, their parts of speech, whether they
are names of companies, people, etc., normalize dates, times, and
numeric quantities, mark up the structure of sentences in terms of
phrases and syntactic dependencies, indicate which noun phrases refer to
the same entities, indicate sentiment, extract particular or open-class
relations between entity mentions, get the quotes people said, etc.

Choose Stanford CoreNLP if you need:

-   An integrated NLP toolkit with a broad range of grammatical analysis
    tools
-   A fast, robust annotator for arbitrary texts, widely used in
    production
-   A modern, regularly updated package, with the overall highest
    quality text analytics
-   Support for a number of major (human) languages
-   Available APIs for most major modern programming languages
-   Ability to run as a simple web service

| 

  --------------------- ------------------------------------------
  **SBB License**       GNU General Public License (GPL) 3.0
  **Core Technology**   Java
  **Project URL**       <https://stanfordnlp.github.io/CoreNLP/>
  **Source Location**   <https://github.com/stanfordnlp/CoreNLP>
  **Tag(s)**            NLP
  --------------------- ------------------------------------------

Sumeval
-------

Well tested & Multi-language evaluation framework for text
summarization. Multi-language.

| 

  --------------------- -------------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Python
  **Project URL**       <https://github.com/chakki-works/sumeval>
  **Source Location**   <https://github.com/chakki-works/sumeval>
  **Tag(s)**            NLP, Python
  --------------------- -------------------------------------------

Texar-PyTorch
-------------

**Texar-PyTorch** is a toolkit aiming to support a broad set of machine
learning, especially natural language processing and text generation
tasks. Texar provides a library of easy-to-use ML modules and
functionalities for composing whatever models and algorithms. The tool
is designed for both researchers and practitioners for fast prototyping
and experimentation.

Texar-PyTorch integrates many of the best features of TensorFlow into
PyTorch, delivering highly usable and customizable modules superior to
PyTorch native ones.

| 

  --------------------- ------------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Python
  **Project URL**       <https://asyml.io/>
  **Source Location**   <https://github.com/asyml/texar-pytorch>
  **Tag(s)**            ML, NLP, Python
  --------------------- ------------------------------------------

TextBlob: Simplified Text Processing
------------------------------------

*TextBlob* is a Python (2 and 3) library for processing textual data. It
provides a simple API for diving into common natural language processing
(NLP) tasks such as part-of-speech tagging, noun phrase extraction,
sentiment analysis, classification, translation, and more.

Features
--------

-   Noun phrase extraction
-   Part-of-speech tagging
-   Sentiment analysis
-   Classification (Naive Bayes, Decision Tree)
-   Language translation and detection powered by Google Translate
-   Tokenization (splitting text into words and sentences)
-   Word and phrase frequencies
-   Parsing
-   n-grams
-   Word inflection (pluralization and singularization) and
    lemmatization
-   Spelling correction
-   Add new models or languages through extensions
-   WordNet integration

| 

  --------------------- -------------------------------------------
  **SBB License**       MIT License
  **Core Technology**   Python
  **Project URL**       <https://textblob.readthedocs.io/en/dev/>
  **Source Location**   <https://github.com/sloria/textblob>
  **Tag(s)**            NLP, Python
  --------------------- -------------------------------------------

Thinc
-----

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

Torchtext
---------

Data loaders and abstractions for text and NLP. Build on PyTorch.

 

| 

  ------------------------------------- ----------------------------------------------------
  **SBB License** **Core Technology**   BSD License 2.0 (3-clause, New or Revised) License
  **Project URL**                       <https://github.com/pytorch/text>
  **Source Location**                   <https://github.com/pytorch/text>
  **Tag(s)**                            NLP
  ------------------------------------- ----------------------------------------------------

Transformers
------------

Transformers (formerly known as `pytorch-transformers` and
`pytorch-pretrained-bert`) provides state-of-the-art general-purpose
architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet...) for
Natural Language Understanding (NLU) and Natural Language Generation
(NLG) with over 32+ pretrained models in 100+ languages and deep
interoperability between TensorFlow 2.0 and PyTorch.

Features:

-   As easy to use as pytorch-transformers
-   As powerful and concise as Keras
-   High performance on NLU and NLG tasks
-   Low barrier to entry for educators and practitioners

State-of-the-art NLP for everyone:

-   Deep learning researchers
-   Hands-on practitioners
-   AI/ML/NLP teachers and educators

Lower compute costs, smaller carbon footprint

-   Researchers can share trained models instead of always retraining
-   Practitioners can reduce compute time and production costs
-   8 architectures with over 30 pretrained models, some in more than
    100 languages

| 

  --------------------- -----------------------------------------------
  **SBB License**       Apache License 2.0
  **Core Technology**   Python
  **Project URL**       <https://huggingface.co/transformers/>
  **Source Location**   <https://github.com/huggingface/transformers>
  **Tag(s)**            NLP, Python
  --------------------- -----------------------------------------------
