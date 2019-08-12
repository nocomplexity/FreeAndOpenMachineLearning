OSS System Building Blocks: Category ML
=======================================

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

+-----------------------+-------------------------------------------------+
| **SBB License**       | Apache License 2.0                              |
+-----------------------+-------------------------------------------------+
| **Core Technology**   | Java                                            |
+-----------------------+-------------------------------------------------+
| **Project URL**       | https://www.acumos.org/                         |
+-----------------------+-------------------------------------------------+
| **Source Location**   | https://gerrit.acumos.org/r/#/admin/projects/   |
+-----------------------+-------------------------------------------------+
| **Tag(s)**            | ML                                              |
+-----------------------+-------------------------------------------------+

| 

AdaNet
------

AdaNet is a lightweight TensorFlow-based framework for automatically
learning high-quality models with minimal expert intervention. AdaNet
builds on recent AutoML efforts to be fast and flexible while providing
learning guarantees. Importantly, AdaNet provides a general framework
for not only learning a neural network architecture, but also for
learning to ensemble to obtain even better models.

This project is based on the *AdaNet algorithm*, presented in
“\ `AdaNet: Adaptive Structural Learning of Artificial Neural
Networks <http://proceedings.mlr.press/v70/cortes17a.html>`__\ ” at
`ICML 2017 <https://icml.cc/Conferences/2017>`__, for learning the
structure of a neural network as an ensemble of subnetworks.

AdaNet has the following goals:

-  *Ease of use*: Provide familiar APIs (e.g. Keras, Estimator) for
   training, evaluating, and serving models.
-  *Speed*: Scale with available compute and quickly produce high
   quality models.
-  *Flexibility*: Allow researchers and practitioners to extend AdaNet
   to novel subnetwork architectures, search spaces, and tasks.
-  *Learning guarantees*: Optimize an objective that offers theoretical
   learning guarantees.

Documentation at https://adanet.readthedocs.io/en/latest/

+-----------------------+--------------------------------------------+
| **SBB License**       | Apache License 2.0                         |
+-----------------------+--------------------------------------------+
| **Core Technology**   | Python                                     |
+-----------------------+--------------------------------------------+
| **Project URL**       | https://adanet.readthedocs.io/en/latest/   |
+-----------------------+--------------------------------------------+
| **Source Location**   | https://github.com/tensorflow/adanet       |
+-----------------------+--------------------------------------------+
| **Tag(s)**            | ML                                         |
+-----------------------+--------------------------------------------+

| 

AI Explainability 360
---------------------

The AI Explainability 360 toolkit is an open-source library that
supports interpretability and explainability of datasets and machine
learning models. The AI Explainability 360 Python package includes a
comprehensive set of algorithms that cover different dimensions of
explanations along with proxy explainability metrics.

It is OSS from IBM (so apache2.0) so mind the history of openness IBM
has regarding OSS product development. The documentation can be found
here: https://aix360.readthedocs.io/en/latest/

+-----------------------+---------------------------------+
| **SBB License**       | Apache License 2.0              |
+-----------------------+---------------------------------+
| **Core Technology**   | Python                          |
+-----------------------+---------------------------------+
| **Project URL**       | http://aix360.mybluemix.net/    |
+-----------------------+---------------------------------+
| **Source Location**   | https://github.com/IBM/AIX360   |
+-----------------------+---------------------------------+
| **Tag(s)**            | Data analytics, ML, Python      |
+-----------------------+---------------------------------+

| 

AllenNLP
--------

An open-source NLP research library, built on PyTorch. AllenNLP is a NLP
research library, built on PyTorch, for developing state-of-the-art deep
learning models on a wide variety of linguistic tasks. AllenNLP makes it
easy to design and evaluate new deep learning models for nearly any NLP
problem, along with the infrastructure to easily run them in the cloud
or on your laptop.

AllenNLP was designed with the following principles:

-  *Hyper-modular and lightweight.* Use the parts which you like
   seamlessly with PyTorch.
-  *Extensively tested and easy to extend.* Test coverage is above 90%
   and the example models provide a template for contributions.
-  *Take padding and masking seriously*, making it easy to implement
   correct models without the pain.
-  *Experiment friendly.* Run reproducible experiments from a json
   specification with comprehensive logging.

+-----------------------+---------------------------------------+
| **SBB License**       | Apache License 2.0                    |
+-----------------------+---------------------------------------+
| **Core Technology**   | Python                                |
+-----------------------+---------------------------------------+
| **Project URL**       | http://allennlp.org/                  |
+-----------------------+---------------------------------------+
| **Source Location**   | https://github.com/allenai/allennlp   |
+-----------------------+---------------------------------------+
| **Tag(s)**            | ML, NLP, Python                       |
+-----------------------+---------------------------------------+

| 

Analytics Zoo
-------------

**Analytics Zoo** provides a unified analytics + AI platform that
seamlessly unites ***Spark, TensorFlow, Keras and BigDL*** programs into
an integrated pipeline; the entire pipeline can then transparently scale
out to a large Hadoop/Spark cluster for distributed training or
inference.

-  *Data wrangling and analysis using PySpark*
-  *Deep learning model development using TensorFlow or Keras*
-  *Distributed training/inference on Spark and BigDL*
-  *All within a single unified pipeline and in a user-transparent
   fashion!*

+-----------------------+----------------------------------------------------+
| **SBB License**       | Apache License 2.0                                 |
+-----------------------+----------------------------------------------------+
| **Core Technology**   | Python                                             |
+-----------------------+----------------------------------------------------+
| **Project URL**       | https://analytics-zoo.github.io/master/            |
+-----------------------+----------------------------------------------------+
| **Source Location**   | https://github.com/intel-analytics/analytics-zoo   |
+-----------------------+----------------------------------------------------+
| **Tag(s)**            | ML, Python                                         |
+-----------------------+----------------------------------------------------+

| 

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
*efficiency* and *flexibility*. It allows you to ***mix*** `symbolic and
imperative
programming <https://mxnet.incubator.apache.org/architecture/index.html#deep-learning-system-design-concepts>`__
to ***maximize*** efficiency and productivity. At its core, MXNet
contains a dynamic dependency scheduler that automatically parallelizes
both symbolic and imperative operations on the fly. A graph optimization
layer on top of that makes symbolic execution fast and memory efficient.
MXNet is portable and lightweight, scaling effectively to multiple GPUs
and multiple machines.

MXNet is also more than a deep learning project. It is also a collection
of `blue prints and
guidelines <https://mxnet.incubator.apache.org/architecture/index.html#deep-learning-system-design-concepts>`__
for building deep learning systems, and interesting insights of DL
systems for hackers.

Gluon is the high-level interface for MXNet. It is more intuitive and
easier to use than the lower level interface. Gluon supports dynamic
(define-by-run) graphs with JIT-compilation to achieve both flexibility
and efficiency. The perfect starters documentation with a great crash
course on deep learning can be found here:\ http://gluon.mxnet.io/

Part of the project is also the the Gluon API specification (see
https://github.com/gluon-api/gluon-api)

The Gluon API specification (Python based) is an effort to improve
speed, flexibility, and accessibility of deep learning technology for
all developers, regardless of their deep learning framework of choice.
The Gluon API offers a flexible interface that simplifies the process of
prototyping, building, and training deep learning models without
sacrificing training speed.

+-----------------------+---------------------------------------------+
| **SBB License**       | Apache License 2.0                          |
+-----------------------+---------------------------------------------+
| **Core Technology**   | CPP                                         |
+-----------------------+---------------------------------------------+
| **Project URL**       | https://mxnet.apache.org/                   |
+-----------------------+---------------------------------------------+
| **Source Location**   | https://github.com/apache/incubator-mxnet   |
+-----------------------+---------------------------------------------+
| **Tag(s)**            | ML                                          |
+-----------------------+---------------------------------------------+

| 

Apache Spark MLlib
------------------

Apache Spark MLlib. MLlib is Apache Spark’s scalable machine learning
library.

Apache Spark is a OSS platform for large-scale data processing. The
Spark engine is written in Scala and is well suited for applications
that reuse a working set of data across multiple parallel operations.
It’s designed to work as a standalone cluster or as part of Hadoop YARN
cluster. It can access data from sources such as HDFS, Cassandra or
Amazon S3. MLlib can be seen as a core Spark’s APIs and interoperates
with NumPy in Python and R libraries. And Spark is very fast!

MLlib library contains many algorithms and utilities, e.g.:

-  Classification: logistic regression, naive Bayes,…
-  Regression: generalized linear regression, survival regression,…
-  Decision trees, random forests, and gradient-boosted trees
-  Recommendation: alternating least squares (ALS)
-  Clustering: K-means, Gaussian mixtures (GMMs),…
-  Topic modeling: latent Dirichlet allocation (LDA)
-  Frequent itemsets, association rules, and sequential pattern mining

+-----------------------+-----------------------------------+
| **SBB License**       | Apache License 2.0                |
+-----------------------+-----------------------------------+
| **Core Technology**   | Java                              |
+-----------------------+-----------------------------------+
| **Project URL**       | https://spark.apache.org/mllib/   |
+-----------------------+-----------------------------------+
| **Source Location**   | https://github.com/apache/spark   |
+-----------------------+-----------------------------------+
| **Tag(s)**            | ML                                |
+-----------------------+-----------------------------------+

| 

Apollo
------

Apollo is a high performance, flexible architecture which accelerates
the development, testing, and deployment of Autonomous Vehicles.

+-----------------------+----------------------------------------+
| **SBB License**       | GNU General Public License (GPL) 2.0   |
+-----------------------+----------------------------------------+
| **Core Technology**   | C++                                    |
+-----------------------+----------------------------------------+
| **Project URL**       | http://apollo.auto/                    |
+-----------------------+----------------------------------------+
| **Source Location**   | https://github.com/ApolloAuto/apollo   |
+-----------------------+----------------------------------------+
| **Tag(s)**            | ML                                     |
+-----------------------+----------------------------------------+

| 

auto\_ml
--------

Automated machine learning for analytics & production.

Automates the whole machine learning process, making it super easy to
use for both analytics, and getting real-time predictions in production.

+-----------------------+------------------------------------------+
| **SBB License**       | MIT License                              |
+-----------------------+------------------------------------------+
| **Core Technology**   | Python                                   |
+-----------------------+------------------------------------------+
| **Project URL**       | http://auto-ml.readthedocs.io            |
+-----------------------+------------------------------------------+
| **Source Location**   | https://github.com/ClimbsRocks/auto_ml   |
+-----------------------+------------------------------------------+
| **Tag(s)**            | ML                                       |
+-----------------------+------------------------------------------+

| 

BigDL
-----

BigDL is a distributed deep learning library for Apache Spark; with
BigDL, users can write their deep learning applications as standard
Spark programs, which can directly run on top of existing Spark or
Hadoop clusters.

-  **Rich deep learning support.** Modeled after
   `Torch <http://torch.ch/>`__, BigDL provides comprehensive support
   for deep learning, including numeric computing (via
   `Tensor <https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/tensor>`__)
   and high level `neural
   networks <https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/nn>`__;
   in addition, users can load pre-trained
   `Caffe <http://caffe.berkeleyvision.org/>`__ or
   `Torch <http://torch.ch/>`__ or
   `Keras <https://faroit.github.io/keras-docs/1.2.2/>`__ models into
   Spark programs using BigDL.
-  **Extremely high performance.** To achieve high performance, BigDL
   uses `Intel MKL <https://software.intel.com/en-us/intel-mkl>`__ and
   multi-threaded programming in each Spark task. Consequently, it is
   orders of magnitude faster than out-of-box open source
   `Caffe <http://caffe.berkeleyvision.org/>`__,
   `Torch <http://torch.ch/>`__ or
   `TensorFlow <https://www.tensorflow.org/>`__ on a single-node Xeon
   (i.e., comparable with mainstream GPU).
-  **Efficiently scale-out.** BigDL can efficiently scale out to perform
   data analytics at “Big Data scale”, by leveraging `Apache
   Spark <http://spark.apache.org/>`__ (a lightning fast distributed
   data processing framework), as well as efficient implementations of
   synchronous SGD and all-reduce communications on Spark.

+-----------------------+--------------------------------------------+
| **SBB License**       | Apache License 2.0                         |
+-----------------------+--------------------------------------------+
| **Core Technology**   | Java                                       |
+-----------------------+--------------------------------------------+
| **Project URL**       | https://bigdl-project.github.io/master/    |
+-----------------------+--------------------------------------------+
| **Source Location**   | https://github.com/intel-analytics/BigDL   |
+-----------------------+--------------------------------------------+
| **Tag(s)**            | ML                                         |
+-----------------------+--------------------------------------------+

| 

Blocks
------

Blocks is a framework that is supposed to make it easier to build
complicated neural network models on top of
`Theano <http://www.deeplearning.net/software/theano/>`__.

Blocks is a framework that helps you build neural network models on top
of Theano. Currently it supports and provides:

-  Constructing parametrized Theano operations, called “bricks”
-  Pattern matching to select variables and bricks in large models
-  Algorithms to optimize your model
-  Saving and resuming of training
-  Monitoring and analyzing values during training progress (on the
   training set as well as on test sets)
-  Application of graph transformations, such as dropout

+-----------------------+-------------------------------------------+
| **SBB License**       | MIT License                               |
+-----------------------+-------------------------------------------+
| **Core Technology**   | Python                                    |
+-----------------------+-------------------------------------------+
| **Project URL**       | http://blocks.readthedocs.io/en/latest/   |
+-----------------------+-------------------------------------------+
| **Source Location**   | https://github.com/mila-udem/blocks       |
+-----------------------+-------------------------------------------+
| **Tag(s)**            | ML                                        |
+-----------------------+-------------------------------------------+

| 

ConvNetJS
---------

ConvNetJS is a Javascript library for training Deep Learning models
(Neural Networks) entirely in your browser. Open a tab and you’re
training. No software requirements, no compilers, no installations, no
GPUs, no sweat.

ConvNetJS is a Javascript implementation of Neural networks, together
with nice browser-based demos. It currently supports:

-  Common **Neural Network modules** (fully connected layers,
   non-linearities)
-  Classification (SVM/Softmax) and Regression (L2) **cost functions**
-  Ability to specify and train **Convolutional Networks** that process
   images
-  An experimental **Reinforcement Learning** module, based on Deep Q
   Learning

For much more information, see the main page at
`convnetjs.com <http://convnetjs.com>`__

Note: Not actively maintained, but still useful to prevent reinventing
the wheel.

 

+-----------------------+------------------------------------------------------+
| **SBB License**       | MIT License                                          |
+-----------------------+------------------------------------------------------+
| **Core Technology**   | Javascript                                           |
+-----------------------+------------------------------------------------------+
| **Project URL**       | https://cs.stanford.edu/people/karpathy/convnetjs/   |
+-----------------------+------------------------------------------------------+
| **Source Location**   | https://github.com/karpathy/convnetjs                |
+-----------------------+------------------------------------------------------+
| **Tag(s)**            | Javascript, ML                                       |
+-----------------------+------------------------------------------------------+

| 

Cookiecutter Data Science
-------------------------

A logical, reasonably standardized, but flexible project structure for
doing and sharing data science work.

 

+-----------------------+-----------------------------------------------------------+
| **SBB License**       | MIT License                                               |
+-----------------------+-----------------------------------------------------------+
| **Core Technology**   | Python                                                    |
+-----------------------+-----------------------------------------------------------+
| **Project URL**       | https://drivendata.github.io/cookiecutter-data-science/   |
+-----------------------+-----------------------------------------------------------+
| **Source Location**   | https://github.com/drivendata/cookiecutter-data-science   |
+-----------------------+-----------------------------------------------------------+
| **Tag(s)**            | Data tool, ML                                             |
+-----------------------+-----------------------------------------------------------+

| 

Data Science Version Control (DVC)
----------------------------------

**Data Science Version Control** or **DVC** is an **open-source** tool
for data science and machine learning projects. With a simple and
flexible Git-like architecture and interface it helps data scientists:

#. manage **machine learning models** – versioning, including data sets
   and transformations (scripts) that were used to generate models;
#. make projects **reproducible**;
#. make projects **shareable**;
#. manage experiments with branching and **metrics** tracking;

It aims to replace tools like Excel and Docs that are being commonly
used as a knowledge repo and a ledger for the team, ad-hoc scripts to
track and move deploy different model versions, ad-hoc data file
suffixes and prefixes.

+-----------------------+------------------------------------+
| **SBB License**       | Apache License 2.0                 |
+-----------------------+------------------------------------+
| **Core Technology**   | Python                             |
+-----------------------+------------------------------------+
| **Project URL**       | https://dvc.org/                   |
+-----------------------+------------------------------------+
| **Source Location**   | https://github.com/iterative/dvc   |
+-----------------------+------------------------------------+
| **Tag(s)**            | ML, Python                         |
+-----------------------+------------------------------------+

| 

Dataexplorer
------------

View, visualize, clean and process data in the browser.

Some features:

-  Classic spreadsheet-style “grid” view
-  Import CSV data from online
-  Geocode data (convert “London” to longitude and latitude)
-  Data and scripts automatically saved and accessible from anywhere
-  “Fork” support – build on others work and let them build on yours

+-----------------------+----------------------------------------+
| **SBB License**       | MIT License                            |
+-----------------------+----------------------------------------+
| **Core Technology**   | javascript                             |
+-----------------------+----------------------------------------+
| **Project URL**       | http://explorer.okfnlabs.org           |
+-----------------------+----------------------------------------+
| **Source Location**   | https://github.com/okfn/dataexplorer   |
+-----------------------+----------------------------------------+
| **Tag(s)**            | Data viewer, ML                        |
+-----------------------+----------------------------------------+

| 

Datastream
----------

An open-source framework for real-time anomaly detection using Python,
ElasticSearch and Kiban. Also uses scikit-learn.

+-----------------------+------------------------------------------------------+
| **SBB License**       | Apache License 2.0                                   |
+-----------------------+------------------------------------------------------+
| **Core Technology**   | Python                                               |
+-----------------------+------------------------------------------------------+
| **Project URL**       | https://github.com/MentatInnovations/datastream.io   |
+-----------------------+------------------------------------------------------+
| **Source Location**   | https://github.com/MentatInnovations/datastream.io   |
+-----------------------+------------------------------------------------------+
| **Tag(s)**            | ML, Monitoring, Security                             |
+-----------------------+------------------------------------------------------+

| 

DeepDetect
----------

DeepDetect implements support for supervised and unsupervised deep
learning of images, text and other data, with focus on simplicity and
ease of use, test and connection into existing applications. It supports
classification, object detection, segmentation, regression, autoencoders
and more.

It has Python and other client libraries.

Deep Detect has also a REST API for Deep Learning with:

-  JSON communication format
-  Pre-trained models
-  Neural architecture templates
-  Python, Java, C# clients
-  Output templating

 

+-----------------------+---------------------------------------+
| **SBB License**       | MIT License                           |
+-----------------------+---------------------------------------+
| **Core Technology**   | C++                                   |
+-----------------------+---------------------------------------+
| **Project URL**       | https://deepdetect.com                |
+-----------------------+---------------------------------------+
| **Source Location**   | https://github.com/beniz/deepdetect   |
+-----------------------+---------------------------------------+
| **Tag(s)**            | ML                                    |
+-----------------------+---------------------------------------+

| 

Deeplearn.js
------------

Deeplearn.js is an open-source library that brings performant machine
learning building blocks to the web, allowing you to train neural
networks in a browser or run pre-trained models in inference mode. And
since Google is behind this project, a lot of eyes are targeted on this
software. Deeplearn.js is an open source hardware accelerated
implementation of deep learning APIs in the browser. So there is no need
to download or install anything.

Deeplearn.js was originally developed by the Google Brain PAIR team to
build powerful interactive machine learning tools for the browser.

+-----------------------+--------------------------------------------+
| **SBB License**       | Apache License 2.0                         |
+-----------------------+--------------------------------------------+
| **Core Technology**   | Javascript                                 |
+-----------------------+--------------------------------------------+
| **Project URL**       | https://deeplearnjs.org/                   |
+-----------------------+--------------------------------------------+
| **Source Location**   | https://github.com/PAIR-code/deeplearnjs   |
+-----------------------+--------------------------------------------+
| **Tag(s)**            | Javascript, ML                             |
+-----------------------+--------------------------------------------+

| 

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
machine learning models to be hosted with Skymind’s model server on a
cloud environment

+-----------------------+----------------------------------------------------+
| **SBB License**       | Apache License 2.0                                 |
+-----------------------+----------------------------------------------------+
| **Core Technology**   | Java                                               |
+-----------------------+----------------------------------------------------+
| **Project URL**       | https://deeplearning4j.org                         |
+-----------------------+----------------------------------------------------+
| **Source Location**   | https://github.com/deeplearning4j/deeplearning4j   |
+-----------------------+----------------------------------------------------+
| **Tag(s)**            | ML                                                 |
+-----------------------+----------------------------------------------------+

| 

Detectron
---------

Detectron is Facebook AI Research’s software system that implements
state-of-the-art object detection algorithms, including `Mask
R-CNN <https://arxiv.org/abs/1703.06870>`__. It is written in Python and
powered by the `Caffe2 <https://github.com/caffe2/caffe2>`__ deep
learning framework.

The goal of Detectron is to provide a high-quality, high-performance
codebase for object detection *research*. It is designed to be flexible
in order to support rapid implementation and evaluation of novel
research.

A number of Facebook teams use this platform to train custom models for
a variety of applications including augmented reality and community
integrity. Once trained, these models can be deployed in the cloud and
on mobile devices, powered by the highly efficient Caffe2 runtime.

+-----------------------+-------------------------------------------------+
| **SBB License**       | Apache License 2.0                              |
+-----------------------+-------------------------------------------------+
| **Core Technology**   | Python                                          |
+-----------------------+-------------------------------------------------+
| **Project URL**       | https://github.com/facebookresearch/Detectron   |
+-----------------------+-------------------------------------------------+
| **Source Location**   | https://github.com/facebookresearch/Detectron   |
+-----------------------+-------------------------------------------------+
| **Tag(s)**            | AI, ML, Python                                  |
+-----------------------+-------------------------------------------------+

| 

Dopamine
--------

Dopamine is a research framework for fast prototyping of reinforcement
learning algorithms. It aims to fill the need for a small, easily
grokked codebase in which users can freely experiment with wild ideas
(speculative research).

Our design principles are:

-  *Easy experimentation*: Make it easy for new users to run benchmark
   experiments.
-  *Flexible development*: Make it easy for new users to try out
   research ideas.
-  *Compact and reliable*: Provide implementations for a few,
   battle-tested algorithms.
-  *Reproducible*: Facilitate reproducibility in results.

+-----------------------+--------------------------------------+
| **SBB License**       | Apache License 2.0                   |
+-----------------------+--------------------------------------+
| **Core Technology**   | Python                               |
+-----------------------+--------------------------------------+
| **Project URL**       | https://github.com/google/dopamine   |
+-----------------------+--------------------------------------+
| **Source Location**   | https://github.com/google/dopamine   |
+-----------------------+--------------------------------------+
| **Tag(s)**            | ML, Reinforcement Learning           |
+-----------------------+--------------------------------------+

| 

Fabrik
------

Fabrik is an online collaborative platform to build, visualize and train
deep learning models via a simple drag-and-drop interface. It allows
researchers to collaboratively develop and debug models using a web GUI
that supports importing, editing and exporting networks written in
widely popular frameworks like Caffe, Keras, and TensorFlow.

+-----------------------+----------------------------------------+
| **SBB License**       | GNU General Public License (GPL) 3.0   |
+-----------------------+----------------------------------------+
| **Core Technology**   | Javascript, Python                     |
+-----------------------+----------------------------------------+
| **Project URL**       | http://fabrik.cloudcv.org/             |
+-----------------------+----------------------------------------+
| **Source Location**   | https://github.com/Cloud-CV/Fabrik     |
+-----------------------+----------------------------------------+
| **Tag(s)**            | Data Visualization, ML                 |
+-----------------------+----------------------------------------+

| 

Fastai
------

The fastai library simplifies training fast and accurate neural nets
using modern best practices. Fast.ai’s mission is to make the power of
state of the art deep learning available to anyone. fastai sits on top
of `PyTorch <https://pytorch.org/>`__, which provides the foundation.

Docs can be found on:\ http://docs.fast.ai/

+-----------------------+-------------------------------------+
| **SBB License**       | Apache License 2.0                  |
+-----------------------+-------------------------------------+
| **Core Technology**   | Python                              |
+-----------------------+-------------------------------------+
| **Project URL**       | http://www.fast.ai/                 |
+-----------------------+-------------------------------------+
| **Source Location**   | https://github.com/fastai/fastai/   |
+-----------------------+-------------------------------------+
| **Tag(s)**            | ML                                  |
+-----------------------+-------------------------------------+

| 

Featuretools
------------

Featuretools is a python library for automated feature engineering.
Featuretools can automatically create a single table of features for any
“target entity”. Featuretools is a framework to perform automated
feature engineering. It excels at transforming transactional and
relational datasets into feature matrices for machine learning.

+-----------------------+------------------------------------------------------+
| **SBB License**       | BSD License 2.0 (3-clause, New or Revised) License   |
+-----------------------+------------------------------------------------------+
| **Core Technology**   | Python                                               |
+-----------------------+------------------------------------------------------+
| **Project URL**       | https://www.featuretools.com/                        |
+-----------------------+------------------------------------------------------+
| **Source Location**   | https://github.com/Featuretools/featuretools         |
+-----------------------+------------------------------------------------------+
| **Tag(s)**            | ML, Python                                           |
+-----------------------+------------------------------------------------------+

| 

Featuretools
------------

*“One of the holy grails of machine learning is to automate more and
more of the feature engineering process.”* ― Pedro

| `Featuretools <https://www.featuretools.com>`__ is a python library
  for automated feature engineering. Featuretools automatically creates
  features from
| temporal and relational datasets. Featuretools works alongside tools
  you already use to build machine learning pipelines. You can load in
  pandas dataframes and automatically create meaningful features in a
  fraction of the time it would take to do manually.

 

+-----------------------+------------------------------------------------------+
| **SBB License**       | BSD License 2.0 (3-clause, New or Revised) License   |
+-----------------------+------------------------------------------------------+
| **Core Technology**   | Python                                               |
+-----------------------+------------------------------------------------------+
| **Project URL**       | https://www.featuretools.com/                        |
+-----------------------+------------------------------------------------------+
| **Source Location**   | https://github.com/Featuretools/featuretools         |
+-----------------------+------------------------------------------------------+
| **Tag(s)**            | ML                                                   |
+-----------------------+------------------------------------------------------+

| 

Flair
-----

A very simple framework for **state-of-the-art NLP**. Developed by
`Zalando Research <https://research.zalando.com/>`__.

Flair is:

-  **A powerful NLP library.** Flair allows you to apply our
   state-of-the-art natural language processing (NLP) models to your
   text, such as named entity recognition (NER), part-of-speech tagging
   (PoS), sense disambiguation and classification.
-  **Multilingual.** Thanks to the Flair community, we support a rapidly
   growing number of languages. We also now include ‘\ *one model, many
   languages*\ ‘ taggers, i.e. single models that predict PoS or NER
   tags for input text in various languages.
-  **A text embedding library.** Flair has simple interfaces that allow
   you to use and combine different word and document embeddings,
   including our proposed **`Flair
   embeddings <https://drive.google.com/file/d/17yVpFA7MmXaQFTe-HDpZuqw9fJlmzg56/view?usp=sharing>`__**,
   BERT embeddings and ELMo embeddings.
-  **A Pytorch NLP framework.** Our framework builds directly on
   `Pytorch <https://pytorch.org/>`__, making it easy to train your own
   models and experiment with new approaches using Flair embeddings and
   classes.

+-----------------------+--------------------------------------------+
| **SBB License**       | MIT License                                |
+-----------------------+--------------------------------------------+
| **Core Technology**   | Python                                     |
+-----------------------+--------------------------------------------+
| **Project URL**       | https://github.com/zalandoresearch/flair   |
+-----------------------+--------------------------------------------+
| **Source Location**   | https://github.com/zalandoresearch/flair   |
+-----------------------+--------------------------------------------+
| **Tag(s)**            | ML, NLP, Python                            |
+-----------------------+--------------------------------------------+

| 

Fuel
----

Fuel is a data pipeline framework which provides your machine learning
models with the data they need. It is planned to be used by both the
`Blocks <https://github.com/mila-udem/blocks>`__ and
`Pylearn2 <https://github.com/lisa-lab/pylearn2>`__ neural network
libraries.

-  Fuel allows you to easily read different types of data (NumPy binary
   files, CSV files, HDF5 files, text files) using a single interface
   which is based on Python’s iterator types.
-  Provides a a series of wrappers around frequently used datasets such
   as MNIST, CIFAR-10 (vision), the One Billion Word Dataset (text
   corpus), and many more.
-  Allows you iterate over data in a variety of ways, e.g. in order,
   shuffled, sampled, etc.
-  Gives you the possibility to process your data on-the-fly through a
   series of (chained) transformation procedures. This way you can
   whiten your data, noise, rotate, crop, pad, sort or shuffle, cache
   it, and much more.
-  Is pickle-friendly, allowing you to stop and resume long-running
   experiments in the middle of a pass over your dataset without losing
   any training progress.

+-----------------------+---------------------------------------------------+
| **SBB License**       | MIT License                                       |
+-----------------------+---------------------------------------------------+
| **Core Technology**   | Python                                            |
+-----------------------+---------------------------------------------------+
| **Project URL**       | http://fuel.readthedocs.io/en/latest/index.html   |
+-----------------------+---------------------------------------------------+
| **Source Location**   | https://github.com/mila-udem/fuel                 |
+-----------------------+---------------------------------------------------+
| **Tag(s)**            | Data tool, ML                                     |
+-----------------------+---------------------------------------------------+

| 

Gensim
------

Gensim is a Python library for *topic modelling*, *document indexing*
and *similarity retrieval* with large corpora. Target audience is the
*natural language processing* (NLP) and *information retrieval* (IR)
community.

 

+-----------------------+-----------------------------------------------+
| **SBB License**       | MIT License                                   |
+-----------------------+-----------------------------------------------+
| **Core Technology**   | Python                                        |
+-----------------------+-----------------------------------------------+
| **Project URL**       | https://github.com/RaRe-Technologies/gensim   |
+-----------------------+-----------------------------------------------+
| **Source Location**   | https://github.com/RaRe-Technologies/gensim   |
+-----------------------+-----------------------------------------------+
| **Tag(s)**            | ML, NLP, Python                               |
+-----------------------+-----------------------------------------------+

| 

Golem
-----

The aim of the Golem project is to create a global prosumer market for
computing power, in which producers may sell spare CPU time of their
personal computers and consumers may acquire resources for
computation-intensive tasks. In technical terms, Golem is designed as a
decentralised peer-to-peer network established by nodes running the
Golem client software. For the purpose of this paper we assume that
there are two types of nodes in the Golem network: requestor nodes that
announce computing tasks and compute nodes that perform computations (in
the actual implementation nodes may switch between both roles).

+-----------------------+-----------------------------------------+
| **SBB License**       | GNU General Public License (GPL) 3.0    |
+-----------------------+-----------------------------------------+
| **Core Technology**   | Python                                  |
+-----------------------+-----------------------------------------+
| **Project URL**       | https://golem.network/                  |
+-----------------------+-----------------------------------------+
| **Source Location**   | https://github.com/golemfactory/golem   |
+-----------------------+-----------------------------------------+
| **Tag(s)**            | Distributed Computing, ML               |
+-----------------------+-----------------------------------------+

| 

HyperTools
----------

`HyperTools <https://github.com/ContextLab/hypertools>`__ is a library
for visualizing and manipulating high-dimensional data in Python. It is
built on top of matplotlib (for plotting), seaborn (for plot styling),
and scikit-learn (for data manipulation).

Some key features of HyperTools are:

#. Functions for plotting high-dimensional datasets in 2/3D
#. Static and animated plots
#. Simple API for customizing plot styles
#. Set of powerful data manipulation tools including hyperalignment,
   k-means clustering, normalizing and more
#. Support for lists of Numpy arrays or Pandas dataframes

+-----------------------+-----------------------------------------------+
| **SBB License**       | MIT License                                   |
+-----------------------+-----------------------------------------------+
| **Core Technology**   | Python                                        |
+-----------------------+-----------------------------------------------+
| **Project URL**       | http://hypertools.readthedocs.io/en/latest/   |
+-----------------------+-----------------------------------------------+
| **Source Location**   | https://github.com/ContextLab/hypertools      |
+-----------------------+-----------------------------------------------+
| **Tag(s)**            | Data tool, ML                                 |
+-----------------------+-----------------------------------------------+

| 

JeelizFaceFilter
----------------

Javascript/WebGL lightweight face tracking library designed for
augmented reality webcam filters. Features : multiple faces detection,
rotation, mouth opening. Various integration examples are provided
(Three.js, Babylon.js, FaceSwap, Canvas2D, CSS3D…).

Enables developers to solve computer-vision problems directly from the
browser.

Features:

-  face detection,
-  face tracking,
-  face rotation detection,
-  mouth opening detection,
-  multiple faces detection and tracking,
-  very robust for all lighting conditions,
-  video acquisition with HD video ability,
-  interfaced with 3D engines like THREE.JS, BABYLON.JS, A-FRAME,
-  interfaced with more accessible APIs like CANVAS, CSS3D.

+-----------------------+----------------------------------------------+
| **SBB License**       | Apache License 2.0                           |
+-----------------------+----------------------------------------------+
| **Core Technology**   | Javascript                                   |
+-----------------------+----------------------------------------------+
| **Project URL**       | https://jeeliz.com/                          |
+-----------------------+----------------------------------------------+
| **Source Location**   | https://github.com/jeeliz/jeelizFaceFilter   |
+-----------------------+----------------------------------------------+
| **Tag(s)**            | face detection, Javascript, ML               |
+-----------------------+----------------------------------------------+

| 

Kedro
-----

Kedro is a workflow development tool that helps you build data pipelines
that are robust, scalable, deployable, reproducible and versioned. We
provide a standard approach so that you can:

-  spend more time building your data pipeline,
-  worry less about how to write production-ready code,
-  standardise the way that your team collaborates across your project,
-  work more efficiently.

Main features of
Kedro:\ ` <https://github.com/quantumblacklabs/kedro#1-project-template-and-coding-standards>`__

1. Project template and coding standards

-  A standard and easy-to-use project template
-  Configuration for credentials, logging, data loading and Jupyter
   Notebooks / Lab
-  Test-driven development using ``pytest``
-  `Sphinx <http://www.sphinx-doc.org/en/master/>`__ integration to
   produce well-documented code

` <https://github.com/quantumblacklabs/kedro#2-data-abstraction-and-versioning>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2. Data abstraction and versioning

-  Separation of the *compute* layer from the *data handling* layer,
   including support for different data formats and storage options
-  Versioning for your data sets and machine learning models

` <https://github.com/quantumblacklabs/kedro#3-modularity-and-pipeline-abstraction>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

3. Modularity and pipeline abstraction

-  Support for pure Python functions, ``nodes``, to break large chunks
   of code into small independent sections
-  Automatic resolution of dependencies between ``nodes``
-  Visualise your data pipeline with
   `Kedro-Viz <https://github.com/quantumblacklabs/kedro-viz>`__, a tool
   that shows the pipeline structure of Kedro projects

Documentation on: ****
`**https://kedro.readthedocs.io/** <https://kedro.readthedocs.io/>`__

The REACT visualization for Kedro is
on:\ `https://github.com/quantumblacklabs/kedro-viz <http://%20https://github.com/quantumblacklabs/kedro-viz%20>`__

+-----------------------+---------------------------------------------+
| **SBB License**       | Apache License 2.0                          |
+-----------------------+---------------------------------------------+
| **Core Technology**   | Python                                      |
+-----------------------+---------------------------------------------+
| **Project URL**       | https://github.com/quantumblacklabs/kedro   |
+-----------------------+---------------------------------------------+
| **Source Location**   | https://github.com/quantumblacklabs/kedro   |
+-----------------------+---------------------------------------------+
| **Tag(s)**            | ML, Python                                  |
+-----------------------+---------------------------------------------+

| 

Keras
-----

Keras is a high-level neural networks API, written in Python and capable
of running on top of TensorFlow, CNTK, or Theano. It was developed with
a focus on enabling fast experimentation. Being able to go from idea to
result with the least possible delay is key to doing good research.

Use Keras if you need a deep learning library that:

-  Allows for easy and fast prototyping (through user friendliness,
   modularity, and extensibility).
-  Supports both convolutional networks and recurrent networks, as well
   as combinations of the two.
-  Runs seamlessly on CPU and GPU.

+-----------------------+---------------------------------------+
| **SBB License**       | MIT License                           |
+-----------------------+---------------------------------------+
| **Core Technology**   | Python                                |
+-----------------------+---------------------------------------+
| **Project URL**       | https://keras.io/                     |
+-----------------------+---------------------------------------+
| **Source Location**   | https://github.com/keras-team/keras   |
+-----------------------+---------------------------------------+
| **Tag(s)**            | ML                                    |
+-----------------------+---------------------------------------+

| 

Klassify
--------

Redis based text classification service with real-time web interface.

What is Text Classification: Text classification, document
classification or document categorization is a problem in library
science, information science and computer science. The task is to assign
a document to one or more classes or categories.

+-----------------------+-------------------------------------------+
| **SBB License**       | MIT License                               |
+-----------------------+-------------------------------------------+
| **Core Technology**   | Python                                    |
+-----------------------+-------------------------------------------+
| **Project URL**       | https://github.com/fatiherikli/klassify   |
+-----------------------+-------------------------------------------+
| **Source Location**   | https://github.com/fatiherikli/klassify   |
+-----------------------+-------------------------------------------+
| **Tag(s)**            | ML, Text classification                   |
+-----------------------+-------------------------------------------+

| 

Lore
----

Lore is a python framework to make machine learning approachable for
Engineers and maintainable for Data Scientists.

Features

-  Models support hyper parameter search over estimators with a data
   pipeline. They will efficiently utilize multiple GPUs (if available)
   with a couple different strategies, and can be saved and distributed
   for horizontal scalability.
-  Estimators from multiple packages are supported:
   `Keras <https://keras.io/>`__ (TensorFlow/Theano/CNTK),
   `XGBoost <https://xgboost.readthedocs.io/>`__ and `SciKit
   Learn <http://scikit-learn.org/stable/>`__. They can all be
   subclassed with build, fit or predict overridden to completely
   customize your algorithm and architecture, while still benefiting
   from everything else.
-  Pipelines avoid information leaks between train and test sets, and
   one pipeline allows experimentation with many different estimators. A
   disk based pipeline is available if you exceed your machines
   available RAM.
-  Transformers standardize advanced feature engineering. For example,
   convert an American first name to its statistical age or gender using
   US Census data. Extract the geographic area code from a free form
   phone number string. Common date, time and string operations are
   supported efficiently through pandas.
-  Encoders offer robust input to your estimators, and avoid common
   problems with missing and long tail values. They are well tested to
   save you from garbage in/garbage out.
-  IO connections are configured and pooled in a standard way across the
   app for popular (no)sql databases, with transaction management and
   read write optimizations for bulk data, rather than typical ORM
   single row operations. Connections share a configurable query cache,
   in addition to encrypted S3 buckets for distributing models and
   datasets.
-  Dependency Management for each individual app in development, that
   can be 100% replicated to production. No manual activation, or magic
   env vars, or hidden files that break python for everything else. No
   knowledge required of venv, pyenv, pyvenv, virtualenv,
   virtualenvwrapper, pipenv, conda. Ain’t nobody got time for that.
-  Tests for your models can be run in your Continuous Integration
   environment, allowing Continuous Deployment for code and training
   updates, without increased work for your infrastructure team.
-  Workflow Support whether you prefer the command line, a python
   console, jupyter notebook, or IDE. Every environment gets readable
   logging and timing statements configured for both production and
   development.

+-----------------------+----------------------------------------+
| **SBB License**       | GNU General Public License (GPL) 2.0   |
+-----------------------+----------------------------------------+
| **Core Technology**   | Python                                 |
+-----------------------+----------------------------------------+
| **Project URL**       | https://github.com/instacart/lore      |
+-----------------------+----------------------------------------+
| **Source Location**   | https://github.com/instacart/lore      |
+-----------------------+----------------------------------------+
| **Tag(s)**            | ML, Python                             |
+-----------------------+----------------------------------------+

| 

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
models’ training and test performance and to compare them.

Ludwig is built with extensibility principles in mind and is based on
data type abstractions, making it easy to add support for new data types
as well as new model architectures.

It can be used by practitioners to quickly train and test deep learning
models as well as by researchers to obtain strong baselines to compare
against and have an experimentation setting that ensures comparability
by performing standard data preprocessing and visualization.

+-----------------------+----------------------------------+
| **SBB License**       | Apache License 2.0               |
+-----------------------+----------------------------------+
| **Core Technology**   | Python                           |
+-----------------------+----------------------------------+
| **Project URL**       | https://uber.github.io/ludwig/   |
+-----------------------+----------------------------------+
| **Source Location**   | https://github.com/uber/ludwig   |
+-----------------------+----------------------------------+
| **Tag(s)**            | ML                               |
+-----------------------+----------------------------------+

| 

Luminoth
--------

Luminoth is an open source toolkit for computer vision. Currently, we
support object detection and image classification, but we are aiming for
much more. It is built in Python, using TensorFlow and Sonnet.

 

+-----------------------+------------------------------------------------------+
| **SBB License**       | BSD License 2.0 (3-clause, New or Revised) License   |
+-----------------------+------------------------------------------------------+
| **Core Technology**   | Python                                               |
+-----------------------+------------------------------------------------------+
| **Project URL**       | https://luminoth.ai                                  |
+-----------------------+------------------------------------------------------+
| **Source Location**   | https://github.com/tryolabs/luminoth                 |
+-----------------------+------------------------------------------------------+
| **Tag(s)**            | ML                                                   |
+-----------------------+------------------------------------------------------+

| 

MacroBase
---------

MacroBase is a new analytic monitoring engine designed to prioritize
human attention in large-scale datasets and data streams. Unlike a
traditional analytics engine, MacroBase is specialized for one task:
finding and explaining unusual or interesting trends in data. Developed
by `Stanford Future Data Systems <http://futuredata.stanford.edu/>`__

Documentation can be found at: https://macrobase.stanford.edu/docs/

+-----------------------+--------------------------------------------------------------+
| **SBB License**       | Apache License 2.0                                           |
+-----------------------+--------------------------------------------------------------+
| **Core Technology**   | Java                                                         |
+-----------------------+--------------------------------------------------------------+
| **Project URL**       | https://macrobase.stanford.edu/                              |
+-----------------------+--------------------------------------------------------------+
| **Source Location**   | https://github.com/stanford-futuredata/macrobase/tree/v1.0   |
+-----------------------+--------------------------------------------------------------+
| **Tag(s)**            | Data analytics, ML                                           |
+-----------------------+--------------------------------------------------------------+

| 

ml5.js
------

ml5.js aims to make machine learning approachable for a broad audience
of artists, creative coders, and students. The library provides access
to machine learning algorithms and models in the browser, building on
top of `TensorFlow.js <https://js.tensorflow.org/>`__ with no other
external dependencies.

The library is supported by code examples, tutorials, and sample data
sets with an emphasis on ethical computing. Bias in data, stereotypical
harms, and responsible crowdsourcing are part of the documentation
around data collection and usage.

ml5.js is heavily inspired by `Processing <https://processing.org/>`__
and `p5.js <https://p5js.org/>`__.

+-----------------------+----------------------------------------+
| **SBB License**       | MIT License                            |
+-----------------------+----------------------------------------+
| **Core Technology**   | Javascript                             |
+-----------------------+----------------------------------------+
| **Project URL**       | https://ml5js.org/                     |
+-----------------------+----------------------------------------+
| **Source Location**   | https://github.com/ml5js/ml5-library   |
+-----------------------+----------------------------------------+
| **Tag(s)**            | Javascript, ML                         |
+-----------------------+----------------------------------------+

| 

MLflow
------

MLflow offers a way to simplify ML development by making it easy to
track, reproduce, manage, and deploy models. MLflow (currently in alpha)
is an open source platform designed to manage the entire machine
learning lifecycle and work with any machine learning library. It
offers:

-  Record and query experiments: code, data, config, results
-  Packaging format for reproducible runs on any platform
-  General format for sending models to diverse deploy tools

 

+-----------------------+----------------------------------------+
| **SBB License**       | Apache License 2.0                     |
+-----------------------+----------------------------------------+
| **Core Technology**   | Python                                 |
+-----------------------+----------------------------------------+
| **Project URL**       | https://mlflow.org/                    |
+-----------------------+----------------------------------------+
| **Source Location**   | https://github.com/databricks/mlflow   |
+-----------------------+----------------------------------------+
| **Tag(s)**            | ML, Python                             |
+-----------------------+----------------------------------------+

| 

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

-  you can run Machine Learning models in your browser, you don’t need
   to code anything. Just upload dataset, click which attributes to use,
   which algorithms to use and go! This makes Machine Learning super
   easy for everyone and make it possible to get really useful models,
-  there is a python wrapper over MLJAR API, so you don’t need to open
   any browser or click on any button, just write fancy python code! We
   like it and hope you will like it too! To start using MLJAR python
   package please go to our
   `github <https://github.com/mljar/mljar-api-python>`__.

+-----------------------+---------------------------------------------+
| **SBB License**       | MIT License                                 |
+-----------------------+---------------------------------------------+
| **Core Technology**   | Python                                      |
+-----------------------+---------------------------------------------+
| **Project URL**       | https://mljar.com/                          |
+-----------------------+---------------------------------------------+
| **Source Location**   | https://github.com/mljar/mljar-supervised   |
+-----------------------+---------------------------------------------+
| **Tag(s)**            | ML, Python                                  |
+-----------------------+---------------------------------------------+

| 

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

+-----------------------+---------------------------------------+
| **SBB License**       | MIT License                           |
+-----------------------+---------------------------------------+
| **Core Technology**   | Python                                |
+-----------------------+---------------------------------------+
| **Project URL**       | https://mlperf.org/                   |
+-----------------------+---------------------------------------+
| **Source Location**   | https://github.com/mlperf/reference   |
+-----------------------+---------------------------------------+
| **Tag(s)**            | ML, Performance                       |
+-----------------------+---------------------------------------+

| 

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

+-----------------------+-------------------------------------+
| **SBB License**       | MIT License                         |
+-----------------------+-------------------------------------+
| **Core Technology**   | Python, Javascript                  |
+-----------------------+-------------------------------------+
| **Project URL**       | https://mitdbg.github.io/modeldb/   |
+-----------------------+-------------------------------------+
| **Source Location**   | https://github.com/mitdbg/modeldb   |
+-----------------------+-------------------------------------+
| **Tag(s)**            | administration, ML                  |
+-----------------------+-------------------------------------+

| 

Netron
------

Netron is a viewer for neural network, deep learning and machine
learning models.

Netron supports **`ONNX <http://onnx.ai>`__** (``.onnx``, ``.pb``),
**Keras** (``.h5``, ``.keras``), **CoreML** (``.mlmodel``) and
**TensorFlow Lite** (``.tflite``). Netron has experimental support for
**Caffe** (``.caffemodel``), **Caffe2** (``predict_net.pb``), **MXNet**
(``-symbol.json``), **TensorFlow.js** (``model.json``, ``.pb``) and
**TensorFlow** (``.pb``, ``.meta``).

+-----------------------+----------------------------------------+
| **SBB License**       | GNU General Public License (GPL) 2.0   |
+-----------------------+----------------------------------------+
| **Core Technology**   | Python, Javascript                     |
+-----------------------+----------------------------------------+
| **Project URL**       | https://www.lutzroeder.com/ai/         |
+-----------------------+----------------------------------------+
| **Source Location**   | https://github.com/lutzroeder/Netron   |
+-----------------------+----------------------------------------+
| **Tag(s)**            | Data viewer, ML                        |
+-----------------------+----------------------------------------+

| 

Neuralcoref
-----------

State-of-the-art coreference resolution based on neural nets and spaCy.

NeuralCoref is a pipeline extension for spaCy 2.0 that annotates and
resolves coreference clusters using a neural network. NeuralCoref is
production-ready, integrated in spaCy’s NLP pipeline and easily
extensible to new training datasets.

+-----------------------+----------------------------------------------+
| **SBB License**       | MIT License                                  |
+-----------------------+----------------------------------------------+
| **Core Technology**   | Python                                       |
+-----------------------+----------------------------------------------+
| **Project URL**       | https://huggingface.co/coref/                |
+-----------------------+----------------------------------------------+
| **Source Location**   | https://github.com/huggingface/neuralcoref   |
+-----------------------+----------------------------------------------+
| **Tag(s)**            | ML, NLP, Python                              |
+-----------------------+----------------------------------------------+

| 

NLP Architect
-------------

NLP Architect is an open-source Python library for exploring the
state-of-the-art deep learning topologies and techniques for natural
language processing and natural language understanding. It is intended
to be a platform for future research and collaboration.

.. raw:: html

   <div id="how-can-nlp-architect-be-used" class="section">

How can NLP Architect be used:

-  Train models using provided algorithms, reference datasets and
   configurations
-  Train models using your own data
-  Create new/extend models based on existing models or topologies
-  Explore how deep learning models tackle various NLP tasks
-  Experiment and optimize state-of-the-art deep learning algorithms
-  integrate modules and utilities from the library to solutions

.. raw:: html

   </div>

+-----------------------+---------------------------------------------------+
| **SBB License**       | Apache License 2.0                                |
+-----------------------+---------------------------------------------------+
| **Core Technology**   | Python                                            |
+-----------------------+---------------------------------------------------+
| **Project URL**       | http://nlp_architect.nervanasys.com/              |
+-----------------------+---------------------------------------------------+
| **Source Location**   | https://github.com/NervanaSystems/nlp-architect   |
+-----------------------+---------------------------------------------------+
| **Tag(s)**            | ML, NLP, Python                                   |
+-----------------------+---------------------------------------------------+

| 

NNI (Neural Network Intelligence)
---------------------------------

NNI (Neural Network Intelligence) is a toolkit to help users run
automated machine learning (AutoML) experiments. The tool dispatches and
runs trial jobs generated by tuning algorithms to search the best neural
architecture and/or hyper-parameters in different environments like
local machine, remote servers and cloud. (Microsoft ML project)

Who should consider using NNI:

-  Those who want to try different AutoML algorithms in their training
   code (model) at their local machine.
-  Those who want to run AutoML trial jobs in different environments to
   speed up search (e.g. remote servers and cloud).
-  Researchers and data scientists who want to implement their own
   AutoML algorithms and compare it with other algorithms.
-  ML Platform owners who want to support AutoML in their platform.

+-----------------------+-----------------------------------------+
| **SBB License**       | MIT License                             |
+-----------------------+-----------------------------------------+
| **Core Technology**   | Python                                  |
+-----------------------+-----------------------------------------+
| **Project URL**       | https://nni.readthedocs.io/en/latest/   |
+-----------------------+-----------------------------------------+
| **Source Location**   | https://github.com/Microsoft/nni        |
+-----------------------+-----------------------------------------+
| **Tag(s)**            | ML                                      |
+-----------------------+-----------------------------------------+

| 

ONNX
----

ONNX provides an open source format for AI models. It defines an
extensible computation graph model, as well as definitions of built-in
operators and standard data types. Initially we focus on the
capabilities needed for inferencing (evaluation).

Caffe2, PyTorch, Microsoft Cognitive Toolkit, Apache MXNet and other
tools are developing ONNX support. Enabling interoperability between
different frameworks and streamlining the path from research to
production will increase the speed of innovation in the AI community. We
are an early stage and we invite the community to submit feedback and
help us further evolve ONNX.

Companies behind ONNX are AWS, Facebook and Microsoft Corporation and
more.

+-----------------------+--------------------------------+
| **SBB License**       | MIT License                    |
+-----------------------+--------------------------------+
| **Core Technology**   | Python                         |
+-----------------------+--------------------------------+
| **Project URL**       | http://onnx.ai/                |
+-----------------------+--------------------------------+
| **Source Location**   | https://github.com/onnx/onnx   |
+-----------------------+--------------------------------+
| **Tag(s)**            | AI, ML                         |
+-----------------------+--------------------------------+

| 

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

+-----------------------+------------------------------------------------------+
| **SBB License**       | BSD License 2.0 (3-clause, New or Revised) License   |
+-----------------------+------------------------------------------------------+
| **Core Technology**   | C                                                    |
+-----------------------+------------------------------------------------------+
| **Project URL**       | https://opencv.org/                                  |
+-----------------------+------------------------------------------------------+
| **Source Location**   | https://github.com/opencv/opencv                     |
+-----------------------+------------------------------------------------------+
| **Tag(s)**            | ML                                                   |
+-----------------------+------------------------------------------------------+

| 

OpenML
------

OpenML is an on-line machine learning platform for sharing and
organizing data, machine learning algorithms and experiments. It claims
to be designed to create a frictionless, networked ecosystem, so that
you can readily integrate into your existing
processes/code/environments. It also allows people from all over the
world to collaborate and build directly on each other’s latest ideas,
data and results, irrespective of the tools and infrastructure they
happen to use. So nice ideas to build an open science movement. The
people behind OpemML are mostly (data)scientist. So using this product
for real world business use cases will take some extra effort.

Altrhough OpenML is exposed as an foundation based on openness, a quick
inspection learned that the OpenML platform  is not as open as you want.
Also the OSS software is not created to be run on premise. So be aware
when doing large (time) investments into this OpenML platform.

+-----------------------+------------------------------------------------------+
| **SBB License**       | BSD License 2.0 (3-clause, New or Revised) License   |
+-----------------------+------------------------------------------------------+
| **Core Technology**   | Java                                                 |
+-----------------------+------------------------------------------------------+
| **Project URL**       | https://openml.org                                   |
+-----------------------+------------------------------------------------------+
| **Source Location**   | https://github.com/openml/OpenML                     |
+-----------------------+------------------------------------------------------+
| **Tag(s)**            | ML                                                   |
+-----------------------+------------------------------------------------------+

| 

Orange
------

Orange is a comprehensive, component-based software suite for machine
learning and data mining, developed at Bioinformatics Laboratory.

Orange is available by default on Anaconda Navigator dashboard.
`Orange <http://orange.biolab.si/>`__ is a component-based data mining
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

 

 

+-----------------------+----------------------------------------+
| **SBB License**       | GNU General Public License (GPL) 3.0   |
+-----------------------+----------------------------------------+
| **Core Technology**   |                                        |
+-----------------------+----------------------------------------+
| **Project URL**       | https://orange.biolab.si/              |
+-----------------------+----------------------------------------+
| **Source Location**   | https://github.com/biolab/orange3      |
+-----------------------+----------------------------------------+
| **Tag(s)**            | Data Visualization, ML, Python         |
+-----------------------+----------------------------------------+

| 

Pattern
-------

Pattern is a web mining module for Python. It has tools for:

-  Data Mining: web services (Google, Twitter, Wikipedia), web crawler,
   HTML DOM parser
-  Natural Language Processing: part-of-speech taggers, n-gram search,
   sentiment analysis, WordNet
-  Machine Learning: vector space model, clustering, classification
   (KNN, SVM, Perceptron)
-  Network Analysis: graph centrality and visualization.

+-----------------------+------------------------------------------------------+
| **SBB License**       | BSD License 2.0 (3-clause, New or Revised) License   |
+-----------------------+------------------------------------------------------+
| **Core Technology**   | Python                                               |
+-----------------------+------------------------------------------------------+
| **Project URL**       | https://www.clips.uantwerpen.be/pages/pattern        |
+-----------------------+------------------------------------------------------+
| **Source Location**   | https://github.com/clips/pattern                     |
+-----------------------+------------------------------------------------------+
| **Tag(s)**            | ML, NLP, Web scraping                                |
+-----------------------+------------------------------------------------------+

| 

Plait
-----

plait.py is a program for generating fake data from composable yaml
templates.

With plait it is easy to model fake data that has an interesting shape.
Currently, many fake data generators model their data as a collection of
`IID <https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables>`__
variables; with plait.py we can stitch together those variables into a
more coherent model.

Example uses for plait.py are:

-  generating mock application data in test environments
-  validating the usefulness of statistical techniques
-  creating synthetic datasets for performance tuning databases

+-----------------------+---------------------------------------+
| **SBB License**       | MIT License                           |
+-----------------------+---------------------------------------+
| **Core Technology**   | Python                                |
+-----------------------+---------------------------------------+
| **Project URL**       | https://github.com/plaitpy/plaitpy    |
+-----------------------+---------------------------------------+
| **Source Location**   | https://github.com/plaitpy/plaitpy    |
+-----------------------+---------------------------------------+
| **Tag(s)**            | Data Generator, ML, text generation   |
+-----------------------+---------------------------------------+

| 

Polyaxon
--------

An open source platform for reproducible machine learning at scale.

Polyaxon is a platform for building, training, and monitoring large
scale deep learning applications.

Polyaxon deploys into any data center, cloud provider, or can be hosted
and managed by Polyaxon, and it supports all the major deep learning
frameworks such as Tensorflow, MXNet, Caffe, Torch, etc.

Polyaxon makes it faster, easier, and more efficient to develop deep
learning applications by managing workloads with smart container and
node management. And it turns GPU servers into shared, self-service
resources for your team or organization.

+-----------------------+----------------------------------------+
| **SBB License**       | MIT License                            |
+-----------------------+----------------------------------------+
| **Core Technology**   | Python                                 |
+-----------------------+----------------------------------------+
| **Project URL**       | https://polyaxon.com/                  |
+-----------------------+----------------------------------------+
| **Source Location**   | https://github.com/polyaxon/polyaxon   |
+-----------------------+----------------------------------------+
| **Tag(s)**            | ML                                     |
+-----------------------+----------------------------------------+

| 

Pylearn2
--------

Pylearn2 is a library designed to make machine learning research easy.

+-----------------------+------------------------------------------------------+
| **SBB License**       | BSD License 2.0 (3-clause, New or Revised) License   |
+-----------------------+------------------------------------------------------+
| **Core Technology**   | Python                                               |
+-----------------------+------------------------------------------------------+
| **Project URL**       | http://deeplearning.net/software/pylearn2/           |
+-----------------------+------------------------------------------------------+
| **Source Location**   | https://github.com/lisa-lab/pylearn2                 |
+-----------------------+------------------------------------------------------+
| **Tag(s)**            | ML                                                   |
+-----------------------+------------------------------------------------------+

| 

Pyodide
-------

**Pyodide** brings the Python runtime to the browser via WebAssembly,
along with the Python scientific stack including NumPy, Pandas,
Matplotlib, parts of SciPy, and NetworkX. The ```packages``
directory <https://github.com/iodide-project/pyodide/tree/master/packages>`__
lists over 35 packages which are currently available.

**Pyodide** provides transparent conversion of objects between
Javascript and Python. When used inside a browser, Python has full
access to the Web APIs.

While closely related to the `iodide project <https://iodide.io>`__, a
tool for *literate scientific computing and communication for the web*,
Pyodide goes beyond running in a notebook environment. To maximize the
flexibility of the modern web, **Pyodide** may be used standalone in any
context where you want to **run Python inside a web browser**.

+-----------------------+--------------------------------------------------------------------------------------------------+
| **SBB License**       | Mozilla Public License (MPL) 1.1                                                                 |
+-----------------------+--------------------------------------------------------------------------------------------------+
| **Core Technology**   | C, Javascript                                                                                    |
+-----------------------+--------------------------------------------------------------------------------------------------+
| **Project URL**       | https://hacks.mozilla.org/2019/04/pyodide-bringing-the-scientific-python-stack-to-the-browser/   |
+-----------------------+--------------------------------------------------------------------------------------------------+
| **Source Location**   | https://github.com/iodide-project/pyodide/                                                       |
+-----------------------+--------------------------------------------------------------------------------------------------+
| **Tag(s)**            | Data Visualization, ML, Python                                                                   |
+-----------------------+--------------------------------------------------------------------------------------------------+

| 

Pyro
----

Deep universal probabilistic programming with Python and PyTorch. Pyro
is in an alpha release. It is developed and used by `Uber AI
Labs <http://uber.ai>`__.

 

+-----------------------+----------------------------------------+
| **SBB License**       | GNU General Public License (GPL) 2.0   |
+-----------------------+----------------------------------------+
| **Core Technology**   | Python                                 |
+-----------------------+----------------------------------------+
| **Project URL**       | http://pyro.ai/                        |
+-----------------------+----------------------------------------+
| **Source Location**   | https://github.com/uber/pyro           |
+-----------------------+----------------------------------------+
| **Tag(s)**            | AI, ML, Python                         |
+-----------------------+----------------------------------------+

| 

Pythia
------

Pythia is a modular framework for supercharging vision and language
research built on top of PyTorch created by Facebook.

You can use Pythia to bootstrap for your next vision and language
multimodal research project. Pythia can also act as starter codebase for
challenges around vision and language datasets (TextVQA challenge, VQA
challenge).

It features:

-  **Model Zoo**: Reference implementations for state-of-the-art vision
   and language model including
   `LoRRA <https://arxiv.org/abs/1904.08920>`__ (SoTA on VQA and
   TextVQA), `Pythia <https://arxiv.org/abs/1807.09956>`__ model (VQA
   2018 challenge winner) and
   `BAN <https://github.com/facebookresearch/pythia/blob/master>`__.
-  **Multi-Tasking**: Support for multi-tasking which allows training on
   multiple dataset together.
-  **Datasets**: Includes support for various datasets built-in
   including VQA, VizWiz, TextVQA and VisualDialog.
-  **Modules**: Provides implementations for many commonly used layers
   in vision and language domain
-  **Distributed**: Support for distributed training based on
   DataParallel as well as DistributedDataParallel.
-  **Unopinionated**: Unopinionated about the dataset and model
   implementations built on top of it.
-  **Customization**: Custom losses, metrics, scheduling, optimizers,
   tensorboard; suits all your custom needs.

+-----------------------+-----------------------------------------------------------+
| **SBB License**       | BSD License 2.0 (3-clause, New or Revised) License        |
+-----------------------+-----------------------------------------------------------+
| **Core Technology**   | Python                                                    |
+-----------------------+-----------------------------------------------------------+
| **Project URL**       | https://learnpythia.readthedocs.io/en/latest/index.html   |
+-----------------------+-----------------------------------------------------------+
| **Source Location**   | https://github.com/facebookresearch/pythia                |
+-----------------------+-----------------------------------------------------------+
| **Tag(s)**            | ML, Python                                                |
+-----------------------+-----------------------------------------------------------+

| 

PyTorch
-------

PyTorch is:

-  a deep learning framework that puts Python first.
-   a research-focused framework.
-  Python package that provides two high-level features:

Pytorch uses tensor computation (like NumPy) with strong GPU
acceleration. It can use deep neural networks built on a tape-based
autograd system.

You can reuse your favorite Python packages such as NumPy, SciPy and
Cython to extend PyTorch when needed.

Note: PyTorch is still in an early-release beta phase (status January
2018). PyTorch was released as OSS by Google January 2017.

+-----------------------+--------------------------------------+
| **SBB License**       | MIT License                          |
+-----------------------+--------------------------------------+
| **Core Technology**   | Python                               |
+-----------------------+--------------------------------------+
| **Project URL**       | http://pytorch.org/                  |
+-----------------------+--------------------------------------+
| **Source Location**   | https://github.com/pytorch/pytorch   |
+-----------------------+--------------------------------------+
| **Tag(s)**            | AI, ML                               |
+-----------------------+--------------------------------------+

| 

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

-  Recursive, weighted branching with several selection modes
-  Queryable dictionaries
-  Automatic capitalization, rhyming, English indefinite articles, and
   multi-lingual number verbalization
-  Print to multiple separate outputs
-  Probability modifiers for pattern elements
-  Loops, conditional statements, and subroutines
-  Fully-functional object model
-  Import/Export resources easily with the .rantpkg format
-  Compatible with Unity 2017

+-----------------------+-------------------------------------+
| **SBB License**       | MIT License                         |
+-----------------------+-------------------------------------+
| **Core Technology**   | .NET                                |
+-----------------------+-------------------------------------+
| **Project URL**       | https://berkin.me/rant/             |
+-----------------------+-------------------------------------+
| **Source Location**   | https://github.com/TheBerkin/rant   |
+-----------------------+-------------------------------------+
| **Tag(s)**            | .NET, ML, NLP, text generation      |
+-----------------------+-------------------------------------+

| 

RAPIDS
------

The RAPIDS suite of software libraries gives you the freedom to execute
end-to-end data science and analytics pipelines entirely on GPUs. It
relies on `NVIDIA® CUDA® <https://developer.nvidia.com/cuda-toolkit>`__
primitives for low-level compute optimization, but exposes that GPU
parallelism and high-bandwidth memory speed through user-friendly Python
interfaces.

RAPIDS also focuses on common data preparation tasks for analytics and
data science. This includes a familiar DataFrame API that integrates
with a variety of machine learning algorithms for end-to-end pipeline
accelerations without paying typical serialization costs–. RAPIDS also
includes support for multi-node, multi-GPU deployments, enabling vastly
accelerated processing and training on much larger dataset sizes.

+-----------------------+--------------------------------+
| **SBB License**       | Apache License 2.0             |
+-----------------------+--------------------------------+
| **Core Technology**   | C++                            |
+-----------------------+--------------------------------+
| **Project URL**       | http://rapids.ai/              |
+-----------------------+--------------------------------+
| **Source Location**   | https://github.com/rapidsai/   |
+-----------------------+--------------------------------+
| **Tag(s)**            | ML                             |
+-----------------------+--------------------------------+

| 

Ray
---

Ray is a flexible, high-performance distributed execution framework for
AI applications. Ray is currently under heavy development. But Ray has
already a good start, with good documentation
(http://ray.readthedocs.io/en/latest/index.html) and a tutorial. Also
Ray is backed by scientific researchers and published papers.

Ray comes with libraries that accelerate deep learning and reinforcement
learning development:

-  `Ray Tune <http://ray.readthedocs.io/en/latest/tune.html>`__:
   Hyperparameter Optimization Framework
-  `Ray RLlib <http://ray.readthedocs.io/en/latest/rllib.html>`__: A
   Scalable Reinforcement Learning Library

+-----------------------+--------------------------------------+
| **SBB License**       | Apache License 2.0                   |
+-----------------------+--------------------------------------+
| **Core Technology**   | Python                               |
+-----------------------+--------------------------------------+
| **Project URL**       | https://ray-project.github.io/       |
+-----------------------+--------------------------------------+
| **Source Location**   | https://github.com/ray-project/ray   |
+-----------------------+--------------------------------------+
| **Tag(s)**            | ML                                   |
+-----------------------+--------------------------------------+

| 

Scikit-learn
------------

scikit-learn is a Python module for machine learning.

Simple and efficient tools for data mining and data analysis

-  Accessible to everybody, and reusable in various contexts
-  Built on NumPy, SciPy, and matplotlib

+-----------------------+------------------------------------------------------+
| **SBB License**       | BSD License 2.0 (3-clause, New or Revised) License   |
+-----------------------+------------------------------------------------------+
| **Core Technology**   | Python                                               |
+-----------------------+------------------------------------------------------+
| **Project URL**       | http://scikit-learn.org                              |
+-----------------------+------------------------------------------------------+
| **Source Location**   | https://github.com/scikit-learn/scikit-learn         |
+-----------------------+------------------------------------------------------+
| **Tag(s)**            | ML                                                   |
+-----------------------+------------------------------------------------------+

| 

Skater
------

Skater is a python package for model agnostic interpretation of
predictive models. With Skater, you can unpack the internal mechanics of
arbitrary models; as long as you can obtain inputs, and use a function
to obtain outputs, you can use Skater to learn about the models internal
decision policies.

The project was started as a research idea to find ways to enable better
interpretability(preferably human interpretability) to predictive “black
boxes” both for researchers and practioners.

Documentation at:\ https://datascienceinc.github.io/Skater/overview.html

+-----------------------+------------------------------------------------------+
| **SBB License**       | MIT License                                          |
+-----------------------+------------------------------------------------------+
| **Core Technology**   | Python                                               |
+-----------------------+------------------------------------------------------+
| **Project URL**       | https://www.datascience.com/resources/tools/skater   |
+-----------------------+------------------------------------------------------+
| **Source Location**   | https://github.com/datascienceinc/Skater             |
+-----------------------+------------------------------------------------------+
| **Tag(s)**            | ML                                                   |
+-----------------------+------------------------------------------------------+

| 

Snorkel
-------

Snorkel is a system for rapidly **creating, modeling, and managing
training data**, currently focused on accelerating the development of
*structured or “dark” data extraction applications* for domains in which
large labeled training sets are not available or easy to obtain.

+-----------------------+-------------------------------------------+
| **SBB License**       | Apache License 2.0                        |
+-----------------------+-------------------------------------------+
| **Core Technology**   | Python                                    |
+-----------------------+-------------------------------------------+
| **Project URL**       | https://hazyresearch.github.io/snorkel/   |
+-----------------------+-------------------------------------------+
| **Source Location**   | https://github.com/HazyResearch/snorkel   |
+-----------------------+-------------------------------------------+
| **Tag(s)**            | ML                                        |
+-----------------------+-------------------------------------------+

| 

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
Brain Team within Google’s Machine Intelligence research organization
for the purposes of conducting machine learning and deep neural networks
research, but the system is general enough to be applicable in a wide
variety of other domains as well.

TensorFlow comes with a tool called
`TensorBoard <https://www.tensorflow.org/versions/r0.11/how_tos/graph_viz/index.html>`__
which you can use to get some insight into what is happening.
TensorBoard is a suite of web applications for inspecting and
understanding your TensorFlow runs and graphs.

There is also a version of TensorFlow that runs in a browser. This is
TensorFlow.js (https://js.tensorflow.org/ ). TensorFlow.js is a WebGL
accelerated, browser based JavaScript library for training and deploying
ML models.

 

+-----------------------+--------------------------------------------+
| **SBB License**       | Apache License 2.0                         |
+-----------------------+--------------------------------------------+
| **Core Technology**   | C                                          |
+-----------------------+--------------------------------------------+
| **Project URL**       | https://www.tensorflow.org/                |
+-----------------------+--------------------------------------------+
| **Source Location**   | https://github.com/tensorflow/tensorflow   |
+-----------------------+--------------------------------------------+
| **Tag(s)**            | AI, ML                                     |
+-----------------------+--------------------------------------------+

| 

TensorWatch
-----------

TensorWatch is a debugging and visualization tool designed for data
science, deep learning and reinforcement learning from Microsoft
Research. It works in Jupyter Notebook to show real-time visualizations
of your machine learning training and perform several other key analysis
tasks for your models and data.

TensorWatch is designed to be flexible and extensible so you can also
build your own custom visualizations, UIs, and dashboards. Besides
traditional “what-you-see-is-what-you-log” approach, it also has a
unique capability to execute arbitrary queries against your live ML
training process, return a stream as a result of the query and view this
stream using your choice of a visualizer (we call this `Lazy Logging
Mode <https://github.com/microsoft/tensorwatch#lazy-logging-mode%5D>`__).

TensorWatch is under heavy development with a goal of providing a
platform for debugging machine learning in one easy to use, extensible,
and hackable package.

+-----------------------+--------------------------------------------+
| **SBB License**       | MIT License                                |
+-----------------------+--------------------------------------------+
| **Core Technology**   | Python                                     |
+-----------------------+--------------------------------------------+
| **Project URL**       | https://github.com/microsoft/tensorwatch   |
+-----------------------+--------------------------------------------+
| **Source Location**   | https://github.com/microsoft/tensorwatch   |
+-----------------------+--------------------------------------------+
| **Tag(s)**            | ML                                         |
+-----------------------+--------------------------------------------+

| 

TextBlob: Simplified Text Processing
------------------------------------

*TextBlob* is a Python (2 and 3) library for processing textual data. It
provides a simple API for diving into common natural language processing
(NLP) tasks such as part-of-speech tagging, noun phrase extraction,
sentiment analysis, classification, translation, and more.

Features
--------

-  Noun phrase extraction
-  Part-of-speech tagging
-  Sentiment analysis
-  Classification (Naive Bayes, Decision Tree)
-  Language translation and detection powered by Google Translate
-  Tokenization (splitting text into words and sentences)
-  Word and phrase frequencies
-  Parsing
-  n-grams
-  Word inflection (pluralization and singularization) and lemmatization
-  Spelling correction
-  Add new models or languages through extensions
-  WordNet integration

+-----------------------+-------------------------------------------+
| **SBB License**       | MIT License                               |
+-----------------------+-------------------------------------------+
| **Core Technology**   | Python                                    |
+-----------------------+-------------------------------------------+
| **Project URL**       | https://textblob.readthedocs.io/en/dev/   |
+-----------------------+-------------------------------------------+
| **Source Location**   | https://github.com/sloria/textblob        |
+-----------------------+-------------------------------------------+
| **Tag(s)**            | ML, NLP, Python                           |
+-----------------------+-------------------------------------------+

| 

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

+-----------------------+------------------------------------+
| **SBB License**       | MIT License                        |
+-----------------------+------------------------------------+
| **Core Technology**   | Python                             |
+-----------------------+------------------------------------+
| **Project URL**       | http://www.deeplearning.net/       |
+-----------------------+------------------------------------+
| **Source Location**   | https://github.com/Theano/Theano   |
+-----------------------+------------------------------------+
| **Tag(s)**            | ML, Python                         |
+-----------------------+------------------------------------+

| 

Thinc
-----

Thinc is the machine learning library powering spaCy. It features a
battle-tested linear model designed for large sparse learning problems,
and a flexible neural network model under development for spaCy v2.0.

Thinc is a practical toolkit for implementing models that follow the
“Embed, encode, attend, predict” architecture. It’s designed to be easy
to install, efficient for CPU usage and optimised for NLP and deep
learning with text – in particular, hierarchically structured input and
variable-length sequences.

+-----------------------+----------------------------------------+
| **SBB License**       | GNU General Public License (GPL) 2.0   |
+-----------------------+----------------------------------------+
| **Core Technology**   | Python                                 |
+-----------------------+----------------------------------------+
| **Project URL**       | https://explosion.ai/                  |
+-----------------------+----------------------------------------+
| **Source Location**   | https://github.com/explosion/thinc     |
+-----------------------+----------------------------------------+
| **Tag(s)**            | ML, NLP, Python                        |
+-----------------------+----------------------------------------+

| 

Turi
----

Turi Create simplifies the development of custom machine learning
models. Turi is OSS machine learning from Apple.

Turi Create simplifies the development of custom machine learning
models. You don’t have to be a machine learning expert to add
recommendations, object detection, image classification, image
similarity or activity classification to your app.

+-----------------------+------------------------------------------------------+
| **SBB License**       | BSD License 2.0 (3-clause, New or Revised) License   |
+-----------------------+------------------------------------------------------+
| **Core Technology**   | Python                                               |
+-----------------------+------------------------------------------------------+
| **Project URL**       | https://github.com/apple/turicreate                  |
+-----------------------+------------------------------------------------------+
| **Source Location**   | https://github.com/apple/turicreate                  |
+-----------------------+------------------------------------------------------+
| **Tag(s)**            | ML                                                   |
+-----------------------+------------------------------------------------------+

| 

TuriCreate
----------

This SBB is from Apple. Apple, is with Siri already for a long time
active in machine learning. But even Apple is releasing building blocks
under OSS licenses now.

Turi Create simplifies the development of custom machine learning
models. You don’t have to be a machine learning expert to add
recommendations, object detection, image classification, image
similarity or activity classification to your app.

-  **Easy-to-use:** Focus on tasks instead of algorithms
-  **Visual:** Built-in, streaming visualizations to explore your data
-  **Flexible:** Supports text, images, audio, video and sensor data
-  **Fast and Scalable:** Work with large datasets on a single machine
-  **Ready To Deploy:** Export models to Core ML for use in iOS, macOS,
   watchOS, and tvOS apps

+-----------------------+------------------------------------------------------+
| **SBB License**       | BSD License 2.0 (3-clause, New or Revised) License   |
+-----------------------+------------------------------------------------------+
| **Core Technology**   | Python                                               |
+-----------------------+------------------------------------------------------+
| **Project URL**       | https://turi.com/index.html                          |
+-----------------------+------------------------------------------------------+
| **Source Location**   | https://github.com/apple/turicreate                  |
+-----------------------+------------------------------------------------------+
| **Tag(s)**            | ML, Python                                           |
+-----------------------+------------------------------------------------------+

| 

VisualDL
--------

VisualDL is an open-source cross-framework web dashboard that richly
visualizes the performance and data flowing through your neural network
training. VisualDL is a deep learning visualization tool that can help
design deep learning jobs. It includes features such as scalar,
parameter distribution, model structure and image visualization.

+-----------------------+--------------------------------------------+
| **SBB License**       | Apache License 2.0                         |
+-----------------------+--------------------------------------------+
| **Core Technology**   | C++                                        |
+-----------------------+--------------------------------------------+
| **Project URL**       | http://visualdl.paddlepaddle.org/          |
+-----------------------+--------------------------------------------+
| **Source Location**   | https://github.com/PaddlePaddle/VisualDL   |
+-----------------------+--------------------------------------------+
| **Tag(s)**            | ML                                         |
+-----------------------+--------------------------------------------+

| 

What-If Tool
------------

The `What-If Tool <https://pair-code.github.io/what-if-tool>`__ (WIT)
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

+-----------------------+---------------------------------------------------------------------------------------------------+
| **SBB License**       | Apache License 2.0                                                                                |
+-----------------------+---------------------------------------------------------------------------------------------------+
| **Core Technology**   | Python                                                                                            |
+-----------------------+---------------------------------------------------------------------------------------------------+
| **Project URL**       | https://pair-code.github.io/what-if-tool/                                                         |
+-----------------------+---------------------------------------------------------------------------------------------------+
| **Source Location**   | https://github.com/tensorflow/tensorboard/tree/master/tensorboard/plugins/interactive_inference   |
+-----------------------+---------------------------------------------------------------------------------------------------+
| **Tag(s)**            | ML                                                                                                |
+-----------------------+---------------------------------------------------------------------------------------------------+

| 

XAI
---

XAI is a Machine Learning library that is designed with AI
explainability in its core. XAI contains various tools that enable for
analysis and evaluation of data and models. The XAI library is
maintained by `The Institute for Ethical AI &
ML <http://ethical.institute/>`__, and it was developed based on the `8
principles for Responsible Machine
Learning <http://ethical.institute/principles.html>`__.

You can find the documentation at
https://ethicalml.github.io/xai/index.html.

+-----------------------+----------------------------------------+
| **SBB License**       | MIT License                            |
+-----------------------+----------------------------------------+
| **Core Technology**   | Python                                 |
+-----------------------+----------------------------------------+
| **Project URL**       | https://ethical.institute/index.html   |
+-----------------------+----------------------------------------+
| **Source Location**   | https://github.com/EthicalML/xai       |
+-----------------------+----------------------------------------+
| **Tag(s)**            | ML, Python                             |
+-----------------------+----------------------------------------+

| 
| End of SBB list
