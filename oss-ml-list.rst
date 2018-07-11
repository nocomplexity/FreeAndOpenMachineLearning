**Acumos AI** 
^^^^^^^^^^^^^^^
**SBB Description:** Acumos AI is a platform and open source framework that makes it easy to build, share, and deploy AI apps. Acumos standardizes the infrastructure stack and components required to run an out-of-the-box general AI environment.
Acumos is a platform which enhances the development, training and deployment of AI models. Its purpose is to scale up the introduction of AI-based software across a wide range of industrial and commercial problems in order to reach a critical mass of applications. In this way, Acumos will drive toward a data-centric process for producing software based upon machine learning as the central paradigm. The platform seeks to empower data scientists to publish more adaptive AI models and shield them from the task of custom development of fully integrated solutions. Ideally, software developers will use Acumos to change the process of software development from a code-writing and editing exercise into a classroom-like code training process in which models will be trained and graded on their ability to successfully analyze datasets that they are fed. Then, the best model can be selected for the job and integrated into a complete application.
Acumos is part of the LF Deep Learning Foundation, an umbrella organization within The Linux Foundation that supports and sustains open source innovation in artificial intelligence, machine learning, and deep learning while striving to make these critical new technologies available to developers and data scientists everywhere.

**SBB License:** Apache License 2.0

**Core Technology:** Java

**Project URL:** https://www.acumos.org/

**Source Location:** https://gerrit.acumos.org/r/#/admin/projects/





**AllenNLP** 
^^^^^^^^^^^^^^
**SBB Description:** An open-source NLP research library, built on PyTorch. AllenNLP is a NLP research library, built on PyTorch, for developing state-of-the-art deep learning models on a wide variety of linguistic tasks. AllenNLP makes it easy to design and evaluate new deep learning models for nearly any NLP problem, along with the infrastructure to easily run them in the cloud or on your laptop.

AllenNLP was designed with the following principles:

Hyper-modular and lightweight. Use the parts which you like seamlessly with PyTorch.
Extensively tested and easy to extend. Test coverage is above 90% and the example models provide a template for contributions.
Take padding and masking seriously, making it easy to implement correct models without the pain.
Experiment friendly. Run reproducible experiments from a json specification with comprehensive logging.

**SBB License:** Apache License 2.0

**Core Technology:** Python

**Project URL:** http://allennlp.org/

**Source Location:** https://github.com/allenai/allennlp





**Apache MXNet** 
^^^^^^^^^^^^^^^^^^
**SBB Description:** Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler; for Python, R, Julia, Scala, Go, Javascript and more.
All major GPU and CPU vendors support this project, but also the real giants like Amazon, Microsoft, Wolfram and a number of very respected universities. So watch this project or play with it to see if it fits your use case.
Apache MXNet (incubating) is a deep learning framework designed for both efficiency and flexibility. It allows you to mix symbolic and imperative programming to maximize efficiency and productivity. At its core, MXNet contains a dynamic dependency scheduler that automatically parallelizes both symbolic and imperative operations on the fly. A graph optimization layer on top of that makes symbolic execution fast and memory efficient. MXNet is portable and lightweight, scaling effectively to multiple GPUs and multiple machines.
MXNet is also more than a deep learning project. It is also a collection of blue prints and guidelines for building deep learning systems, and interesting insights of DL systems for hackers.
Gluon is the high-level interface for MXNet. It is more intuitive and easier to use than the lower level interface. Gluon supports dynamic (define-by-run) graphs with JIT-compilation to achieve both flexibility and efficiency. The perfect starters documentation with a great crash course on deep learning can be found here: http://gluon.mxnet.io/
Part of the project is also the the Gluon API specification (see https://github.com/gluon-api/gluon-api)
The Gluon API specification (Python based) is an effort to improve speed, flexibility, and accessibility of deep learning technology for all developers, regardless of their deep learning framework of choice. The Gluon API offers a flexible interface that simplifies the process of prototyping, building, and training deep learning models without sacrificing training speed.

**SBB License:** Apache License 2.0

**Core Technology:** 

**Project URL:** http://mxnet.incubator.apache.org/

**Source Location:** https://github.com/apache/incubator-mxnet





**Apache Spark MLlib** 
^^^^^^^^^^^^^^^^^^^^^^^^
**SBB Description:** Apache Spark MLlib. MLlib is Apache Spark&#8217;s scalable machine learning library.
Apache Spark is a OSS platform for large-scale data processing. The Spark engine is written in Scala and is well suited for applications that reuse a working set of data across multiple parallel operations. It’s designed to work as a standalone cluster or as part of Hadoop YARN cluster. It can access data from sources such as HDFS, Cassandra or Amazon S3. MLlib can be seen as a core Spark&#8217;s APIs and interoperates with NumPy in Python and R libraries. And Spark is very fast!
MLlib library contains many algorithms and utilities, e.g.:

Classification: logistic regression, naive Bayes,&#8230;
Regression: generalized linear regression, survival regression,&#8230;
Decision trees, random forests, and gradient-boosted trees
Recommendation: alternating least squares (ALS)
Clustering: K-means, Gaussian mixtures (GMMs),&#8230;
Topic modeling: latent Dirichlet allocation (LDA)
Frequent itemsets, association rules, and sequential pattern mining

**SBB License:** Apache License 2.0

**Core Technology:** Java

**Project URL:** https://spark.apache.org/mllib/

**Source Location:** https://github.com/apache/spark





**Apollo** 
^^^^^^^^^^^^
**SBB Description:** Apollo is a high performance, flexible architecture which accelerates the development, testing, and deployment of Autonomous Vehicles.

**SBB License:** GNU General Public License (GPL) 2.0

**Core Technology:** C++

**Project URL:** http://apollo.auto/

**Source Location:** https://github.com/ApolloAuto/apollo





**auto_ml** 
^^^^^^^^^^^^^
**SBB Description:** Automated machine learning for analytics &#38; production.
Automates the whole machine learning process, making it super easy to use for both analytics, and getting real-time predictions in production.

**SBB License:** MIT License

**Core Technology:** Python

**Project URL:** http://auto-ml.readthedocs.io

**Source Location:** https://github.com/ClimbsRocks/auto_ml





**BigDL** 
^^^^^^^^^^^
**SBB Description:** BigDL is a distributed deep learning library for Apache Spark; with BigDL, users can write their deep learning applications as standard Spark programs, which can directly run on top of existing Spark or Hadoop clusters.

Rich deep learning support. Modeled after Torch, BigDL provides comprehensive support for deep learning, including numeric computing (via Tensor) and high level neural networks; in addition, users can load pre-trained Caffe or Torch or Keras models into Spark programs using BigDL.
Extremely high performance. To achieve high performance, BigDL uses Intel MKL and multi-threaded programming in each Spark task. Consequently, it is orders of magnitude faster than out-of-box open source Caffe, Torch or TensorFlow on a single-node Xeon (i.e., comparable with mainstream GPU).
Efficiently scale-out. BigDL can efficiently scale out to perform data analytics at &#8220;Big Data scale&#8221;, by leveraging Apache Spark (a lightning fast distributed data processing framework), as well as efficient implementations of synchronous SGD and all-reduce communications on Spark.

**SBB License:** Apache License 2.0

**Core Technology:** Java

**Project URL:** https://bigdl-project.github.io/master/

**Source Location:** https://github.com/intel-analytics/BigDL





**Blocks** 
^^^^^^^^^^^^
**SBB Description:** Blocks is a framework that is supposed to make it easier to build complicated neural network models on top of Theano.
Blocks is a framework that helps you build neural network models on top of Theano. Currently it supports and provides:

Constructing parametrized Theano operations, called &#8220;bricks&#8221;
Pattern matching to select variables and bricks in large models
Algorithms to optimize your model
Saving and resuming of training
Monitoring and analyzing values during training progress (on the training set as well as on test sets)
Application of graph transformations, such as dropout

**SBB License:** MIT License

**Core Technology:** Python

**Project URL:** http://blocks.readthedocs.io/en/latest/

**Source Location:** https://github.com/mila-udem/blocks





**ConvNetJS** 
^^^^^^^^^^^^^^^
**SBB Description:** ConvNetJS is a Javascript library for training Deep Learning models (Neural Networks) entirely in your browser. Open a tab and you&#8217;re training. No software requirements, no compilers, no installations, no GPUs, no sweat.
ConvNetJS is a Javascript implementation of Neural networks, together with nice browser-based demos. It currently supports:

Common Neural Network modules (fully connected layers, non-linearities)
Classification (SVM/Softmax) and Regression (L2) cost functions
Ability to specify and train Convolutional Networks that process images
An experimental Reinforcement Learning module, based on Deep Q Learning

For much more information, see the main page at convnetjs.com
Note: Not actively maintained, but still useful to prevent reinventing the wheel.
&#160;

**SBB License:** MIT License

**Core Technology:** Javascript

**Project URL:** https://cs.stanford.edu/people/karpathy/convnetjs/

**Source Location:** https://github.com/karpathy/convnetjs





**Dataexplorer** 
^^^^^^^^^^^^^^^^^^
**SBB Description:** View, visualize, clean and process data in the browser.
Some features:

Classic spreadsheet-style “grid” view
Import CSV data from online
Geocode data (convert “London” to longitude and latitude)
Data and scripts automatically saved and accessible from anywhere
“Fork” support &#8211; build on others work and let them build on yours

**SBB License:** MIT License

**Core Technology:** javascript

**Project URL:** http://explorer.okfnlabs.org

**Source Location:** https://github.com/okfn/dataexplorer





**Datastream** 
^^^^^^^^^^^^^^^^
**SBB Description:** An open-source framework for real-time anomaly detection using Python, ElasticSearch and Kiban. Also uses scikit-learn.

&#160;

**SBB License:** Apache License 2.0

**Core Technology:** Python

**Project URL:** https://github.com/MentatInnovations/datastream.io

**Source Location:** https://github.com/MentatInnovations/datastream.io





**DeepDetect** 
^^^^^^^^^^^^^^^^
**SBB Description:** DeepDetect implements support for supervised and unsupervised deep learning of images, text and other data, with focus on simplicity and ease of use, test and connection into existing applications. It supports classification, object detection, segmentation, regression, autoencoders and more.
It has Python and other client libraries.
Deep Detect has also a REST API for Deep Learning with:

JSON communication format
Pre-trained models
Neural architecture templates
Python, Java, C# clients
Output templating

&#160;

**SBB License:** MIT License

**Core Technology:** C++

**Project URL:** https://deepdetect.com

**Source Location:** https://github.com/beniz/deepdetect





**Deeplearn.js** 
^^^^^^^^^^^^^^^^^^
**SBB Description:** Deeplearn.js is an open-source library that brings performant machine learning building blocks to the web, allowing you to train neural networks in a browser or run pre-trained models in inference mode. And since Google is behind this project, a lot of eyes are targeted on this software. Deeplearn.js is an open source hardware accelerated implementation of deep learning APIs in the browser. So there is no need to download or install anything.
Deeplearn.js was originally developed by the Google Brain PAIR team to build powerful interactive machine learning tools for the browser.

**SBB License:** Apache License 2.0

**Core Technology:** Javascript

**Project URL:** https://deeplearnjs.org/

**Source Location:** https://github.com/PAIR-code/deeplearnjs





**Deeplearning4j** 
^^^^^^^^^^^^^^^^^^^^
**SBB Description:** Deep Learning for Java, Scala &#38; Clojure on Hadoop &#38; Spark With GPUs.
Eclipse Deeplearning4J is an distributed neural net library written in Java and Scala.
Eclipse Deeplearning4j a commercial-grade, open-source, distributed deep-learning library written for Java and Scala. DL4J is designed to be used in business environments on distributed GPUs and CPUs.
Deeplearning4J integrates with Hadoop and Spark and runs on several backends that enable use of CPUs and GPUs. The aim of this project is to create a plug-and-play solution that is more convention than configuration, and which allows for fast prototyping. This project is created by Skymind who delivers support and offers also the option for machine learning models to be hosted with Skymind&#8217;s model server on a cloud environment

**SBB License:** Apache License 2.0

**Core Technology:** Java

**Project URL:** https://deeplearning4j.org

**Source Location:** https://github.com/deeplearning4j/deeplearning4j





**Detectron** 
^^^^^^^^^^^^^^^
**SBB Description:** Detectron is Facebook AI Research&#8217;s software system that implements state-of-the-art object detection algorithms, including Mask R-CNN. It is written in Python and powered by the Caffe2 deep learning framework.
The goal of Detectron is to provide a high-quality, high-performance codebase for object detection research. It is designed to be flexible in order to support rapid implementation and evaluation of novel research.
A number of Facebook teams use this platform to train custom models for a variety of applications including augmented reality and community integrity. Once trained, these models can be deployed in the cloud and on mobile devices, powered by the highly efficient Caffe2 runtime.

**SBB License:** Apache License 2.0

**Core Technology:** Python

**Project URL:** https://github.com/facebookresearch/Detectron

**Source Location:** https://github.com/facebookresearch/Detectron





**Fabrik** 
^^^^^^^^^^^^
**SBB Description:** Fabrik is an online collaborative platform to build, visualize and train deep learning models via a simple drag-and-drop interface. It allows researchers to collaboratively develop and debug models using a web GUI that supports importing, editing and exporting networks written in widely popular frameworks like Caffe, Keras, and TensorFlow.

**SBB License:** GNU General Public License (GPL) 3.0

**Core Technology:** Javascript, Python

**Project URL:** http://fabrik.cloudcv.org/

**Source Location:** https://github.com/Cloud-CV/Fabrik





**Featuretools** 
^^^^^^^^^^^^^^^^^^
**SBB Description:** Featuretools is a python library for automated feature engineering. Featuretools can automatically create a single table of features for any &#8220;target entity&#8221;. Featuretools is a framework to perform automated feature engineering. It excels at transforming transactional and relational datasets into feature matrices for machine learning.

**SBB License:** BSD License 2.0 (3-clause, New or Revised) License

**Core Technology:** Python

**Project URL:** https://www.featuretools.com/

**Source Location:** https://github.com/Featuretools/featuretools





**Fuel** 
^^^^^^^^^^
**SBB Description:** Fuel is a data pipeline framework which provides your machine learning models with the data they need. It is planned to be used by both the Blocks and Pylearn2 neural network libraries.

Fuel allows you to easily read different types of data (NumPy binary files, CSV files, HDF5 files, text files) using a single interface which is based on Python’s iterator types.
Provides a a series of wrappers around frequently used datasets such as MNIST, CIFAR-10 (vision), the One Billion Word Dataset (text corpus), and many more.
Allows you iterate over data in a variety of ways, e.g. in order, shuffled, sampled, etc.
Gives you the possibility to process your data on-the-fly through a series of (chained) transformation procedures. This way you can whiten your data, noise, rotate, crop, pad, sort or shuffle, cache it, and much more.
Is pickle-friendly, allowing you to stop and resume long-running experiments in the middle of a pass over your dataset without losing any training progress.

**SBB License:** MIT License

**Core Technology:** Python

**Project URL:** http://fuel.readthedocs.io/en/latest/index.html

**Source Location:** https://github.com/mila-udem/fuel





**Gensim** 
^^^^^^^^^^^^
**SBB Description:** Gensim is a Python library for topic modelling, document indexing and similarity retrieval with large corpora. Target audience is the natural language processing (NLP) and information retrieval (IR) community.
&#160;

**SBB License:** MIT License

**Core Technology:** Python

**Project URL:** https://github.com/RaRe-Technologies/gensim

**Source Location:** https://github.com/RaRe-Technologies/gensim





**Golem** 
^^^^^^^^^^^
**SBB Description:** The aim of the Golem project is to create a global prosumer market for computing power, in which producers may sell spare CPU time of their personal computers and consumers may acquire resources for computation-intensive tasks. In technical terms, Golem is designed as a decentralised peer-to-peer network established by nodes running the Golem client software. For the purpose of this paper we assume that there are two types of nodes in the Golem network: requestor nodes that announce computing tasks and compute nodes that perform computations (in the actual implementation nodes may switch between both roles).

**SBB License:** GNU General Public License (GPL) 3.0

**Core Technology:** Python

**Project URL:** https://golem.network/

**Source Location:** https://github.com/golemfactory/golem





**HyperTools** 
^^^^^^^^^^^^^^^^
**SBB Description:** HyperTools is a library for visualizing and manipulating high-dimensional data in Python. It is built on top of matplotlib (for plotting), seaborn (for plot styling), and scikit-learn (for data manipulation).
Some key features of HyperTools are:

Functions for plotting high-dimensional datasets in 2/3D
Static and animated plots
Simple API for customizing plot styles
Set of powerful data manipulation tools including hyperalignment, k-means clustering, normalizing and more
Support for lists of Numpy arrays or Pandas dataframes

**SBB License:** MIT License

**Core Technology:** Python

**Project URL:** http://hypertools.readthedocs.io/en/latest/

**Source Location:** https://github.com/ContextLab/hypertools





**JeelizFaceFilter** 
^^^^^^^^^^^^^^^^^^^^^^
**SBB Description:** Javascript/WebGL lightweight face tracking library designed for augmented reality webcam filters. Features : multiple faces detection, rotation, mouth opening. Various integration examples are provided (Three.js, Babylon.js, FaceSwap, Canvas2D, CSS3D&#8230;).
Enables developers to solve computer-vision problems directly from the browser.
Features:

face detection,
face tracking,
face rotation detection,
mouth opening detection,
multiple faces detection and tracking,
very robust for all lighting conditions,
video acquisition with HD video ability,
interfaced with 3D engines like THREE.JS, BABYLON.JS, A-FRAME,
interfaced with more accessible APIs like CANVAS, CSS3D.

**SBB License:** Apache License 2.0

**Core Technology:** Javascript

**Project URL:** https://jeeliz.com/

**Source Location:** https://github.com/jeeliz/jeelizFaceFilter





**Keras** 
^^^^^^^^^^^
**SBB Description:** Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.
Use Keras if you need a deep learning library that:

Allows for easy and fast prototyping (through user friendliness, modularity, and extensibility).
Supports both convolutional networks and recurrent networks, as well as combinations of the two.
Runs seamlessly on CPU and GPU.

**SBB License:** MIT License

**Core Technology:** Python

**Project URL:** https://keras.io/

**Source Location:** https://github.com/keras-team/keras





**Klassify** 
^^^^^^^^^^^^^^
**SBB Description:** Redis based text classification service with real-time web interface.
What is Text Classification: Text classification, document classification or document categorization is a problem in library science, information science and computer science. The task is to assign a document to one or more classes or categories.

**SBB License:** MIT License

**Core Technology:** Python

**Project URL:** https://github.com/fatiherikli/klassify

**Source Location:** https://github.com/fatiherikli/klassify





**Luminoth** 
^^^^^^^^^^^^^^
**SBB Description:** Luminoth is an open source toolkit for computer vision. Currently, we support object detection and image classification, but we are aiming for much more. It is built in Python, using TensorFlow and Sonnet.
&#160;

**SBB License:** BSD License 2.0 (3-clause, New or Revised) License

**Core Technology:** Python

**Project URL:** https://luminoth.ai

**Source Location:** https://github.com/tryolabs/luminoth





**MacroBase** 
^^^^^^^^^^^^^^^
**SBB Description:** MacroBase is a new analytic monitoring engine designed to prioritize human attention in large-scale datasets and data streams. Unlike a traditional analytics engine, MacroBase is specialized for one task: finding and explaining unusual or interesting trends in data. Developed by Stanford Future Data Systems
Documentation can be found at: https://macrobase.stanford.edu/docs/

**SBB License:** Apache License 2.0

**Core Technology:** Java

**Project URL:** https://macrobase.stanford.edu/

**Source Location:** https://github.com/stanford-futuredata/macrobase/tree/v1.0





**ml5.js** 
^^^^^^^^^^^^
**SBB Description:** ml5.js aims to make machine learning approachable for a broad audience of artists, creative coders, and students. The library provides access to machine learning algorithms and models in the browser, building on top of TensorFlow.js with no other external dependencies.
The library is supported by code examples, tutorials, and sample data sets with an emphasis on ethical computing. Bias in data, stereotypical harms, and responsible crowdsourcing are part of the documentation around data collection and usage.
ml5.js is heavily inspired by Processing and p5.js.

**SBB License:** MIT License

**Core Technology:** Javascript

**Project URL:** https://ml5js.org/

**Source Location:** https://github.com/ml5js/ml5-library





**MLflow** 
^^^^^^^^^^^^
**SBB Description:** MLflow (currently in alpha) is an open source platform designed to manage the entire machine learning lifecycle and work with any machine learning library. It offers:

Record and query experiments: code, data, config, results
Packaging format for reproducible runs on any platform
General format for sending models to diverse deploy tools

**SBB License:** Apache License 2.0

**Core Technology:** Python

**Project URL:** https://mlflow.org/

**Source Location:** https://github.com/databricks/mlflow





**MLPerf** 
^^^^^^^^^^^^
**SBB Description:** A broad ML benchmark suite for measuring performance of ML software frameworks, ML hardware accelerators, and ML cloud platforms.
The MLPerf effort aims to build a common set of benchmarks that enables the machine learning (ML) field to measure system performance for both training and inference from mobile devices to cloud services. We believe that a widely accepted benchmark suite will benefit the entire community, including researchers, developers, builders of machine learning frameworks, cloud service providers, hardware manufacturers, application providers, and end users.

**SBB License:** MIT License

**Core Technology:** Python

**Project URL:** https://mlperf.org/

**Source Location:** https://github.com/mlperf/reference





**ModelDB** 
^^^^^^^^^^^^^
**SBB Description:** A system to manage machine learning models.
ModelDB is an end-to-end system to manage machine learning models. It ingests models and associated metadata as models are being trained, stores model data in a structured format, and surfaces it through a web-frontend for rich querying. ModelDB can be used with any ML environment via the ModelDB Light API. ModelDB native clients can be used for advanced support in spark.ml and scikit-learn.
The ModelDB frontend provides rich summaries and graphs showing model data. The frontend provides functionality to slice and dice this data along various attributes (e.g. operations like filter by hyperparameter, group by datasets) and to build custom charts showing model performance.

**SBB License:** MIT License

**Core Technology:** Python, Javascript

**Project URL:** https://mitdbg.github.io/modeldb/

**Source Location:** https://github.com/mitdbg/modeldb





**Netron** 
^^^^^^^^^^^^
**SBB Description:** Netron is a viewer for neural network, deep learning and machine learning models.
Netron supports ONNX (.onnx, .pb), Keras (.h5, .keras), CoreML (.mlmodel) and TensorFlow Lite (.tflite). Netron has experimental support for Caffe (.caffemodel), Caffe2 (predict_net.pb), MXNet (-symbol.json), TensorFlow.js (model.json, .pb) and TensorFlow (.pb, .meta).

**SBB License:** GNU General Public License (GPL) 2.0

**Core Technology:** Python, Javascript

**Project URL:** https://www.lutzroeder.com/ai/

**Source Location:** https://github.com/lutzroeder/Netron





**Neuralcoref** 
^^^^^^^^^^^^^^^^^
**SBB Description:** State-of-the-art coreference resolution based on neural nets and spaCy.
NeuralCoref is a pipeline extension for spaCy 2.0 that annotates and resolves coreference clusters using a neural network. NeuralCoref is production-ready, integrated in spaCy&#8217;s NLP pipeline and easily extensible to new training datasets.

**SBB License:** MIT License

**Core Technology:** Python

**Project URL:** https://huggingface.co/coref/

**Source Location:** https://github.com/huggingface/neuralcoref





**NLP Architect** 
^^^^^^^^^^^^^^^^^^^
**SBB Description:** NLP Architect is an open-source Python library for exploring the state-of-the-art deep learning topologies and techniques for natural language processing and natural language understanding. It is intended to be a platform for future research and collaboration.

How can NLP Architect be used:

Train models using provided algorithms, reference datasets and configurations
Train models using your own data
Create new/extend models based on existing models or topologies
Explore how deep learning models tackle various NLP tasks
Experiment and optimize state-of-the-art deep learning algorithms
integrate modules and utilities from the library to solutions

**SBB License:** Apache License 2.0

**Core Technology:** Python

**Project URL:** http://nlp_architect.nervanasys.com/

**Source Location:** https://github.com/NervanaSystems/nlp-architect





**ONNX** 
^^^^^^^^^^
**SBB Description:** ONNX provides an open source format for AI models. It defines an extensible computation graph model, as well as definitions of built-in operators and standard data types. Initially we focus on the capabilities needed for inferencing (evaluation).
Caffe2, PyTorch, Microsoft Cognitive Toolkit, Apache MXNet and other tools are developing ONNX support. Enabling interoperability between different frameworks and streamlining the path from research to production will increase the speed of innovation in the AI community. We are an early stage and we invite the community to submit feedback and help us further evolve ONNX.
Companies behind ONNX are AWS, Facebook and Microsoft Corporation and more.

**SBB License:** MIT License

**Core Technology:** Python

**Project URL:** http://onnx.ai/

**Source Location:** https://github.com/onnx/onnx





**OpenCV: Open Source Computer Vision Library** 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**SBB Description:** OpenCV (Open Source Computer Vision Library) is an open source computer vision and machine learning software library. OpenCV was built to provide a common infrastructure for computer vision applications and to accelerate the use of machine perception in the commercial products. Being a BSD-licensed product, OpenCV makes it easy for businesses to utilize and modify the code.
The library has more than 2500 optimized algorithms, which includes a comprehensive set of both classic and state-of-the-art computer vision and machine learning algorithms. These algorithms can be used to detect and recognize faces, identify objects, classify human actions in videos, track camera movements, track moving objects, extract 3D models of objects, produce 3D point clouds from stereo cameras, stitch images together to produce a high resolution image of an entire scene, find similar images from an image database, remove red eyes from images taken using flash, follow eye movements, recognize scenery and establish markers to overlay it with augmented reality, etc.

**SBB License:** BSD License 2.0 (3-clause, New or Revised) License

**Core Technology:** C

**Project URL:** https://opencv.org/

**Source Location:** https://github.com/opencv/opencv





**OpenML** 
^^^^^^^^^^^^
**SBB Description:** OpenML is an on-line machine learning platform for sharing and organizing data, machine learning algorithms and experiments. It claims to be designed to create a frictionless, networked ecosystem, so that you can readily integrate into your existing processes/code/environments. It also allows people from all over the world to collaborate and build directly on each other’s latest ideas, data and results, irrespective of the tools and infrastructure they happen to use. So nice ideas to build an open science movement. The people behind OpemML are mostly (data)scientist. So using this product for real world business use cases will take some extra effort.
Altrhough OpenML is exposed as an foundation based on openness, a quick inspection learned that the OpenML platform  is not as open as you want. Also the OSS software is not created to be run on premise. So be aware when doing large (time) investments into this OpenML platform.

**SBB License:** BSD License 2.0 (3-clause, New or Revised) License

**Core Technology:** Java

**Project URL:** https://openml.org

**Source Location:** https://github.com/openml/OpenML





**Orange** 
^^^^^^^^^^^^
**SBB Description:** Orange is a comprehensive, component-based software suite for machine learning and data mining, developed at Bioinformatics Laboratory.
Orange is available by default on Anaconda Navigator dashboard. Orange is a component-based data mining software. It includes a range of data visualization, exploration, preprocessing and modeling techniques. It can be used through a nice and intuitive user interface or, for more advanced users, as a module for the Python programming language.
One of the nice features is the option for visual programming. Can you do visual interactive data exploration for rapid qualitative analysis with clean visualizations. The graphic user interface allows you to focus on exploratory data analysis instead of coding, while clever defaults make fast prototyping of a data analysis workflow extremely easy.
&#160;
&#160;

**SBB License:** GNU General Public License (GPL) 3.0

**Core Technology:** 

**Project URL:** https://orange.biolab.si/

**Source Location:** https://github.com/biolab/orange3





**Pattern** 
^^^^^^^^^^^^^
**SBB Description:** Pattern is a web mining module for Python. It has tools for:

Data Mining: web services (Google, Twitter, Wikipedia), web crawler, HTML DOM parser
Natural Language Processing: part-of-speech taggers, n-gram search, sentiment analysis, WordNet
Machine Learning: vector space model, clustering, classification (KNN, SVM, Perceptron)
Network Analysis: graph centrality and visualization.

**SBB License:** BSD License 2.0 (3-clause, New or Revised) License

**Core Technology:** Python

**Project URL:** https://www.clips.uantwerpen.be/pages/pattern

**Source Location:** https://github.com/clips/pattern





**Plait** 
^^^^^^^^^^^
**SBB Description:** plait.py is a program for generating fake data from composable yaml templates.
With plait it is easy to model fake data that has an interesting shape. Currently, many fake data generators model their data as a collection of IID variables; with plait.py we can stitch together those variables into a more coherent model.
Example uses for plait.py are:

generating mock application data in test environments
validating the usefulness of statistical techniques
creating synthetic datasets for performance tuning databases

**SBB License:** MIT License

**Core Technology:** Python

**Project URL:** https://github.com/plaitpy/plaitpy

**Source Location:** https://github.com/plaitpy/plaitpy





**Polyaxon** 
^^^^^^^^^^^^^^
**SBB Description:** An open source platform for reproducible machine learning at scale.
Polyaxon is a platform for building, training, and monitoring large scale deep learning applications.
Polyaxon deploys into any data center, cloud provider, or can be hosted and managed by Polyaxon, and it supports all the major deep learning frameworks such as Tensorflow, MXNet, Caffe, Torch, etc.
Polyaxon makes it faster, easier, and more efficient to develop deep learning applications by managing workloads with smart container and node management. And it turns GPU servers into shared, self-service resources for your team or organization.

**SBB License:** MIT License

**Core Technology:** Python

**Project URL:** https://polyaxon.com/

**Source Location:** https://github.com/polyaxon/polyaxon





**Pylearn2** 
^^^^^^^^^^^^^^
**SBB Description:** Pylearn2 is a library designed to make machine learning research easy.

**SBB License:** BSD License 2.0 (3-clause, New or Revised) License

**Core Technology:** Python

**Project URL:** http://deeplearning.net/software/pylearn2/

**Source Location:** https://github.com/lisa-lab/pylearn2





**Pyro** 
^^^^^^^^^^
**SBB Description:** Deep universal probabilistic programming with Python and PyTorch. Pyro is in an alpha release. It is developed and used by Uber AI Labs.

&#160;

**SBB License:** GNU General Public License (GPL) 2.0

**Core Technology:** Python

**Project URL:** http://pyro.ai/

**Source Location:** https://github.com/uber/pyro





**PyTorch** 
^^^^^^^^^^^^^
**SBB Description:** PyTorch is:

a deep learning framework that puts Python first.
 a research-focused framework.
Python package that provides two high-level features:

Pytorch uses tensor computation (like NumPy) with strong GPU acceleration. It can use deep neural networks built on a tape-based autograd system.
You can reuse your favorite Python packages such as NumPy, SciPy and Cython to extend PyTorch when needed.
Note: PyTorch is still in an early-release beta phase (status January 2018). PyTorch was released as OSS by Google January 2017.

**SBB License:** MIT License

**Core Technology:** Python

**Project URL:** http://pytorch.org/

**Source Location:** https://github.com/pytorch/pytorch





**Ray** 
^^^^^^^^^
**SBB Description:** Ray is a flexible, high-performance distributed execution framework for AI applications. Ray is currently under heavy development. But Ray has already a good start, with good documentation (http://ray.readthedocs.io/en/latest/index.html) and a tutorial. Also Ray is backed by scientific researchers and published papers.
Ray comes with libraries that accelerate deep learning and reinforcement learning development:

Ray Tune: Hyperparameter Optimization Framework
Ray RLlib: A Scalable Reinforcement Learning Library

**SBB License:** Apache License 2.0

**Core Technology:** Python

**Project URL:** https://ray-project.github.io/

**Source Location:** https://github.com/ray-project/ray





**Scikit-learn** 
^^^^^^^^^^^^^^^^^^
**SBB Description:** scikit-learn is a Python module for machine learning.
Simple and efficient tools for data mining and data analysis

Accessible to everybody, and reusable in various contexts
Built on NumPy, SciPy, and matplotlib

**SBB License:** BSD License 2.0 (3-clause, New or Revised) License

**Core Technology:** Python

**Project URL:** http://scikit-learn.org

**Source Location:** https://github.com/scikit-learn/scikit-learn





**Skater** 
^^^^^^^^^^^^
**SBB Description:** Skater is a python package for model agnostic interpretation of predictive models. With Skater, you can unpack the internal mechanics of arbitrary models; as long as you can obtain inputs, and use a function to obtain outputs, you can use Skater to learn about the models internal decision policies.
The project was started as a research idea to find ways to enable better interpretability(preferably human interpretability) to predictive &#8220;black boxes&#8221; both for researchers and practioners.
Documentation at: https://datascienceinc.github.io/Skater/overview.html

**SBB License:** MIT License

**Core Technology:** Python

**Project URL:** https://www.datascience.com/resources/tools/skater

**Source Location:** https://github.com/datascienceinc/Skater





**Snorkel** 
^^^^^^^^^^^^^
**SBB Description:** Snorkel is a system for rapidly creating, modeling, and managing training data, currently focused on accelerating the development of structured or &#8220;dark&#8221; data extraction applications for domains in which large labeled training sets are not available or easy to obtain.

**SBB License:** Apache License 2.0

**Core Technology:** Python

**Project URL:** https://hazyresearch.github.io/snorkel/

**Source Location:** https://github.com/HazyResearch/snorkel





**Tensorflow** 
^^^^^^^^^^^^^^^^
**SBB Description:** TensorFlow is an Open Source Software Library for Machine Intelligence. TensorFlow is by far the most used and popular ML open source project. And since the first initial release was only just in November 2015 it is expected that the impact of this OSS package will expand even more.
TensorFlow™ is an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them. The flexible architecture allows you to deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device with a single API. TensorFlow was originally developed by researchers and engineers working on the Google Brain Team within Google&#8217;s Machine Intelligence research organization for the purposes of conducting machine learning and deep neural networks research, but the system is general enough to be applicable in a wide variety of other domains as well.
TensorFlow comes with a tool called TensorBoard which you can use to get some insight into what is happening. TensorBoard is a suite of web applications for inspecting and understanding your TensorFlow runs and graphs.
There is also a version of TensorFlow that runs in a browser. This is TensorFlow.js (https://js.tensorflow.org/ ). TensorFlow.js is a WebGL accelerated, browser based JavaScript library for training and deploying ML models.
&#160;

**SBB License:** Apache License 2.0

**Core Technology:** C

**Project URL:** https://www.tensorflow.org/

**Source Location:** https://github.com/tensorflow/tensorflow





**TextBlob: Simplified Text Processing** 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**SBB Description:** TextBlob is a Python (2 and 3) library for processing textual data. It provides a simple API for diving into common natural language processing (NLP) tasks such as part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, translation, and more.
Features

Noun phrase extraction
Part-of-speech tagging
Sentiment analysis
Classification (Naive Bayes, Decision Tree)
Language translation and detection powered by Google Translate
Tokenization (splitting text into words and sentences)
Word and phrase frequencies
Parsing
n-grams
Word inflection (pluralization and singularization) and lemmatization
Spelling correction
Add new models or languages through extensions
WordNet integration

**SBB License:** MIT License

**Core Technology:** Python

**Project URL:** https://textblob.readthedocs.io/en/dev/

**Source Location:** https://github.com/sloria/textblob





**Theano** 
^^^^^^^^^^^^
**SBB Description:** Theano is a Python library that allows you to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays efficiently. It can use GPUs and perform efficient symbolic differentiation.
Note: After almost ten years of development the company behind Theano has stopped development and support(Q4-2017). But this library has been an innovation driver for many other OSS ML packages!
Since a lot of ML libraries and packages use Theano you should check (as always) the health of your ML stack.

**SBB License:** MIT License

**Core Technology:** Python

**Project URL:** http://www.deeplearning.net/

**Source Location:** https://github.com/Theano/Theano





**Thinc** 
^^^^^^^^^^^
**SBB Description:** Thinc is the machine learning library powering spaCy. It features a battle-tested linear model designed for large sparse learning problems, and a flexible neural network model under development for spaCy v2.0.
Thinc is a practical toolkit for implementing models that follow the &#8220;Embed, encode, attend, predict&#8221; architecture. It&#8217;s designed to be easy to install, efficient for CPU usage and optimised for NLP and deep learning with text – in particular, hierarchically structured input and variable-length sequences.

**SBB License:** GNU General Public License (GPL) 2.0

**Core Technology:** Python

**Project URL:** https://explosion.ai/

**Source Location:** https://github.com/explosion/thinc





**Turi** 
^^^^^^^^^^
**SBB Description:** Turi Create simplifies the development of custom machine learning models. Turi is OSS machine learning from Apple.
Turi Create simplifies the development of custom machine learning models. You don&#8217;t have to be a machine learning expert to add recommendations, object detection, image classification, image similarity or activity classification to your app.

**SBB License:** BSD License 2.0 (3-clause, New or Revised) License

**Core Technology:** Python

**Project URL:** https://github.com/apple/turicreate

**Source Location:** https://github.com/apple/turicreate





**TuriCreate** 
^^^^^^^^^^^^^^^^
**SBB Description:** This SBB is from Apple. Apple, is with Siri already for a long time active in machine learning. But even Apple is releasing building blocks under OSS licenses now.
Turi Create simplifies the development of custom machine learning models. You don&#8217;t have to be a machine learning expert to add recommendations, object detection, image classification, image similarity or activity classification to your app.

Easy-to-use: Focus on tasks instead of algorithms
Visual: Built-in, streaming visualizations to explore your data
Flexible: Supports text, images, audio, video and sensor data
Fast and Scalable: Work with large datasets on a single machine
Ready To Deploy: Export models to Core ML for use in iOS, macOS, watchOS, and tvOS apps

**SBB License:** BSD License 2.0 (3-clause, New or Revised) License

**Core Technology:** Python

**Project URL:** https://turi.com/index.html

**Source Location:** https://github.com/apple/turicreate





**VisualDL** 
^^^^^^^^^^^^^^
**SBB Description:** VisualDL is an open-source cross-framework web dashboard that richly visualizes the performance and data flowing through your neural network training. VisualDL is a deep learning visualization tool that can help design deep learning jobs. It includes features such as scalar, parameter distribution, model structure and image visualization.

**SBB License:** Apache License 2.0

**Core Technology:** C++

**Project URL:** http://visualdl.paddlepaddle.org/

**Source Location:** https://github.com/PaddlePaddle/VisualDL





End of SBB list <br>