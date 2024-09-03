# ML Frameworks

## Acme

<p>Acme is a library of reinforcement learning (RL) agents and agent building blocks. Acme strives to expose simple, efficient, and readable agents, that serve both as reference implementations of popular algorithms and as strong baselines, while still providing enough flexibility to do novel research. The design of Acme also attempts to provide multiple points of entry to the RL problem at differing levels of complexity.</p>



<p>Overall Acme strives to expose simple, efficient, and readable agent baselines while still providing enough flexibility to create novel implementations.</p>

Item | Value 
----- | -----
**SBB License** | Apache License 2.0
**Core Technology** | Python
**Project URL** | [https://github.com/deepmind/acme](https://github.com/deepmind/acme)
**Source Location** | [https://github.com/deepmind/acme](https://github.com/deepmind/acme)
*Tag(s)* |ML Framework


## AdaNet

<p>AdaNet is a lightweight TensorFlow-based framework  for automatically learning high-quality models with minimal expert  intervention. AdaNet builds on recent AutoML efforts to be fast and  flexible while providing learning guarantees. Importantly, AdaNet  provides a general framework for not only learning a neural network  architecture, but also for learning to ensemble to obtain even better  models.</p>



<p>This project is based on the <em>AdaNet algorithm</em>, presented in “<a href="http://proceedings.mlr.press/v70/cortes17a.html">AdaNet: Adaptive Structural Learning of Artificial Neural Networks</a>” at <a href="https://icml.cc/Conferences/2017">ICML 2017</a>, for learning the structure of a neural network as an ensemble of subnetworks.</p>



<p>AdaNet has the following goals:</p>



<ul><li><em>Ease of use</em>: Provide familiar APIs (e.g. Keras, Estimator) for training, evaluating, and serving models.</li><li><em>Speed</em>: Scale with available compute and quickly produce high quality models.</li><li><em>Flexibility</em>: Allow researchers and practitioners to extend AdaNet to novel subnetwork architectures, search spaces, and tasks.</li><li><em>Learning guarantees</em>: Optimize an objective that offers theoretical learning guarantees.</li></ul>



<p>Documentation at <a href="https://adanet.readthedocs.io/en/latest/">https://adanet.readthedocs.io/en/latest/ </a></p>

Item | Value 
----- | -----
**SBB License** | Apache License 2.0
**Core Technology** | Python
**Project URL** | [https://adanet.readthedocs.io/en/latest/](https://adanet.readthedocs.io/en/latest/)
**Source Location** | [https://github.com/tensorflow/adanet](https://github.com/tensorflow/adanet)
*Tag(s)* |ML, ML Framework


## Analytics Zoo

<p>Analytics Zoo provides a unified analytics + AI platform that seamlessly unites <em>Spark, TensorFlow, Keras and BigDL</em>  programs into an integrated pipeline; the entire pipeline can then  transparently scale out to a large Hadoop/Spark cluster for distributed  training or inference.</p>



<ul><li><em>Data wrangling and analysis using PySpark</em></li><li><em>Deep learning model development using TensorFlow or Keras</em></li><li><em>Distributed training/inference on Spark and BigDL</em></li><li><em>All within a single unified pipeline and in a user-transparent fashion!</em></li></ul>

Item | Value 
----- | -----
**SBB License** | Apache License 2.0
**Core Technology** | Python
**Project URL** | [https://analytics-zoo.github.io/master/](https://analytics-zoo.github.io/master/)
**Source Location** | [https://github.com/intel-analytics/analytics-zoo](https://github.com/intel-analytics/analytics-zoo)
*Tag(s)* |ML, ML Framework, Python


## Apache MXNet

<p><span class="col-11 text-gray-dark mr-2">Lightweight, Portable, Flexible Distributed/Mobile Deep Learning with Dynamic, Mutation-aware Dataflow Dep Scheduler; for Python, R, Julia, Scala, Go, Javascript and more.</span></p>



<p>All major GPU and CPU vendors support this project, but also the real giants like Amazon, Microsoft, Wolfram and a number of very respected universities. So watch this project or play with it to see if it fits your use case.</p>



<p>Apache MXNet (incubating) is a deep learning framework designed for both <em>efficiency</em> and <em>flexibility</em>. It allows you to <em><strong>mix</strong></em> <a href="https://mxnet.incubator.apache.org/architecture/index.html#deep-learning-system-design-concepts" rel="nofollow">symbolic and imperative programming</a> to <em><strong>maximize</strong></em> efficiency and productivity. At its core, MXNet contains a dynamic dependency scheduler that automatically parallelizes both symbolic and imperative operations on the fly. A graph optimization layer on top of that makes symbolic execution fast and memory efficient. MXNet is portable and lightweight, scaling effectively to multiple GPUs and multiple machines.</p>



<p>MXNet is also more than a deep learning project. It is also a collection of <a href="https://mxnet.incubator.apache.org/architecture/index.html#deep-learning-system-design-concepts" rel="nofollow">blue prints and guidelines</a> for building deep learning systems, and interesting insights of DL systems for hackers.</p>



<p>Gluon is the high-level interface for MXNet. It is more intuitive and easier to use than the lower level interface. Gluon supports dynamic (define-by-run) graphs with JIT-compilation to achieve both flexibility and efficiency. The perfect starters documentation with a great crash course on deep learning can be found here: <a href="https://d2l.ai/index.html">https://d2l.ai/index.html</a>&#160; An earlier version of this documentation is still available on:<a href="http://gluon.mxnet.io/" target="_blank" rel="noopener noreferrer">&#160; http://gluon.mxnet.io/</a></p>



<p>Part of the project is also the the Gluon API specification (see <a href="https://github.com/gluon-api/gluon-api" target="_blank" rel="noopener noreferrer">https://github.com/gluon-api/gluon-api</a>)</p>



<p>The Gluon API specification (Python based) is an effort to improve speed, flexibility, and accessibility of deep learning technology for all developers, regardless of their deep learning framework of choice. The Gluon API offers a flexible interface that simplifies the process of prototyping, building, and training deep learning models without sacrificing training speed.</p>

Item | Value 
----- | -----
**SBB License** | Apache License 2.0
**Core Technology** | CPP
**Project URL** | [https://mxnet.apache.org/](https://mxnet.apache.org/)
**Source Location** | [https://github.com/apache/incubator-mxnet](https://github.com/apache/incubator-mxnet)
*Tag(s)* |ML, ML Framework


## Apache Spark MLlib

<p>Apache Spark MLlib. MLlib is Apache Spark&#8217;s scalable machine learning library. MLlib is a Spark subproject providing machine learning primitives. MLlib is a standard component of Spark providing machine learning primitives on top of Spark platform.</p>



<p>Apache Spark is a FOSS platform for large-scale data processing. The Spark engine is written in Scala and is well suited for applications that reuse a working set of data across multiple parallel operations. It’s designed to work as a standalone cluster or as part of Hadoop YARN cluster. It can access data from sources such as HDFS, Cassandra or Amazon S3. </p>



<p>MLlib can be seen as a core Spark&#8217;s APIs and interoperates with NumPy in Python and R libraries. And Spark is very fast! MLlib  ships with Spark as a standard component.</p>



<p>MLlib library contains many algorithms and utilities, e.g.:</p>



<ul><li>Classification: logistic regression, naive Bayes.</li><li>Regression: generalized linear regression, survival regression.</li><li>Decision trees, random forests, and gradient-boosted trees.</li><li>Recommendation: alternating least squares (ALS).</li><li>Clustering: K-means, Gaussian mixtures (GMMs).</li><li>Topic modeling: latent Dirichlet allocation (LDA).</li><li>Frequent item sets, association rules, and sequential pattern mining.</li></ul>



<p>Using Spark MLlib gives the following advantages:</p>



<ul><li>Excellent scalability options</li><li>Performance </li><li>User-friendly APIs </li><li>Integration with Spark and its other components</li></ul>



<p>But using Spark means that also the Spark platform must be used.  </p>

Item | Value 
----- | -----
**SBB License** | Apache License 2.0
**Core Technology** | Java
**Project URL** | [https://spark.apache.org/mllib/](https://spark.apache.org/mllib/)
**Source Location** | [https://github.com/apache/spark](https://github.com/apache/spark)
*Tag(s)* |ML, ML Framework


## auto_ml

<p><span class="col-11 text-gray-dark mr-2">Automated machine learning for analytics &#38; production.</span></p>



<p>Automates the whole machine learning process, making it super easy to use for both analytics, and getting real-time predictions in production.</p>



<p>Unfortunate unmaintained currently, but still worth playing with.</p>

Item | Value 
----- | -----
**SBB License** | MIT License
**Core Technology** | Python
**Project URL** | [http://auto-ml.readthedocs.io](http://auto-ml.readthedocs.io)
**Source Location** | [https://github.com/ClimbsRocks/auto_ml](https://github.com/ClimbsRocks/auto_ml)
*Tag(s)* |ML, ML Framework


## BigDL

<p>BigDL is a distributed deep learning library for Apache Spark; with BigDL, users can write their deep learning applications as standard Spark programs, which can directly run on top of existing Spark or Hadoop clusters.</p>



<ul><li><strong>Rich deep learning support.</strong> Modeled after <a href="http://torch.ch/" data-wm-adjusted="done">Torch</a>, BigDL provides comprehensive support for deep learning, including numeric computing (via <a href="https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/tensor" data-wm-adjusted="done">Tensor</a>) and high level <a href="https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/nn" data-wm-adjusted="done">neural networks</a>; in addition, users can load pre-trained <a href="http://caffe.berkeleyvision.org/" data-wm-adjusted="done">Caffe</a> or <a href="http://torch.ch/" data-wm-adjusted="done">Torch</a> or <a href="https://faroit.github.io/keras-docs/1.2.2/" data-wm-adjusted="done">Keras</a> models into Spark programs using BigDL.</li><li><strong>Extremely high performance.</strong> To achieve high performance, BigDL uses <a href="https://software.intel.com/en-us/intel-mkl" data-wm-adjusted="done">Intel MKL</a> and multi-threaded programming in each Spark task. Consequently, it is orders of magnitude faster than out-of-box open source <a href="http://caffe.berkeleyvision.org/" data-wm-adjusted="done">Caffe</a>, <a href="http://torch.ch/" data-wm-adjusted="done">Torch</a> or <a href="https://www.tensorflow.org/" data-wm-adjusted="done">TensorFlow</a> on a single-node Xeon (i.e., comparable with mainstream GPU).</li><li><strong>Efficiently scale-out.</strong> BigDL can efficiently scale out to perform data analytics at &#8220;Big Data scale&#8221;, by leveraging <a href="http://spark.apache.org/" data-wm-adjusted="done">Apache Spark</a> (a lightning fast distributed data processing framework), as well as efficient implementations of synchronous SGD and all-reduce communications on Spark.</li></ul>

Item | Value 
----- | -----
**SBB License** | Apache License 2.0
**Core Technology** | Java
**Project URL** | [https://bigdl-project.github.io/master/](https://bigdl-project.github.io/master/)
**Source Location** | [https://github.com/intel-analytics/BigDL](https://github.com/intel-analytics/BigDL)
*Tag(s)* |ML, ML Framework


## Blocks

<p>Blocks is a framework that is supposed to make it easier to build complicated neural network models on top of <a class="reference external" href="http://www.deeplearning.net/software/theano/">Theano</a>.</p>



<p>Blocks is a framework that helps you build neural network models on top of Theano. Currently it supports and provides:</p>



<ul><li>Constructing parametrized Theano operations, called &#8220;bricks&#8221;</li><li>Pattern matching to select variables and bricks in large models</li><li>Algorithms to optimize your model</li><li>Saving and resuming of training</li><li>Monitoring and analyzing values during training progress (on the training set as well as on test sets)</li><li>Application of graph transformations, such as dropout</li></ul>

Item | Value 
----- | -----
**SBB License** | MIT License
**Core Technology** | Python
**Project URL** | [http://blocks.readthedocs.io/en/latest/](http://blocks.readthedocs.io/en/latest/)
**Source Location** | [https://github.com/mila-udem/blocks](https://github.com/mila-udem/blocks)
*Tag(s)* |ML, ML Framework


## Caffe

<p>Caffe is a deep learning framework made with expression, speed, and modularity in mind. It is developed by Berkeley AI Research (<a href="http://bair.berkeley.edu">BAIR</a>)/The Berkeley Vision and Learning Center (BVLC) and community contributors.</p>



<p>Caffe is an Open framework, models, and worked examples for deep learning:</p>



<ul><li>4.5 years old</li><li>7,000+ citations, 250+ contributors, 24,000+ stars</li><li>15,000+ forks, &#62;1 pull request / day average at peak</li></ul>



<p>Focus has been vision, but also handles , reinforcement learning, speech and text.</p>



<p>Why Caffe?</p>



<ul><li><strong>Expressive architecture</strong> encourages application and  innovation. Models and optimization are defined by configuration without  hard-coding. Switch between CPU and GPU by setting a single flag to train on a GPU  machine then deploy to commodity clusters or mobile devices.</li><li><strong>Extensible code</strong> fosters active development. In Caffe’s first year, it has been forked by over 1,000 developers and had many significant changes contributed back. Thanks to these contributors the framework tracks the state-of-the-art in both code and models.</li><li><strong>Speed</strong> makes Caffe perfect for research experiments and industry deployment. Caffe can process <strong>over 60M images per day</strong> with a single NVIDIA K40 GPU*. That’s 1 ms/image for inference and 4 ms/image for learning and more recent library versions and hardware are faster still. We believe that Caffe is among the fastest convnet implementations available.</li></ul>

Item | Value 
----- | -----
**SBB License** | BSD License 2.0 (3-clause, New or Revised) License
**Core Technology** | CPP
**Project URL** | [http://caffe.berkeleyvision.org/](http://caffe.berkeleyvision.org/)
**Source Location** | [https://github.com/BVLC/caffe](https://github.com/BVLC/caffe)
*Tag(s)* |ML, ML Framework


## ConvNetJS

<p>ConvNetJS is a Javascript library for training Deep Learning models (Neural Networks) entirely in your browser. Open a tab and you&#8217;re training. No software requirements, no compilers, no installations, no GPUs, no sweat.</p>



<p>ConvNetJS is a Javascript implementation of Neural networks, together with nice browser-based demos. It currently supports:</p>



<ul><li>Common <strong>Neural Network modules</strong> (fully connected layers, non-linearities)</li><li>Classification (SVM/Softmax) and Regression (L2) <strong>cost functions</strong></li><li>Ability to specify and train <strong>Convolutional Networks</strong> that process images</li><li>An experimental <strong>Reinforcement Learning</strong> module, based on Deep Q Learning</li></ul>



<p>For much more information, see the main page at <a href="http://convnetjs.com" rel="nofollow">convnetjs.com</a></p>



<p>Note: Not actively maintained, but still useful to prevent reinventing the wheel.</p>

Item | Value 
----- | -----
**SBB License** | MIT License
**Core Technology** | Javascript
**Project URL** | [https://cs.stanford.edu/people/karpathy/convnetjs/](https://cs.stanford.edu/people/karpathy/convnetjs/)
**Source Location** | [https://github.com/karpathy/convnetjs](https://github.com/karpathy/convnetjs)
*Tag(s)* |Javascript, ML, ML Framework


## Datumbox

<p>The Datumbox Machine Learning Framework is an open-source framework written in Java which allows the rapid development Machine Learning and Statistical applications. The main focus of the framework is to include a large number of machine learning algorithms &#38; statistical methods and to be able to handle large sized datasets.</p>



<p>Datumbox comes with a large number of pre-trained models which allow you to perform Sentiment Analysis (Document &#38; Twitter), Subjectivity Analysis, Topic Classification, Spam Detection, Adult Content Detection, Language Detection, Commercial Detection, Educational Detection and Gender Detection. </p>



<p>Datumbox is not supported by a large team of commercial developers or large group of FOSS developers.  Basically one developer maintains it as a side project. So review this FOSS project before you make large investments building applications on top of it.</p>



<p></p>

Item | Value 
----- | -----
**SBB License** | Apache License 2.0
**Core Technology** | Java
**Project URL** | [http://www.datumbox.com/](http://www.datumbox.com/)
**Source Location** | [https://github.com/datumbox/datumbox-framework](https://github.com/datumbox/datumbox-framework)
*Tag(s)* |ML, ML Framework


## DeepDetect

<p>DeepDetect implements support for supervised and unsupervised deep learning of images, text and other data, with focus on simplicity and ease of use, test and connection into existing applications. It supports classification, object detection, segmentation, regression, autoencoders and more.</p>



<p>It has Python and other client libraries.</p>



<p>Deep Detect has also a REST API for Deep Learning with:</p>



<ul><li>JSON communication format</li><li>Pre-trained models</li><li>Neural architecture templates</li><li>Python, Java, C# clients</li><li>Output templating</li></ul>

Item | Value 
----- | -----
**SBB License** | MIT License
**Core Technology** | C++
**Project URL** | [https://deepdetect.com](https://deepdetect.com)
**Source Location** | [https://github.com/beniz/deepdetect](https://github.com/beniz/deepdetect)
*Tag(s)* |ML, ML Framework


## Deeplearning4j

<p><span class="col-11 text-gray-dark mr-2">Deep Learning for Java, Scala &#38; Clojure on Hadoop &#38; Spark With GPUs</span>.</p>



<p>Eclipse Deeplearning4J is an distributed neural net library written in Java and Scala.</p>



<p>Eclipse Deeplearning4j a commercial-grade, open-source, distributed deep-learning library written for Java and Scala. DL4J is designed to be used in business environments on distributed GPUs and CPUs.</p>



<p>Deeplearning4J integrates with Hadoop and Spark and runs on several backends that enable use of CPUs and GPUs. The aim of this project is to create a plug-and-play solution that is more convention than configuration, and which allows for fast prototyping. This project is created by Skymind who delivers support and offers also the option for machine learning models to be hosted with Skymind&#8217;s model server on a cloud environment</p>

Item | Value 
----- | -----
**SBB License** | Apache License 2.0
**Core Technology** | Java
**Project URL** | [https://deeplearning4j.org](https://deeplearning4j.org)
**Source Location** | [https://github.com/deeplearning4j/deeplearning4j](https://github.com/deeplearning4j/deeplearning4j)
*Tag(s)* |ML, ML Framework


## Detectron2

<p>Detectron is Facebook AI Research&#8217;s software system that implements state-of-the-art object detection algorithms, including <a rel="nofollow" href="https://arxiv.org/abs/1703.06870">Mask R-CNN</a>. Detectron2 is a ground-up rewrite of Detectron that started with maskrcnn-benchmark. The platform is now implemented in <a href="https://pytorch.org/">PyTorch</a>.  With a new, more modular design.  Detectron2 is flexible and extensible,  and able to provide fast training on single or multiple GPU servers.  Detectron2 includes high-quality implementations of state-of-the-art  object detection algorithms,</p>



<p>New in Detctron 2:</p>



<ul><li>It is powered by the <a href="https://pytorch.org" rel="nofollow">PyTorch</a> deep learning framework.</li><li>Includes more features such as panoptic segmentation, densepose, Cascade R-CNN, rotated bounding boxes, etc.</li><li>Can be used as a library to support <a href="https://github.com/facebookresearch/detectron2/blob/master/projects">different projects</a> on top of it. We&#8217;ll open source more research projects in this way.</li><li>It <a href="https://detectron2.readthedocs.io/notes/benchmarks.html" rel="nofollow">trains much faster</a>.</li></ul>



<p>The goal of Detectron is to provide a high-quality, high-performance codebase for object detection <em>research</em>. It is designed to be flexible in order to support rapid implementation and evaluation of novel research.</p>



<p>A number of Facebook teams use this platform to train custom models for a variety of applications including augmented reality and community integrity. Once trained, these models can be deployed in the cloud and on mobile devices, powered by the highly efficient Caffe2 runtime.</p>



<p>Documentation on: <a href="https://detectron2.readthedocs.io/index.html" target="_blank" rel="noreferrer noopener" aria-label="https://detectron2.readthedocs.io/index.html (opens in a new tab)">https://detectron2.readthedocs.io/index.html</a></p>

Item | Value 
----- | -----
**SBB License** | Apache License 2.0
**Core Technology** | Python
**Project URL** | [https://github.com/facebookresearch/Detectron2](https://github.com/facebookresearch/Detectron2)
**Source Location** | [https://github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)
*Tag(s)* |ML, ML Framework, Python


## Dopamine

<p>Dopamine is a research framework for fast prototyping of reinforcement learning algorithms. It aims to fill the need for a small, easily grokked codebase in which users can freely experiment with wild ideas (speculative research).</p>



<p>Our design principles are:</p>



<ul><li><em>Easy experimentation</em>: Make it easy for new users to run benchmark experiments.</li><li><em>Flexible development</em>: Make it easy for new users to try out research ideas.</li><li><em>Compact and reliable</em>: Provide implementations for a few, battle-tested algorithms.</li><li><em>Reproducible</em>: Facilitate reproducibility in results.</li></ul>

Item | Value 
----- | -----
**SBB License** | Apache License 2.0
**Core Technology** | Python
**Project URL** | [https://github.com/google/dopamine](https://github.com/google/dopamine)
**Source Location** | [https://github.com/google/dopamine](https://github.com/google/dopamine)
*Tag(s)* |ML, ML Framework, Reinforcement Learning


## Fastai

<p>The fastai library simplifies training fast and accurate neural nets using modern best practices. Fast.ai’s mission is to make the power of state of the art deep learning available to anyone. fastai sits on top of <a href="https://pytorch.org/">PyTorch</a>, which provides the foundation.</p>



<p>fastai is a deep learning library which provides high-level components that can quickly and easily provide state-of-the-art results in standard deep learning domains, and provides researchers with low-level components that can be mixed and matched to build new approaches. It aims to do both things without substantial compromises in ease of use, flexibility, or performance.</p>



<p>Docs can be found on:<a href="http://docs.fast.ai/" target="_blank" rel="noopener noreferrer"> http://docs.fast.ai/</a></p>

Item | Value 
----- | -----
**SBB License** | Apache License 2.0
**Core Technology** | Python
**Project URL** | [http://www.fast.ai/](http://www.fast.ai/)
**Source Location** | [https://github.com/fastai/fastai/](https://github.com/fastai/fastai/)
*Tag(s)* |ML, ML Framework


## Featuretools

<p><em>One of the holy grails of machine learning is to automate more and more of the feature engineering process.&#8221;</em> ― Pedro</p>



<p><a href="https://www.featuretools.com">Featuretools</a> is a python library for automated feature engineering. Featuretools automatically creates features from temporal and relational datasets. Featuretools works alongside tools you already use to build machine learning pipelines. You can load in pandas dataframes and automatically create meaningful features in a fraction of the time it would take to do manually.</p>



<p>Featuretools is a python library for automated feature engineering. Featuretools can automatically create a single table of features for any &#8220;target entity&#8221;. </p>



<p>Featuretools is a framework to perform automated feature engineering. It excels at transforming transactional and relational datasets into feature matrices for machine learning.</p>

Item | Value 
----- | -----
**SBB License** | BSD License 2.0 (3-clause, New or Revised) License
**Core Technology** | Python
**Project URL** | [https://www.featuretools.com/](https://www.featuretools.com/)
**Source Location** | [https://github.com/Featuretools/featuretools](https://github.com/Featuretools/featuretools)
*Tag(s)* |ML, ML Framework, Python


## FlyingSquid

<p>FlyingSquid is a ML framework for automatically building models from multiple noisy label sources. Users write functions that generate noisy labels for data, and FlyingSquid uses the agreements and disagreements between them to learn a <em>label model</em> of how accurate the <em>labeling functions</em> are. The label model can be used directly for downstream applications, or it can be used to train a powerful end model.</p>

Item | Value 
----- | -----
**SBB License** | Apache License 2.0
**Core Technology** | Python
**Project URL** | [http://hazyresearch.stanford.edu/flyingsquid](http://hazyresearch.stanford.edu/flyingsquid)
**Source Location** | [https://github.com/HazyResearch/flyingsquid](https://github.com/HazyResearch/flyingsquid)
*Tag(s)* |ML Framework, Python


## Igel

<p>A delightful machine learning tool that allows you to train/fit, test and use models <strong>without writing code</strong>. </p>



<p>The goal of the project is to provide machine learning for <strong>everyone</strong>, both technical and non-technical users.</p>



<p>I needed a tool sometimes, which I can use to fast create a machine learning prototype. Whether to build some proof of concept or create a fast draft model to prove a point. I find myself often stuck at writing boilerplate code and/or thinking too much of how to start this.</p>



<p>Therefore, I decided to create <strong>igel</strong>. Hopefully, it will make it easier for technical and non-technical users to build machine learning models.</p>



<p>Features:</p>



<ul><li>Supports all state of the art machine learning models (even preview models)</li><li>Supports different data preprocessing methods</li><li>Provides flexibility and data control while writing configurations</li><li>Supports cross validation</li><li>Supports both hyperparameter search (version &#62;= 0.2.8)</li><li>Supports yaml and json format</li><li>Supports different sklearn metrics for regression, classification and clustering</li><li>Supports multi-output/multi-target regression and classification</li><li>Supports multi-processing for parallel model construction</li></ul>



<p>Docs on:<a href="https://igel.readthedocs.io/en/latest/" target="_blank" rel="noreferrer noopener"> https://igel.readthedocs.io/en/latest/</a></p>

Item | Value 
----- | -----
**SBB License** | MIT License
**Core Technology** | Python
**Project URL** | [https://github.com/nidhaloff/igel](https://github.com/nidhaloff/igel)
**Source Location** | [https://github.com/nidhaloff/igel](https://github.com/nidhaloff/igel)
*Tag(s)* |ML Framework


## Karate Club

<p>Karate Club is an unsupervised machine learning extension library for <a rel="noreferrer noopener" href="https://networkx.github.io/" target="_blank">NetworkX</a>.</p>



<p><em>Karate Club</em> consists of state-of-the-art methods to do unsupervised learning on graph structured data. To put it simply it is a Swiss Army knife for small-scale graph mining research. First, it provides network embedding techniques at the node and graph level. Second, it includes a variety of overlapping and non-overlapping community detection methods. Implemented methods cover a wide range of network science (NetSci, Complenet), data mining (<a href="http://icdm2019.bigke.org/">ICDM</a>, <a href="http://www.cikm2019.net/">CIKM</a>, <a href="https://www.kdd.org/kdd2020/">KDD</a>), artificial intelligence (<a href="http://www.aaai.org/Conferences/conferences.php">AAAI</a>, <a href="https://www.ijcai.org/">IJCAI</a>) and machine learning (<a href="https://nips.cc/">NeurIPS</a>, <a href="https://icml.cc/">ICML</a>, <a href="https://iclr.cc/">ICLR</a>) conferences, workshops, and pieces from prominent journals.</p>



<p>The documentation can be found at: <a rel="noreferrer noopener" href="https://karateclub.readthedocs.io/en/latest/" target="_blank">https://karateclub.readthedocs.io/en/latest/</a></p>



<p>The Karate ClubAPI draws heavily from the ideas of scikit-learn and theoutput generated is suitable as input for scikit-learn’s machinelearning procedures.</p>



<p>The paper can be found at: <a href="https://arxiv.org/pdf/2003.04819.pdf" target="_blank" rel="noreferrer noopener">https://arxiv.org/pdf/2003.04819.pdf</a></p>

Item | Value 
----- | -----
**SBB License** | GNU General Public License (GPL) 3.0
**Core Technology** | Python
**Project URL** | [https://karateclub.readthedocs.io/en/latest/](https://karateclub.readthedocs.io/en/latest/)
**Source Location** | [https://github.com/benedekrozemberczki/karatecluB](https://github.com/benedekrozemberczki/karatecluB)
*Tag(s)* |ML Framework


## Keras

<p>Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.</p>



<p>Use Keras if you need a deep learning library that:</p>



<ul><li>Allows for easy and fast prototyping (through user friendliness, modularity, and extensibility).</li><li>Supports both convolutional networks and recurrent networks, as well as combinations of the two.</li><li>Runs seamlessly on CPU and GPU.</li></ul>

Item | Value 
----- | -----
**SBB License** | MIT License
**Core Technology** | Python
**Project URL** | [https://keras.io/](https://keras.io/)
**Source Location** | [https://github.com/keras-team/keras](https://github.com/keras-team/keras)
*Tag(s)* |ML, ML Framework


## learn2learn

<p>learn2learn is a PyTorch library for meta-learning implementations.</p>



<p></p>



<p>The goal of meta-learning is to enable agents to <em>learn how to learn</em>. That is, we would like our agents to become better learners as they solve more and more tasks. </p>



<p>Features:</p>



<p>learn2learn provides high- and low-level utilities for meta-learning.
The high-level utilities allow arbitrary users to take advantage of exisiting meta-learning algorithms.
The low-level utilities enable researchers to develop new and better meta-learning algorithms.</p>



<p>Some features of learn2learn include:</p>



<ul><li>Modular API: implement your own training loops with our low-level utilities.</li><li>Provides various meta-learning algorithms (e.g. MAML, FOMAML, MetaSGD, ProtoNets, DiCE)</li><li>Task generator with unified API, compatible with torchvision, torchtext, torchaudio, and cherry.</li><li>Provides standardized meta-learning tasks for vision (Omniglot, 
mini-ImageNet), reinforcement learning (Particles, Mujoco), and even 
text (news classification).</li><li>100% compatible with PyTorch &#8212; use your own modules, datasets, or libraries!</li></ul>

Item | Value 
----- | -----
**SBB License** | MIT License
**Core Technology** | Python
**Project URL** | [http://learn2learn.net/](http://learn2learn.net/)
**Source Location** | [https://github.com/learnables/learn2learn/](https://github.com/learnables/learn2learn/)
*Tag(s)* |ML Framework


## Lore

<p>Lore is a python framework to make machine learning approachable for Engineers and maintainable for Data Scientists.</p>



<p>Features</p>



<ul><li>Models support hyper parameter search over estimators with a data pipeline. They will efficiently utilize multiple GPUs (if available) with a couple different strategies, and can be saved and distributed for horizontal scalability.</li><li>Estimators from multiple packages are supported: <a href="https://keras.io/" rel="nofollow">Keras</a> (TensorFlow/Theano/CNTK), <a href="https://xgboost.readthedocs.io/" rel="nofollow">XGBoost</a> and <a href="http://scikit-learn.org/stable/" rel="nofollow">SciKit Learn</a>. They can all be subclassed with build, fit or predict overridden to completely customize your algorithm and architecture, while still benefiting from everything else.</li><li>Pipelines avoid information leaks between train and test sets, and one pipeline allows experimentation with many different estimators. A disk based pipeline is available if you exceed your machines available RAM.</li><li>Transformers standardize advanced feature engineering. For example, convert an American first name to its statistical age or gender using US Census data. Extract the geographic area code from a free form phone number string. Common date, time and string operations are supported efficiently through pandas.</li><li>Encoders offer robust input to your estimators, and avoid common problems with missing and long tail values. They are well tested to save you from garbage in/garbage out.</li><li>IO connections are configured and pooled in a standard way across the app for popular (no)sql databases, with transaction management and read write optimizations for bulk data, rather than typical ORM single row operations. Connections share a configurable query cache, in addition to encrypted S3 buckets for distributing models and datasets.</li><li>Dependency Management for each individual app in development, that can be 100% replicated to production. No manual activation, or magic env vars, or hidden files that break python for everything else. No knowledge required of venv, pyenv, pyvenv, virtualenv, virtualenvwrapper, pipenv, conda. Ain’t nobody got time for that.</li><li>Tests for your models can be run in your Continuous Integration environment, allowing Continuous Deployment for code and training updates, without increased work for your infrastructure team.</li><li>Workflow Support whether you prefer the command line, a python console, jupyter notebook, or IDE. Every environment gets readable logging and timing statements configured for both production and development.</li></ul>

Item | Value 
----- | -----
**SBB License** | GNU General Public License (GPL) 2.0
**Core Technology** | Python
**Project URL** | [https://github.com/instacart/lore](https://github.com/instacart/lore)
**Source Location** | [https://github.com/instacart/lore](https://github.com/instacart/lore)
*Tag(s)* |ML, ML Framework, Python


## Microsoft Cognitive Toolkit (CNTK)

<p>The Microsoft Cognitive Toolkit (<a href="https://cntk.ai">https://cntk.ai</a>)  is a unified deep learning toolkit that describes neural networks as a  series of computational steps via a directed graph. In this directed  graph, leaf nodes represent input values or network parameters, while  other nodes represent matrix operations upon their inputs. CNTK allows  users to easily realize and combine popular model types such as  feed-forward DNNs, convolutional nets (CNNs), and recurrent networks  (RNNs/LSTMs). It implements stochastic gradient descent (SGD, error  backpropagation) learning with automatic differentiation and  parallelization across multiple GPUs and servers. CNTK has been  available under an open-source license since April 2015. </p>



<p>Docs on:<a href="https://docs.microsoft.com/en-us/cognitive-toolkit/" target="_blank" rel="noreferrer noopener" aria-label=" https://docs.microsoft.com/en-us/cognitive-toolkit/  (opens in a new tab)"> https://docs.microsoft.com/en-us/cognitive-toolkit/ </a></p>

Item | Value 
----- | -----
**SBB License** | MIT License
**Core Technology** | C++
**Project URL** | [https://docs.microsoft.com/en-us/cognitive-toolkit/](https://docs.microsoft.com/en-us/cognitive-toolkit/)
**Source Location** | [https://github.com/Microsoft/CNTK](https://github.com/Microsoft/CNTK)
*Tag(s)* |ML, ML Framework


## ml5.js

<p>ml5.js aims to make machine learning approachable for a broad audience of artists, creative coders, and students. The library provides access to machine learning algorithms and models in the browser, building on top of <a href="https://js.tensorflow.org/" rel="nofollow">TensorFlow.js</a> with no other external dependencies.</p>



<p>The library is supported by code examples, tutorials, and sample data sets with an emphasis on ethical computing. Bias in data, stereotypical harms, and responsible crowdsourcing are part of the documentation around data collection and usage.</p>



<p>ml5.js is heavily inspired by <a href="https://processing.org/" rel="nofollow">Processing</a> and <a href="https://p5js.org/" rel="nofollow">p5.js</a>.</p>

Item | Value 
----- | -----
**SBB License** | MIT License
**Core Technology** | Javascript
**Project URL** | [https://ml5js.org/](https://ml5js.org/)
**Source Location** | [https://github.com/ml5js/ml5-library](https://github.com/ml5js/ml5-library)
*Tag(s)* |Javascript, ML, ML Framework


## Mljar

<p>MLJAR is a platform for rapid prototyping, developing and deploying machine learning models. </p>



<p>MLJAR makes algorithm search and tuning painless. It checks many 
different algorithms for you. For each algorithm hyper-parameters are 
separately tuned. All computations run in parallel in MLJAR cloud, so 
you get your results very quickly. At the end the ensemble of models is 
created, so your predictive model will be super accurate.</p>



<p>There are two types of interface available in MLJAR:</p>



<ul><li>you can run Machine Learning models in your browser, you don&#8217;t need 
to code anything. Just upload dataset, click which attributes to use, 
which algorithms to use and go! This makes Machine Learning super easy 
for everyone and make it possible to get really useful models,</li><li>there is a python wrapper over MLJAR API, so you don&#8217;t need to open 
any browser or click on any button, just write fancy python code! We 
like it and hope you will like it too! To start using MLJAR python 
package please go to our <a href="https://github.com/mljar/mljar-api-python">github</a>.</li></ul>

Item | Value 
----- | -----
**SBB License** | MIT License
**Core Technology** | Python
**Project URL** | [https://mljar.com/](https://mljar.com/)
**Source Location** | [https://github.com/mljar/mljar-supervised](https://github.com/mljar/mljar-supervised)
*Tag(s)* |ML, ML Framework, Python


## MLsquare

<p>[ML]² &#8211; ML Square is a python library that utilises deep learning techniques to:</p>



<ul><li>Enable interoperability between existing standard machine learning frameworks.</li><li>Provide explainability as a first-class function.</li><li>Make ML self learnable.</li></ul>



<p>The following are the design goals:</p>



<ul><li>Bring Your Own Spec First.</li><li>Bring Your Own Experience First.</li><li>Consistent.</li><li>Compositional.</li><li>Modular.</li><li>Extensible</li></ul>



<p>See https://arxiv.org/pdf/2001.00818.pdf<strong> </strong>for a in depth explanation. </p>

Item | Value 
----- | -----
**SBB License** | MIT License
**Core Technology** | Python
**Project URL** | [https://mlsquare.readthedocs.io/en/latest/](https://mlsquare.readthedocs.io/en/latest/)
**Source Location** | [https://github.com/mlsquare/mlsquare](https://github.com/mlsquare/mlsquare)
*Tag(s)* |ML Framework


## NeuralStructuredLearning

<p>Neural Structured Learning (NSL) is a new learning paradigm to train neural networks by leveraging structured signals in addition to feature inputs. Structure can be explicit as represented by a graph or implicit as induced by adversarial perturbation.</p>



<p>Structured signals are commonly used to represent relations or similarity among samples that may be labeled or unlabeled. Leveraging these signals during neural network training harnesses both labeled and unlabeled data, which can improve model accuracy, particularly when the amount of labeled data is relatively small. Additionally, models trained with samples that are generated by adversarial perturbation have been shown to be robust against malicious attacks, which are designed to mislead a model&#8217;s prediction or classification.</p>



<p>NSL generalizes to Neural Graph Learning  as well as to Adversarial Learning. The NSL framework in TensorFlow provides the following easy-to-use APIs and tools for developers to train models with structured signals:</p>



<ul><li> <strong>Keras APIs</strong> to enable training with graphs (explicit structure) and adversarial pertubations (implicit structure). </li><li> <strong>TF ops and functions</strong> to enable training with structure when using lower-level TensorFlow APIs </li><li> <strong>Tools</strong> to build graphs and construct graph inputs for training </li></ul>



<p>NSL is part of the TensorFlow framework. More info on: <a href="https://www.tensorflow.org/neural_structured_learning/" target="_blank" rel="noreferrer noopener" aria-label="https://www.tensorflow.org/neural_structured_learning/ (opens in a new tab)">https://www.tensorflow.org/neural_structured_learning/</a></p>

Item | Value 
----- | -----
**SBB License** | Apache License 2.0
**Core Technology** | Python
**Project URL** | [https://www.tensorflow.org/neural_structured_learning/](https://www.tensorflow.org/neural_structured_learning/)
**Source Location** | [https://github.com/tensorflow/neural-structured-learning](https://github.com/tensorflow/neural-structured-learning)
*Tag(s)* |ML, ML Framework, Python


## NNI (Neural Network Intelligence)

<p>NNI (Neural Network Intelligence) is a toolkit to help users run automated machine learning (AutoML) experiments. The tool dispatches and runs trial jobs generated by tuning algorithms to search the best neural architecture and/or hyper-parameters in different environments like local machine, remote servers and cloud.  (Microsoft ML project)</p>



<p>Who should consider using NNI:</p>



<ul><li>Those who want to try different AutoML algorithms in their training code (model) at their local machine.</li><li>Those who want to run AutoML trial jobs in different environments to speed up search (e.g. remote servers and cloud).</li><li>Researchers and data scientists who want to implement their own AutoML algorithms and compare it with other algorithms.</li><li>ML Platform owners who want to support AutoML in their platform.</li></ul>



<p></p>

Item | Value 
----- | -----
**SBB License** | MIT License
**Core Technology** | Python
**Project URL** | [https://nni.readthedocs.io/en/latest/](https://nni.readthedocs.io/en/latest/)
**Source Location** | [https://github.com/Microsoft/nni](https://github.com/Microsoft/nni)
*Tag(s)* |ML, ML Framework


## NuPIC

<p>The Numenta Platform for Intelligent Computing (<strong>NuPIC</strong>) is a machine intelligence platform that implements the <a href="https://numenta.com/resources/papers-videos-and-more/">HTM learning algorithms</a>.  HTM is a detailed computational theory of the neocortex. At the core of  HTM are time-based continuous learning algorithms that store and recall  spatial and temporal patterns. NuPIC is suited to a variety of  problems, particularly anomaly detection and prediction of streaming  data sources. </p>



<p>Note: This project is in Maintenance Mode. </p>

Item | Value 
----- | -----
**SBB License** | GNU Affero General Public License Version 3
**Core Technology** | Python
**Project URL** | [https://numenta.org/](https://numenta.org/)
**Source Location** | [https://github.com/numenta/nupic](https://github.com/numenta/nupic)
*Tag(s)* |ML Framework, Python


## Plato

<p>The Plato Research Dialogue System is a flexible framework that can be used to create, train, and evaluate conversational AI agents in various environments. It supports interactions through speech, text, or dialogue acts and each conversational agent can interact with data, human users, or other conversational agents (in a multi-agent setting). Every component of every agent can be trained independently online or offline and Plato provides an easy way of wrapping around virtually any existing model, as long as Plato&#8217;s interface is adhered to.</p>



<p>OSS by Uber.</p>

Item | Value 
----- | -----
**SBB License** | MIT License
**Core Technology** | Python
**Project URL** | [https://github.com/uber-research/plato-research-dialogue-system](https://github.com/uber-research/plato-research-dialogue-system)
**Source Location** | [https://github.com/uber-research/plato-research-dialogue-system](https://github.com/uber-research/plato-research-dialogue-system)
*Tag(s)* |ML, ML Framework


## Polyaxon
<p><span class="text-gray-dark mr-2">A platform for reproducible and scalable machine learning and deep learning on kubernetes </span></p>
<p>Polyaxon is a platform for building, training, and monitoring large scale deep learning applications.</p>
<p>Polyaxon deploys into any data center, cloud provider, or can be hosted and managed by Polyaxon, and it supports all the major deep learning frameworks such as Tensorflow, MXNet, Caffe, Torch, etc.</p>
<p>Polyaxon makes it faster, easier, and more efficient to develop deep learning applications by managing workloads with smart container and node management. And it turns GPU servers into shared, self-service resources for your team or organization.</p>

Item | Value 
----- | -----
**SBB License** | MIT License
**Core Technology** | Python
**Project URL** | [https://polyaxon.com/](https://polyaxon.com/)
**Source Location** | [https://github.com/polyaxon/polyaxon](https://github.com/polyaxon/polyaxon)
*Tag(s)* |ML, ML Framework


## PyCaret

<p>PyCaret is an open source <code>low-code</code> machine learning library in Python that aims to reduce the hypothesis to insights cycle time in a ML experiment. It enables data scientists to perform end-to-end experiments quickly and efficiently. In comparison with the other open source machine learning libraries, PyCaret is an alternate low-code library that can be used to perform complex machine learning tasks with only few lines of code. PyCaret is essentially a Python wrapper around several machine learning libraries and frameworks such as <code>scikit-learn</code>, <code>XGBoost</code>, <code>Microsoft LightGBM</code>, <code>spaCy</code> and many more.</p>



<p>The design and simplicity of PyCaret is inspired by the emerging role of <code>citizen data scientists</code>, a term first used by Gartner. Citizen Data Scientists are <code>power users</code> who can perform both simple and moderately sophisticated analytical tasks that would previously have required more expertise. Seasoned data scientists are often difficult to find and expensive to hire but citizen data scientists can be an effective way to mitigate this gap and address data related challenges in business setting.</p>



<p>PyCaret claims to be <code>imple</code>, <code>easy to use</code> and <code>deployment ready</code>. All the steps performed in a ML experiment can be reproduced using a pipeline that is automatically developed and orchestrated in PyCaret as you progress through the experiment. A <code>pipeline</code> can be saved in a binary file format that is transferable across environments.</p>

Item | Value 
----- | -----
**SBB License** | MIT License
**Core Technology** | Python
**Project URL** | [https://www.pycaret.org](https://www.pycaret.org)
**Source Location** | [https://github.com/pycaret/pycaret](https://github.com/pycaret/pycaret)
*Tag(s)* |ML Framework


## Pyro

<p><span class="col-11 text-gray-dark mr-2">Deep universal probabilistic programming with Python and PyTorch. Pyro is in an alpha release. It is developed and used by <a rel="nofollow" href="http://uber.ai">Uber AI Labs</a>.</span></p>



<p>Pyro is a universal probabilistic programming language (PPL) written in Python and supported by <a href="http://pytorch.org">PyTorch</a>  on the backend. Pyro enables flexible and expressive deep probabilistic  modeling, unifying the best of modern deep learning and Bayesian  modeling. It was designed with these key principles:</p>



<ul><li>Universal: Pyro can represent any computable probability distribution.</li><li>Scalable: Pyro scales to large data sets with little overhead.</li><li>Minimal: Pyro is implemented with a small core of powerful, composable abstractions.</li><li>Flexible: Pyro aims for automation when you want it, control when you need it. </li></ul>



<p>Documentation on: <a rel="noreferrer noopener" aria-label="http://docs.pyro.ai/ (opens in a new tab)" href="http://docs.pyro.ai/" target="_blank">http://docs.pyro.ai/</a></p>

Item | Value 
----- | -----
**SBB License** | GNU General Public License (GPL) 2.0
**Core Technology** | Python
**Project URL** | [http://pyro.ai/](http://pyro.ai/)
**Source Location** | [https://github.com/uber/pyro](https://github.com/uber/pyro)
*Tag(s)* |ML, ML Framework, Python


## Pythia

<p>Pythia is a modular framework for supercharging vision and language research built on top of PyTorch created by Facebook. </p>



<p>You can use Pythia to bootstrap for your next vision and language  multimodal research project.  Pythia can also act as starter codebase  for challenges around vision and language datasets (TextVQA challenge,  VQA challenge). </p>



<p>It features:</p>



<ul><li><strong>Model Zoo</strong>: Reference implementations for state-of-the-art vision and language model including
<a href="https://arxiv.org/abs/1904.08920">LoRRA</a> (SoTA on VQA and TextVQA),
<a href="https://arxiv.org/abs/1807.09956">Pythia</a> model (VQA 2018 challenge winner) and <a href="https://github.com/facebookresearch/pythia/blob/master">BAN</a>.</li><li><strong>Multi-Tasking</strong>: Support for multi-tasking which allows training on multiple dataset together.</li><li><strong>Datasets</strong>: Includes support for various datasets built-in including VQA, VizWiz, TextVQA and VisualDialog.</li><li><strong>Modules</strong>: Provides implementations for many commonly used layers in vision and language domain</li><li><strong>Distributed</strong>: Support for distributed training based on DataParallel as well as DistributedDataParallel.</li><li><strong>Unopinionated</strong>: Unopinionated about the dataset and model implementations built on top of it.</li><li><strong>Customization</strong>: Custom losses, metrics, scheduling, optimizers, tensorboard; suits all your custom needs.</li></ul>

Item | Value 
----- | -----
**SBB License** | BSD License 2.0 (3-clause, New or Revised) License
**Core Technology** | Python
**Project URL** | [https://learnpythia.readthedocs.io/en/latest/index.html](https://learnpythia.readthedocs.io/en/latest/index.html)
**Source Location** | [https://github.com/facebookresearch/pythia](https://github.com/facebookresearch/pythia)
*Tag(s)* |ML, ML Framework, Python


## PyTorch

<p>PyTorch is a Python-first machine learning framework that is utilized  heavily towards deep learning. It supports CUDA technology (From NVIDIA)  to fully use the the power of the dedicated GPUs in training, analyzing  and validating neural networks models.</p>



<p>Deep learning frameworks have often focused on either usability or speed, but not both.  PyTorch is a machine learning library that shows that these two goals are in fact compatible: it provides an imperative and Pythonic programming style that supports code as a model, makes debugging easy and is consistent with other popular scientific computing libraries, while remaining efficient and supporting hardware accelerators such as GPUs.</p>



<p>PyTorch is very widely used, and is under active development and support. PyTorch is:</p>



<ul><li>a deep learning framework that puts Python first.</li><li>&#160;a research-focused framework.</li><li>Python package that provides two high-level features:</li></ul>



<p>Pytorch uses tensor computation (like NumPy) with strong GPU acceleration. It can use deep neural networks built on a tape-based autograd system.</p>



<p>PyTorch is a Python package that provides two high-level features:</p>



<ul><li>Tensor computation (like NumPy) with strong GPU acceleration</li><li>Deep neural networks built on a tape-based autograd system</li></ul>



<p>You can reuse your favorite Python packages such as NumPy, SciPy and Cython to extend PyTorch when needed. PyTorch has become a popular tool in the deep learning research community by combining a focus on usability with careful performance considerations.</p>



<p>A very good overview of the design principles and architecture of PyTorch can be found in this paper <a rel="noreferrer noopener" aria-label="https://arxiv.org/pdf/1912.01703.pdf (opens in a new tab)" href="https://arxiv.org/pdf/1912.01703.pdf" target="_blank">https://arxiv.org/pdf/1912.01703.pdf</a> .</p>

Item | Value 
----- | -----
**SBB License** | MIT License
**Core Technology** | Python
**Project URL** | [http://pytorch.org/](http://pytorch.org/)
**Source Location** | [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)
*Tag(s)* |ML, ML Framework


## ReAgent

<p>ReAgent is an open source end-to-end platform for applied 
reinforcement learning (RL) developed and used at Facebook. ReAgent is 
built in Python and uses PyTorch for modeling and training and 
TorchScript for model serving. The platform contains workflows to train 
popular deep RL algorithms and includes data preprocessing, feature 
transformation, distributed training, counterfactual policy evaluation, 
and optimized serving. For more detailed information about ReAgent see 
the white paper <a href="https://research.fb.com/publications/horizon-facebooks-open-source-applied-reinforcement-learning-platform/">here</a>.</p>



<p>The platform was once named &#8220;Horizon&#8221; but we have adopted the name 
&#8220;ReAgent&#8221; recently to emphasize its broader scope in decision making and
 reasoning.</p>

Item | Value 
----- | -----
**SBB License** | BSD License 2.0 (3-clause, New or Revised) License
**Core Technology** | Python
**Project URL** | [https://engineering.fb.com/ml-applications/horizon/](https://engineering.fb.com/ml-applications/horizon/)
**Source Location** | [https://github.com/facebookresearch/ReAgent](https://github.com/facebookresearch/ReAgent)
*Tag(s)* |ML, ML Framework, Python


## RLCard

<p>RLCard is a toolkit for Reinforcement Learning (RL) in card games. It
 supports multiple card environments with easy-to-use interfaces. The 
goal of RLCard is to bridge reinforcement learning and imperfect 
information games, and push forward the research of reinforcement 
learning in domains with multiple agents, large state and action space, 
and sparse reward. RLCard is developed by <a href="http://faculty.cs.tamu.edu/xiahu/">DATA Lab</a> at Texas A&#38;M University.</p>



<ul><li>Paper: <a href="https://arxiv.org/abs/1910.04376">https://arxiv.org/abs/1910.04376</a></li></ul>

Item | Value 
----- | -----
**SBB License** | MIT License
**Core Technology** | Python
**Project URL** | [http://rlcard.org/](http://rlcard.org/)
**Source Location** | [https://github.com/datamllab/rlcard](https://github.com/datamllab/rlcard)
*Tag(s)* |ML Framework, Python


## Scikit-learn

<p>scikit-learn is a Python module for machine learning. s cikit-learn is a Python module for machine learning built on top of SciPy and is distributed under the 3-Clause BSD license.</p>



<p>Key features:</p>



<ul><li>Simple and efficient tools for predictive data analysis</li><li>Accessible to everybody, and reusable in various contexts</li><li>Built on NumPy, SciPy, and matplotlib</li><li>Open source, commercially usable &#8211; BSD license</li></ul>

Item | Value 
----- | -----
**SBB License** | BSD License 2.0 (3-clause, New or Revised) License
**Core Technology** | Python
**Project URL** | [http://scikit-learn.org](http://scikit-learn.org)
**Source Location** | [https://github.com/scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn)
*Tag(s)* |ML, ML Framework


## SINGA

<p>Distributed deep learning system. </p>



<p>SINGA was initiated by the DB System Group at National University of Singapore in 2014, in collaboration with the database group of Zhejiang University.</p>



<p>SINGA‘s software stack includes three major components, namely, core, IO and model:</p>



<ol><li> The core component provides memory management and tensor operations.</li><li>IO has classes for reading (and writing) data from (to) disk and network.</li><li>The model component provides data structures and algorithms for machine learning models, e.g., layers for neural network models, optimizers/initializer/metric/loss for general machine learning models.</li></ol>

Item | Value 
----- | -----
**SBB License** | Apache License 2.0
**Core Technology** | Java
**Project URL** | [http://singa.apache.org/](http://singa.apache.org/)
**Source Location** | [https://github.com/apache/singa](https://github.com/apache/singa)
*Tag(s)* |ML Framework


## Streamlit

<p>The fastest way to build custom ML tools. Streamlit lets you create apps for your machine learning projects  with deceptively simple Python scripts. It supports hot-reloading, so  your app updates live as you edit and save your file. No need to mess  with HTTP requests, HTML, JavaScript, etc. All you need is your favorite  editor and a browser.</p>



<p>Documentation on: <a href="https://streamlit.io/docs/" target="_blank" rel="noreferrer noopener" aria-label="https://streamlit.io/docs/ (opens in a new tab)">https://streamlit.io/docs/</a></p>

Item | Value 
----- | -----
**SBB License** | Apache License 2.0
**Core Technology** | Javascipt, Python
**Project URL** | [https://streamlit.io/](https://streamlit.io/)
**Source Location** | [https://github.com/streamlit/streamlit](https://github.com/streamlit/streamlit)
*Tag(s)* |ML, ML Framework, ML Hosting, ML Tool, Python


## Tensorflow

<p>TensorFlow is an Open Source Software Library for Machine Intelligence. TensorFlow is by far the most used and popular ML open source project. And since the first initial release was only just in November 2015 it is expected that the impact of this OSS package will expand even more.</p>



<p>TensorFlow™ is an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them. The flexible architecture allows you to deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device with a single API. TensorFlow was originally developed by researchers and engineers working on the Google Brain Team within Google&#8217;s Machine Intelligence research organization for the purposes of conducting machine learning and deep neural networks research, but the system is general enough to be applicable in a wide variety of other domains as well.</p>



<p>TensorFlow comes with a tool called <a href="https://www.tensorflow.org/versions/r0.11/how_tos/graph_viz/index.html">TensorBoard</a> which you can use to get some insight into what is happening. TensorBoard is a suite of web applications for inspecting and understanding your TensorFlow runs and graphs.</p>



<p>There is also a version of TensorFlow that runs in a browser. This is TensorFlow.js (<a rel="noopener noreferrer" href="https://js.tensorflow.org/" target="_blank">https://js.tensorflow.org/</a> ). TensorFlow.js is a WebGL accelerated, browser based JavaScript library for training and deploying ML models.</p>



<p>Since privacy is a contentious fight TensorFlow has now (2020) also a library called &#8216;TensorFlow Privacy&#8217; . This is a python library that includes implementations of TensorFlow optimizers for training machine learning models with differential privacy. The library comes with tutorials and analysis tools for computing the privacy guarantees provided. See: <a href="https://github.com/tensorflow/privacy" target="_blank" rel="noreferrer noopener" aria-label="https://github.com/tensorflow/privacy  (opens in a new tab)">https://github.com/tensorflow/privacy </a></p>

Item | Value 
----- | -----
**SBB License** | Apache License 2.0
**Core Technology** | C
**Project URL** | [https://www.tensorflow.org/](https://www.tensorflow.org/)
**Source Location** | [https://github.com/tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)
*Tag(s)* |ML, ML Framework


## TF Encrypted

<p>TF Encrypted is a framework for encrypted machine learning in TensorFlow. It looks and feels like TensorFlow, taking advantage of the ease-of-use of the Keras API while enabling training and prediction over encrypted data via secure multi-party computation and homomorphic encryption. TF Encrypted aims to make privacy-preserving machine learning readily available, without requiring expertise in cryptography, distributed systems, or high performance computing.</p>

Item | Value 
----- | -----
**SBB License** | Apache License 2.0
**Core Technology** | Python
**Project URL** | [https://tf-encrypted.io/](https://tf-encrypted.io/)
**Source Location** | [https://github.com/tf-encrypted/tf-encrypted](https://github.com/tf-encrypted/tf-encrypted)
*Tag(s)* |ML, ML Framework, Privacy


## Theano
<p>Theano is a Python library that allows you to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays efficiently. It can use GPUs and perform efficient symbolic differentiation.</p>
<p>Note: After almost ten years of development the company behind Theano has stopped development and support(Q4-2017). But this library has been an innovation driver for many other OSS ML packages!</p>
<p>Since a lot of ML libraries and packages use Theano you should check (as always) the health of your ML stack.</p>

Item | Value 
----- | -----
**SBB License** | MIT License
**Core Technology** | Python
**Source Location** | [https://github.com/Theano/Theano](https://github.com/Theano/Theano)
*Tag(s)* |ML, ML Framework, Python


## Thinc

<p>Thinc is the machine learning library powering spaCy. It features a battle-tested linear model designed for large sparse learning problems, and a flexible neural network model under development for spaCy v2.0.</p>



<p>Thinc is a lightweight deep learning library that offers an elegant, type-checked, functional-programming API for composing models, with support for layers defined in other frameworks such as PyTorch, TensorFlow and MXNet. You can use Thinc as an interface layer, a standalone toolkit or a flexible way to develop new models.</p>



<p>Thinc is a practical toolkit for implementing models that follow the &#8220;Embed, encode, attend, predict&#8221; architecture. It&#8217;s designed to be easy to install, efficient for CPU usage and optimised for NLP and deep learning with text – in particular, hierarchically structured input and variable-length sequences.</p>

Item | Value 
----- | -----
**SBB License** | MIT License
**Core Technology** | Python
**Project URL** | [https://thinc.ai/](https://thinc.ai/)
**Source Location** | [https://github.com/explosion/thinc](https://github.com/explosion/thinc)
*Tag(s)* |ML, ML Framework, NLP, Python


## Turi
<p><span class="col-11 text-gray-dark mr-2">Turi Create simplifies the development of custom machine learning models. </span>Turi is OSS machine learning from Apple.</p>
<p>Turi Create simplifies the development of custom machine learning models. You don&#8217;t have to be a machine learning expert to add recommendations, object detection, image classification, image similarity or activity classification to your app.</p>

Item | Value 
----- | -----
**SBB License** | BSD License 2.0 (3-clause, New or Revised) License
**Core Technology** | Python
**Project URL** | [https://github.com/apple/turicreate](https://github.com/apple/turicreate)
**Source Location** | [https://github.com/apple/turicreate](https://github.com/apple/turicreate)
*Tag(s)* |ML, ML Framework, ML Hosting


## TuriCreate
<p>This SBB is from Apple. Apple, is with Siri already for a long time active in machine learning. But even Apple is releasing building blocks under OSS licenses now.</p>
<p>Turi Create simplifies the development of custom machine learning models. You don&#8217;t have to be a machine learning expert to add recommendations, object detection, image classification, image similarity or activity classification to your app.</p>
<ul>
<li><strong>Easy-to-use:</strong> Focus on tasks instead of algorithms</li>
<li><strong>Visual:</strong> Built-in, streaming visualizations to explore your data</li>
<li><strong>Flexible:</strong> Supports text, images, audio, video and sensor data</li>
<li><strong>Fast and Scalable:</strong> Work with large datasets on a single machine</li>
<li><strong>Ready To Deploy:</strong> Export models to Core ML for use in iOS, macOS, watchOS, and tvOS apps</li>
</ul>

Item | Value 
----- | -----
**SBB License** | BSD License 2.0 (3-clause, New or Revised) License
**Core Technology** | Python
**Project URL** | [https://turi.com/index.html](https://turi.com/index.html)
**Source Location** | [https://github.com/apple/turicreate](https://github.com/apple/turicreate)
*Tag(s)* |ML, ML Framework, Python


## Vowpal Wabbit

<p>Vowpal Wabbit is a machine learning system which pushes the frontier 
of machine learning with techniques such as online, hashing, allreduce, 
reductions, learning2search, active, and interactive learning. There is a
 specific focus on reinforcement learning with several contextual bandit
 algorithms implemented and the online nature lending to the problem 
well. Vowpal Wabbit is a destination for implementing and maturing state
 of the art algorithms with performance in mind.</p>



<ul><li><strong>Input Format.</strong> The input format for the learning 
algorithm is substantially more flexible than might be expected. 
Examples can have features consisting of free form text, which is 
interpreted in a bag-of-words way. There can even be multiple sets of 
free form text in different namespaces.</li><li><strong>Speed.</strong> The learning algorithm is fast &#8212; similar to
 the few other online algorithm implementations out there. There are 
several optimization algorithms available with the baseline being sparse
 gradient descent (GD) on a loss function.</li><li><strong>Scalability.</strong> This is not the same as fast. Instead,
 the important characteristic here is that the memory footprint of the 
program is bounded independent of data. This means the training set is 
not loaded into main memory before learning starts. In addition, the 
size of the set of features is bounded independent of the amount of 
training data using the hashing trick.</li><li><strong>Feature Interaction.</strong> Subsets of features can be 
internally paired so that the algorithm is linear in the cross-product 
of the subsets. This is useful for ranking problems. The alternative of 
explicitly expanding the features before feeding them into the learning 
algorithm can be both computation and space intensive, depending on how 
it&#8217;s handled.</li></ul>



<p>Microsoft Research is a major contributor to Vowpal Wabbit.           </p>



<p></p>

Item | Value 
----- | -----
**SBB License** | MIT License
**Core Technology** | CPP
**Project URL** | [https://vowpalwabbit.org/](https://vowpalwabbit.org/)
**Source Location** | [https://github.com/VowpalWabbit/vowpal_wabbit](https://github.com/VowpalWabbit/vowpal_wabbit)
*Tag(s)* |ML, ML Framework


## XAI

<p>XAI is a Machine Learning library that is designed with AI 
explainability in its core. XAI contains various tools that enable for 
analysis and evaluation of data and models. The XAI library is 
maintained by <a href="http://ethical.institute/">The Institute for Ethical AI &#38; ML</a>, and it was developed based on the <a href="http://ethical.institute/principles.html">8 principles for Responsible Machine Learning</a>.</p>



<p>You can find the documentation at <a href="https://ethicalml.github.io/xai/index.html">https://ethicalml.github.io/xai/index.html</a>. </p>

Item | Value 
----- | -----
**SBB License** | MIT License
**Core Technology** | Python
**Project URL** | [https://ethical.institute/index.html](https://ethical.institute/index.html)
**Source Location** | [https://github.com/EthicalML/xai](https://github.com/EthicalML/xai)
*Tag(s)* |ML, ML Framework, Python

End of SBB list <br>