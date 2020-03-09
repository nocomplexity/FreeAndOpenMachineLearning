ML implementation challenges
==============================


Performance
---------------

Machine learning is amazingly slow. So before you use cloud services for your machine learning performance challenges, it is wise to do some experiments on a simple laptop. After you have a feeling of your model and the main performance bottlenecks than using clusters of advantaged GPUs offered by all Cloud providers is more effective. Also from a cost perspective. 

ML hardware is at its infancy. Even faster systems and wider deployment will lead to many more breakthroughs across a wide range of domains.


Many ML OSS software solutions are created in the Python programming language. Python is considered ‘slow’ and hard to parallelize. However many solutions exist to solve this problem. The most important OSS machine learning software solutions are by design capable to run on complete clusters of GPU (Graphical Processor Units). Scalability over GPU has proven to be more efficient for machine learning calculations than using more CPUs. This is because GPUs are by design more suited for the complex calculations needed to perform than CPUs.
GPUs are better for speeding up calculations that are needed for distributed machine learning algorithms. A GPU core is designed for handling multiple tasks simultaneously. A core for a GPU is simpler than a CPU core, but a GPU has many more cores than a CPU. 


New CPUs are being developed especially for machine learning algorithms. E.g. Google is developing Tensor Processing Units (TPU’s) to speed up the machine learning calculations. Of course this TPU’s are optimized for Google’s tensorflow software. But since this software is OSS everyone can take advantages if needed, since Google will offer TPU’s in their Cloud offerings. Of course Microsoft and other big Cloud providers are also developing their specialized machine learning processing units.

Performance is hard. Despite the fact that Python is a good choice for machine learning, processing large JSON lists can be slow. So optimization can be needed to speed up pre-processing during the data preparation phase.

Testing machine learning models
---------------------------------

Automated testing is a large part of software development. Unfortunately performing testing on software and infrastructure is still a mandatory  requirement for solid ML projects. 
Once a project reaches a certain level of complexity, the only way that it can be maintained is if it has a set of tests that identify the main functionality and allow you to verify that functionality is intact. Without automatic tests, it’s nearly impossible to identify where errors are occurring, and to fix those errors without causing further problems.

Testing should be done on:

- Data (training data)
- Infrastructure
- Software QA aspects. In fact all ISO QA factors should be evaluated. 
- Security, Privacy and safety aspects.

Overview ISO Qualitiy Standard(25010)

.. image:: /images/iso-25010.png  
   :alt: Typical NLP Architecture 
   :align: center 

The material of ISO is not open. But since quality matters and ISO 25010 is used heavily for managing quality aspects within business IT systems you should keep these factors in mind when developing test to minimize business risks.

Data testing for ML pipelines is different and can be complex. Data preparation and testing for machine learning  is not comparable with data testing for traditional IT projects. This is because it requires a statistical test performed on the data set. If input data changes this can have a significant effect on the outcome. 
Good statistical practice is an essential component of good scientific practice but also for real world ML applications. Especially when safety aspects play a role. Also mind that ML can be in essence is still seen as applied statistics. Validation of outcomes using statistical methods is proven science. 


Good statistical practice is an essential component of good scientific practice but also for real world ML applications. Especially when safety aspects play a role. Also mind that ML can be in essence is still seen as applied statistics. Validation of outcomes using statistical methods is proven science. 
In statistical hypothesis testing, the p-value or probability value is the probability of obtaining test results at least as extreme as the results actually observed during the test.
The p-value was never intended to be a substitute for scientific reasoning. Well-reasoned statistical arguments contain much more than the value of a single number and whether that number exceeds an arbitrary threshold. So for evaluating machine learning results the principles of American Statistical Association (ASA) can be useful. These principles are:

* P-values can indicate how incompatible the data are with a specified statistical model.
* P-values do not measure the probability that the studied hypothesis is true, or the probability that the data were produced by random chance alone. 
* Scientific conclusions and business or policy decisions should not be based only on whether a p-value passes a specific threshold. 
* Proper inference requires full reporting and transparency.  
* A p-value, or statistical significance, does not measure the size of an effect or the importance of a result.
* By itself, a p-value does not provide a good measure of evidence regarding a model or hypothesis. 



Machine learning projects tend to be pretty under-tested, which is unfortunate because they have all of the same complexity and maintainability issues as software projects.

* Tests codify your expectations at the point when you actually remember what you’re expecting.
* They allow you to offload verification to a machine.
* You can make changes to your code with confidence.



Interoperability
-----------------

Open standards do help. And with open standards you should look for real open standards. There are standards that not only specify how things should work, but also have an open source implementation that is using the standards for real. Keep away from standards that exist on paper only or standards that only have a reference implementation. Good standards are used and born from a practical need. 

In this way everyone implementing the standards will be forced to make sure the outcome and use of APIs is the same. Most of the time open standards lack an open implementation, so vendors can implement the specification and still lock you in. 

With interoperability for machine learning a trained model can be reused using different frameworks for an application. A trained model is the result of applying a machine learning algorithm to a set of training data. A new model using an already trained model can make predictions based on new input data. For example, a model that's been trained on a region's historical house prices may be able to predict a house's price when given the number of bedrooms and bathrooms.


Currently de facto standards on machine learning are just emerging. But due to all tools offered for applying machine learning it makes sense that models can be reused between machine learning frameworks.
Open Neural Network Exchange (ONNX) is the first step toward an open ecosystem that empowers AI developers to choose the right tools as their project evolves. ONNX provides an open source format for AI models. It defines an extensible computation graph model, as well as definitions of built-in operators and standard data types. 
Caffe2, PyTorch, Microsoft Cognitive Toolkit, Apache MXNet and other tools are developing ONNX support. Enabling interoperability between different frameworks and streamlining the path from research to production will increase the speed of innovation for ML applications.
See: http://onnx.ai/ for more information.

A standard that is already for many years (first version in 1998) available is the PMML standard. This Predictive Model Markup Language (PMML) is an XML-based predictive model interchange format. However many disadvantages exist that seem to prevent PMML from becoming a real interoperability standard for ML. (See http://dmg.org/pmml/v4-3/GeneralStructure.html ) 


Besides standards on interoperability for use of machine learning frameworks you need some standardization on datasets first. The good news is that raw datasets are often presented in a standard format like csv, json or xml. In this way some reuse of data is already possible. But knowing the data pipeline needed for machine learning more is needed. E.g. Currently there is no standard way to identify how a dataset was created, and what characteristics, motivations, and potential skews it represents.
Some answers that a good standardized metadata description on data should provide are e.g.:

* Why was the dataset created?
* What (other) tasks could the dataset be used for?
* Has the dataset been used for any tasks already?
* Who funded the creation of the dataset?
* Are relationships between instances made explicit in the data? 
* What preprocessing/cleaning was done?
* Was the “raw” data saved in addition to the preprocessed/cleaned data?
* Under what license can the data be (re)used?
* Are there privacy or security concerns related to the content of the data?


Debugging
----------

Machine learning is a fundamentally hard debugging problem. Debugging for machine learning is needed when:

- your algorithm doesn't work or 
- your algorithm doesn't work well enough.

What is unique about machine learning is that it is ‘exponentially’ harder to figure out what is wrong when things don’t work as expected. Compounding this debugging difficulty, there is often a delay in debugging cycles between implementing a fix or upgrade and seeing the result. Very rarely does an algorithm work the first time and so this ends up being where the majority of time is spent in building algorithms.



Continuous improvements
------------------------------

Machine learning models will degrade in accuracy in production. This since new input data is used that will be different from used training data. Input data will change over time.This problem of the changes in the data and relationships within data sets is called concept drift. 

Machine learning models are not a typical category of software. In fact a machine learning model should not be regarded as software at all. This means that maintenance should be organized and handled in a different way. There is never a final version of a machine learning model. So when using machine learning you need engineers that continuously updated and improved the model. 

So setting up end user feedback, accuracy measurements, monitoring data trends are important factors for organizations when using machine learning. But the traditional IT maintenance task as monitoring servers, network and infrastructure, security threats and application health are also still needed.

