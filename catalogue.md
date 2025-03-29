# Solutions for FOSS ML

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

## Open Machine Learning Frameworks


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

## Categories

:::{note}
Using categories makes searching simple. However there is and will never be an unified topology to categorise all various Solution Building Blocks.
:::

* ML Frameworks: Within this category you will find mainly solutions to build applications. It contains APIs and a lot of functionality so you can focus on solving your business problem that you want to solve with AI or an LLM model.

* ML Tools: Mainly tools that are usable when working with various development tools, models or frameworks. Most ML Tools can be used within a Framework. 
