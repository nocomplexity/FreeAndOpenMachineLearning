Security,Privacy and Safety 
=============================

Introduction
-------------

This section outlines security, privacy and safety concerns that matter when applying machine learning for real business use.

The complexity of ML technologies has fuelled fears that machine learning applications will cause harm in unforeseen circumstances, or that they will be manipulated to act in harmful ways. Think of a self driving car with its own ethics or algorithms that make prediction based on your personal data that really scare you. E.g. Predicting what diseases will hit you based on data from your grocery store. 

As with any technology: Technology is never neutral. You have to think before starting what values you implicitly use to design your new technology. All technology can and will be misused. But it is up to the designers to think of the risks when technology will be misused. On purpose or by accident. 

Machine learning systems should be operated reliably, safely and consistently. Not only under normal circumstances but also in unexpected conditions or when they are under attack for misuse. 

Machine learning software differs from traditional software because:

* The outcome is not easily predictable. 
* The used trained models are a black box, with very few options for transparency.
* Logical reasoning (or cause and effect) is not present. Predictions are made based on statistical number crunching complex algorithms which are non linear. 
* Both Non IT people and trained IT people will have a hard time figuring out machine learning systems, due to the new paradigms in use.


What makes security and safety more than normal aspects for machine learning driven applications is that by design neural networks are not designed to to make the inner workings easy to understand for humans and quality and risk managers.

Without a solid background in mathematics and software engineering evaluating the correct working of most machine learning application is impossible for security researchers safety auditors. 

However more and more people will dependent on the correct outcome of decisions made by machine learning software. So we should ask some critical questions:

* Is the system making any mistakes? 
* How do you know what alternatives were considered?
* What is the risk of trusting the outcome blind? 

Understanding how output produced by machine learning software is created will make more people comfortable with self-driving cars and other safety critical systems that will be machine learning enabled. In the end systems that can kill you must be secure and safe to use. So how do we get the process and trust chain to a level that we are not longer depended of:

* Software bugs
* Machine learning experts
* Auditors
* A proprietary certification process that end with a stamp (if paid enough)

From other sectors, like finance or oil industry we know that there is no simple solution. However regarding the risks involved only FOSS machine learning applications have the right elements needed to start working on processes that will give enough trust to use machine learning system for society at large.


.. include:: security.rst

.. include:: privacy.rst


