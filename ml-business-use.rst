Machine Learning for Business Problems
=======================================

Reading and talking about the in potential endless options for machine learning is nice and should be done. But applying machine learning for your business is where you can make a difference. This section is focussed on applying machine learning technology for real business use. Some business use cases where machine learning is already applied are outlined. This to give you some inspiration. But besides the technology more is needed for applying machine learning in a business with success. This section will also present some more in depth discussion of the several factors that should be taken into account when applying machine learning for real business use.

When to use machine learning for business problems?
-------------------------------------------------------

With the use of machine learning it is possible to learn from patterns and conditions to get new solid outcomes or predictions based on new data. Machine learning is able to learn from change patterns (data) at a pace that the human mind can not. This makes that machine learning is useful for a set of use cases were learning from data is possible or needed.
 

Machine learning should not be used for use cases that can be easily solved in another way. 
For example do not use machine learning driven solutions if your use case matches on of the following criteria:

* If it’s possible to structure a set of rules or “if-then scenarios” to handle your problem entirely, then there may be no need for machine learning at all.
* Your problem can be solved using statistical tools and software.

Common business use cases
-----------------------------

Healthcare
^^^^^^^^^^^^

Healthcare is due to the large amounts of data already available a perfect domain for solving challenges using machine learning. E.g. a challenging question for machine learning for healthcare is: Given the current use of machine learning for healthcare is given a patient’s electronic medical record data, can we prevent a person getting sick?

Machine learning is more and more used for automatic diagnostics. This can be data provided by X-ray scans or data retrieved from blood and tissue samples. machine learning has already proven to be valuable in detecting and predicting diseases for real people.

Predictive tasks for healthcare is maybe the way to keep people healthier and lower healthcare cost. The transformation from making people better towards preventing people getting sick will be long and hard, since this will be a real shift for the healthcare industry.

But given a large set of training data of de-identified medical records it is already possible to predict interesting aspects of the future for a patient that is not in training set.

Machine learning applications for healthcare are also to create better medicines by making use of all the data already available.

Language translation
^^^^^^^^^^^^^^^^^^^^^^^
Machine learning is already be used for automatic real-time message translation. E.g. Rocket Chat (The OSS Slack alternative, https://rocket.chat/ ) is using machine learning for this reason.
Since language translation needs context and lots of data, typically this use case is often NLP driven.
Language translation is as speech recognition a typical NLP application. Natural language processing (NLP) is area of machine learning that operates on (human)text and speech. See the section on NLP in this book for more use cases and insight in the specific NLP technologies.

Other areas for language translation are Speech recognition. Some great real time machine learning driven application already exist.

Chat bots
^^^^^^^^^^
 Currently all major tech companies like Amazon(Alexis), Google, Apple (Siri) have built a smart chatbot for the consumer market. Creating a chatbot (e.g. IRQ bot) was not new and difficult, however building a real ‘intelligent’ chat bot that has learning capabilities is another challenge. 
 
 Machine learning powered chatbots with real human like voices will help computers communicate with humans. But algorithms still have a hard time trying to figure out what you are saying, because context and tone of voice are hard to get right. Even for us humans, communication with other humans is most of the time hard. So building a smart chatbot that understands basic emotions in your voice is difficult. Machine learning isn’t advanced enough yet to carry on a dialogue without help, so a lot of the current chat bot software needs to be hand-coded. 



.. warning::

   This document is in alfa-stage!! 
   We are now working on some great NLP things. So not yet a NLP section in this alfa verion of this eBook!
   Collaboration is fun, so :ref:`Help Us <Help>` by contributing !

eCommerce Recommendation systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A well known application for machine learning for eCommerce systems is a machine learning enabled recommendation system system. Whether you buy a book, trip, music or visit a movie: On all major online ecommerce site you get a recommendation for a product that seems to fit your interest perfectly. Of course the purpose is to drive up the online sale, but the algorithms used are examples of still evolving machine learning algorithms. 
Examples of these systems are:

* Selling tickets to concerts based on your profile.
* NetFlix or cinema systems to make sure you stay hooked on watching more series and films you like. 
* Finding similar products in an eCommerce environment with a great chance you buy it. E.g. similar hotels, movies, books, etc.



Quality inspection and improvement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When computer vision technologies are combined with machine learning capabilities new opportunities arise. Examples of real world applications are:

* Inspecting tomatoes (and other fruit / vegetables) for quality and diseases.
* Inspecting quality of automatic created constructions (e.a. Constructions made by robots)

Vision
^^^^^^^^^

Since vision is captured in data machine learning is a great tool for building applications using vision (images, movies) technology. E.g.:

* Face detection. Writing software to detect faces and do recognition is very very hard to do using traditional programming methods.
* Image classification. In the old days we were happy when software was able to distinguish a cat and dog. In 2018 far more advanced applications are possible. E.g. giving details on all kind of aspects of photos. E.g. when you organize a conference you can use software to check the amount of suits or hoodies visiting your conference. Which is of course great for marketing. 
* Image similarity. Given an image, the goal of an image similarity model is to find "similar" images. Just like in image classification, deep learning methods have been shown to give incredible results on this challenging problem. However, unlike in image similarity, there isn't a need to generate labelled images for model creation. This model is completely unsupervised.
* Object Detection. Object detection is the task of simultaneously classifying (what) and localizing (where) object instances in an image.

Security
^^^^^^^^^^

* Email spam filters. Although simple rules can and should be applied, the enormous creativity of spammers and the amount send good fighting spam is a solid use case for a supervised machine learning problem. 
* Network filtering. Due to the learning capability of machine learning network security devices are improved using machine learning techniques.
* Fraud detection. Fraud detection is possible using enormous data and searching for strange patterns.

Besides fraud detection machine learning can also applied for IT security detections since intrusion detection systems and virus scanners are more and more shipped with self learning algorithms. Also Complex financial fraud schemes can be easily detected using predictive machine learning models.




Risk and compliance
^^^^^^^^^^^^^^^^^^^^
Evaluating risks can be done using large amounts of data. Natural language processing techniques can be used to validate highly automatic if your company meets regulations. Since audit and inspecting work is mostly based on standardized rules performed by knowledge workers this kind of work can be automated using machine learning techniques.

Example use cases 
---------------------

To outline some use cases that have been realized using machine learning technology, this paragraph summarize some real world cases to get some inspiration.

Applications for real business use of machine learning to solve real tangible problems is growing at a rapid pace. Below some examples of practical use of machine learning applications:

* Medical researchers are using machine learning to assess a person’s cardiovascular risk of a heart attack and stroke. 
* Air Traffic Controllers are using TensorFlow to predict flight routes through crowded airspace for safe and efficient landings.
* Engineers are using TensorFlow to analyze auditory data in the rainforest to detect logging trucks and other illegal activities.
* Scientists in Africa are using TensorFlow to detect diseases in Cassava plants to improving yield for farmers.
* Finding free parking space. http://www.peazy.in has developed an app using machine learning to assist with finding a free parking space in crowded cities. 


Business principles for Machine Learning applications
-------------------------------------------------------

Every good architecture is based on principles, requirements and constraints.This machine learning reference architecture is designed to easy the process of creating machine learning solutions. So below some general principles for machine learning applications. For your specific machine learning application use the principles that apply and make them smart. So include implications and consequences per principle.


Collaborate
^^^^^^^^^^^^^^^^

Statement: Collaborate
Rationale: Successful creation of ML applications require the collaboration of people with different expertises. You need e.g. business experts, infrastructure engineers, data engineers and innovation experts.
Implications: Organisational and culture must allow open collaboration. 

Unfair bias
^^^^^^^^^^^^^^

Statement: Avoid creating or reinforcing unfair bias
Rationale: Machine learning algorithms and datasets can reflect, reinforce, or reduce unfair biases. Recognize fair from unfair biases is not simple, and differs across cultures and societies. However always make sure to avoid unjust impacts on sensitive characteristics such as race, ethnicity, gender, nationality, income, sexual orientation, ability, and political or religious belief. 
Implications: Be transparent about your data and training datasets. Make models reproducible and auditable.

Built and test for safety
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Statement: Built and test for safety.
Rationale:  Use safety and security practices to avoid unintended results that create risks of harm.  Design your machine learning driven systems to be appropriately cautious
Implications: Perform risk assessments and safety tests.

Privacy by design
^^^^^^^^^^^^^^^^^^^

Statement: Incorporate privacy by design principles.
Rationale: Privacy by principles is more than being compliant with legal constraints as e.g. EU GDPR. It means that privacy safeguards,transparency and control over the use of data should be taken into account from the start. This is a hard and complex challenge. 






Business ethics
---------------------


Working with machine learning can, will and must raise severe ethical questions. Saying that you ‘Don't be evil’ , like the mission statement of Google (https://en.wikipedia.org/wiki/Don%27t_be_evil) was for decades, will not save you. Any business that uses machine learning should develop a process in order to handle ethical issues before they arrive. And ethical questions will arise!

A growing number of experts believe that a third revolution will occur during the 21st century, through the invention of machines with intelligence which surpasses our own intelligence. The rapid progress in machine learning technology turns out to be input for all kind of disaster scenarios. When the barriers to apply machine learning will be lowered more one of the fears is that knowledge work and various mental tasks currently performed by humans will become obsolete. 

When machine learning develops and the border with artificial intelligence will be hit many more philosophical and ethical discussions will take place. The core question is of course: What is human intelligence? Or to put it in the context of machine learning: What is the real value of human intelligence when machine learning algorithms can take over many common mental tasks of humans? 

Many experts believe that there is a significant chance we will develop machines more intelligent than ourselves within a few decades. This could lead to large, rapid improvements in human welfare, or mass unemployment and poverty on large scale. And yes  history learns that there are good reasons to think that it could lead to disastrous outcomes for our current societies.  If machine learning research advances without enough research work going on security, safety on privacy, catastrophic accidents are likely to occur.

With FOSS machine learning capabilities you should be able to take some control over  the rapid pace machine learning driven software is hitting our lives. So instead of trying to stop developments and use, it is better to steer developments into a positive, safe, human centric direction. So apply machine learning using a decent machine learning architecture were also some critical ethical business questions are addressed. 

Advances within machine learning could lead to extremely positive developments, presenting solutions to now-intractable global problems. But applying machine learning without good architectures where ethical questions are also addressed, using machine learning at large can pose severe risks. Humanity’s superior intelligence is the sole reason that we are the dominant species on our planet. If technology with advanced machine learning algorithms surpass humans in intelligence, then just as the fate of gorillas currently depends on the actions of humans, the fate of humanity may come to depend more on the actions of machines than our own.

To address ethical questions for your machine learning solution architecture you can use the high level framework below.


.. image:: /images/ml-ethics.png
   :width: 600px
   :alt: ML Ethics
   :align: center 


Some basic common ethical questions for every machine learning architecture are:

* Bias in data sets. How do you weight this? Are you fully aware of the impact?
* Impact on your company.
* Impact on your employees.
* Impact on your customers (short and long term).
* Impact on society.
* Impact on available jobs and future man force needed.
* Who is responsible and who is liable when the application developed using machine learning goes seriously wrong?
* Do you and your customers find it acceptable all kinds of data are combined to make more profit?
* How transparent should you inform your customers on how privacy aspects are taken into account when using the machine learning  software? Legal baselines, like the EU GDPR do not answer these ethical questions for you! 
* How transparent are you towards stakeholders regarding various direct and indirect risks factors involved when applying machine learning applications?
* Who is responsible and liable when risks in your machine learning application do occur?

A lot of ethical questions come back to crucial privacy and other risks questions like safety and security. We live in a digital world were our digital traces are everywhere. Most of the time we are fully unaware. In most western countries mass digital surveillance cameras generates great data to be used for machine learning algorithms. This can be noble by detecting diseases based on cameras, but all nasty use cases thinkable are of course also under development. Continuous track and trace of civilians including face recognition is not that uncommon any more! 


The question regarding who is responsible for negative effects regarding machine learning technology is simple to answer. You are! If you do not understand the technology, the impact for your business and on society you should not use it. 

Regulations for applying machine learning are not yet developed. Although some serious thinking is already be done in the field regarding:

* Safety and
* Liability

Government rules, laws will be formed during the transition the coming decade. Machine learning techniques are perfect to use for autonomous weapons. So drones will in near future decide based on hopefully predefined rules when to launch a missile and when not. But as with all technologies: Failures will happen! And we all hope it will not hit us.

