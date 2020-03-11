Machine Learning for Business Problems
=======================================

Reading and talking about futuristic potential options for machine learning is nice and should be done. But applying machine learning today for your business is where you can make a real difference. This section is focussed on applying machine learning for real business use cases. Example use cases are outlined that are possible with current available FOSS machine learning building blocks. Some real world business use cases where machine learning is applied are shown. This to give you inspiration and information on possible options for your business. But beware that besides technology more is needed for applying machine learning in a business with success. This section will also give you some more in depth input of organisational factors that should be taken into account when applying machine learning for real business use.


When to use machine learning?
------------------------------

Before starting and applying machine learning for solving business problems you must be aware that machine learning is not a tool for every problem. Or to put it even more clear: In most cases applying machine learning is overkill, too expensive, will not work and other traditional software solutions make far more sense.
So the short answer for most use case is: do not use machine learning. Keep it simple! Using machine learning is a complex and risky journey and it will make your business more complex.

But the temptation to use machine learning to solve complex problems is too promising to ignore. So you should try it. Preferably using a fast innovation project with minimal cost and no strings attached. If only try it to see if is has real some real opportunities for your use case. But be aware from the start that  machine learning doesn’t give perfect answers or a perfect solution. Risk will always exist,so you should get a feeling on the likelihood of a risk occurring. 

In some use cases machine learning can save you a lot of time and can make things possible that are out of reach using normal traditional software approaches. 

With the use of machine learning it is possible to learn from patterns and conditions to get new solid outcomes or predictions based on new data. Machine learning is able to learn from changes in patterns (data) at a pace that the human mind can not. This makes that machine learning as a technology is useful for a set of use cases were learning from data is possible or needed. So that is one reason why machine learning only makes sense for a limited class of use cases.  

Machine learning should not be used for use cases that can be easily solved in another way. 
For example do not use machine learning driven solutions if your use case matches one of the following criteria:

* If it’s possible to structure a set of rules or “if-then scenarios” to handle your problem entirely, then there is usually no need to use machine learning at all.

* Your problem can be solved using traditional statistical tools(algorithms) and software.

Machine learning is an appropriate tool to use for problems whose only commonality is that they involve statistical inference. This means that problems where machine learning makes real sense have e.g. the following characteristics:

- Classification challenges. E.g., is this a picture of a cat or a gorilla? Looks this human happy? Is this person writing emotional replies on twitter?

- Clustering challenges. E.g., group all cat pictures by ones that are most similar. 

- Reinforcement learning challenges. E.g., learn to predict how people behave when they book a holiday with large discount. Are you willing to buy something you do not need without discount?

A good question to ask is: Can this problem be solved by looking at statistical outcomes? If the answer is yes, use traditional statistical software and avoid machine learning directly. Avoid complexity at all cost before trying to find if using machine learning is a viable option.

In general: All areas where there is a lot of data *and* too much data for manual inspection are a candidate for applying machine learning.

So summarized for most business problems using machine learning should be avoided. Like blockchain or other industry IT buzzwords: Avoid the trap of using a solution and finding a problem to use it on! A particularly bad use case for machine learning is when the problem can be described using clear and precise mathematical equations. Only when a problem can not be described using clear and existing mathematical equations and an outcome can be predicted using large numbers of input data, than the use of machine learning should be considered.

When you want to apply machine learning for your business use cases you will need to develop a solid architecture before starting. A standard solution for your business use case does not exist. Your company and your context is unique. So for real and significant business advantage you should also develop your own machine learning enabled application or ML powered information system. Machine learning is just a component in the complete system architecture needed. But a good and simple overall architecture when applying machine learning is needed. Especially since all developed solutions deployed in production will need maintenance. In the section 'ML Reference Architecture' a view of the complete system architecture is given.

The usage of a Cloud (SaaS or ML-SaaS) machine learning solution will not always give you the competitive advantage you are searching for. This beceause standard solutions will only work on standard use cases. Most use cases are unique.  So if your business is special, your data is unique and your use case is unique than your own developed machine learning driven application should give you a head start and competitive advantage. In the section ‘ Machine learning reference architecture’  an in depth outline is given on the various system building blocks that are needed for applying machine learning in a successful way. Make use of the machine learning reference architecture outlined in this publication to create your own ML enabled solution faster.

In order to solve business problems using machine learning technology  you need to have an organisation structure that powers innovation and experimenting. Experimenting with machine learning should be simple and can be done in a short time. But this requires a business culture with an innovation approach where learning and playing with new technology is possible without without predefined rules.


Common business use cases
-----------------------------

Healthcare
^^^^^^^^^^^^

Healthcare is due to the large amounts of data available a perfect domain for solving challenges using machine learning. E.g. a challenging question for machine learning for healthcare is: Given a patient’s electronic medical record data, can we prevent a person getting sick?

Machine learning is more and more used for automatic diagnostics. This can be data provided by X-ray scans or data retrieved from blood and tissue samples. Machine learning has already proven to be valuable in detecting and predicting diseases for real people. But beware already sensors and camera data in public spaces are used to gather data, also for healthcare related use cases without your approval. 

Predictive tasks for healthcare is maybe the way to keep people healthier and lower healthcare cost. The transformation from making people better towards preventing people getting sick will be long and hard, since this will be a real shift for the healthcare industry.

But given a large set of training data of de-identified medical records it is already possible to predict interesting aspects of the future for a patient that is not in training set.

Machine learning applications for healthcare are also to create better medicines by making use of all the data already available.

Language translation
^^^^^^^^^^^^^^^^^^^^^^^

Machine learning is already used for automatic real-time message translation. E.g. Rocket Chat (The OSS Slack alternative, https://rocket.chat/ ) is using machine learning for real time translation.

Since language translation needs context and lots of data, typically these use cases are often NLP driven.
Language translation as speech recognition is a typical NLP application. Natural language processing (NLP) is area of machine learning that operates on (human)text and speech. See the section on NLP in this book for more use cases and insight in the specific NLP technologies.

Other areas for language translation are speech recognition. Some great real time machine learning driven application already exist.   

When building speech recognition machine learning applications you will find out that data needed for speech recognition is not quite open. To create voice systems you need an extremely large amount of voice data. Most of the data used by large companies isn’t available to the majority of people. E.g. Amazon , Microsoft and Google offer great APIs but you interact with a black-box model. Also speech recognition needs openness and freedom.  Mozilla launched Common Voice project in 2017. A project to make voice recognition data and APIs open and accessible to everyone. Contributing to this great project is simple: Go to https://voice.mozilla.org/ and speak some sentences and validate some. All you need is a browser and a few minutes to contribute so everyone can make use of this technology in the future. 


Chat bots
^^^^^^^^^^

Currently all major tech companies like Amazon(Alexis), Google, Apple (Siri) have built a smart chatbot for the consumer market. Creating a chatbot (e.g. IRQ bot) was not new and difficult, however building a real ‘intelligent’ chat bot that has learning capabilities is another challenge. 
 
Machine learning powered chatbots with real human like voices will help computers communicate with humans. But algorithms still have a hard time trying to figure out what you are saying, because context and tone of voice are hard to get right. Even for us humans, communication with other humans is most of the time hard. So building a smart chatbot that understands basic emotions in your voice is difficult. Machine learning isn’t advanced enough yet to carry on a dialogue without help, so a lot of the current chat bot software needs to be hand-coded. 



eCommerce Recommendation systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A well known application for machine learning for eCommerce systems is a machine learning enabled recommendation system system. Whether you buy a book, trip, music or visit a movie: On all major online ecommerce sites you get a recommendation for a product that seems to fit your interest perfectly. Of course the purpose is to drive up the sale, but these algorithms used are good examples of still evolving machine learning algorithms for recommendation systems.

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

Financial services
^^^^^^^^^^^^^^^^^^^^

* Real-time trade: Like bidding sites for online advertising and stock exchange markets are more and more driven by software algorithms. Knowing this you must be a fool if you take part on a stock exchange market without the power these automated machine learning driven systems have. You will never earn anything...

* Banking and credit services: More and more banks and large financial companies are using their data to get more profit out of existing customers. Based on a smart combination of banking data and financial transaction data, your bank will know better than you can image how to make even more profit from their customers. 

Marketing
^^^^^^^^^^^

* Marketing and acquisition: By analysing mass amounts of data you can better target your existing and potential users for your service. Machine learning makes a large difference here, as proven by Google and Facebook (both ad-service companies in essence). Analysing works using machine learning works well for consumer markets where user data and user behaviour data is widespread and for sale. And since tracking users on the internet is the number one dataleak almost all data is available somewhere. Also business to business marketing is perfect to automate using machine learning. This because also here the only input needed is often data.

Of course if you do care about privacy and embrace the values of Free and Open machine learning the marketing use cases for machine learning are almost impossible to create due to privacy issues involved.

HR services
^^^^^^^^^^^^^

* HR management and HR services: Finding the right new employing, talent management, performance management all tangible HR work will be powered by ML software more and more. Even the scary face and voice recognition tools will be used to check if your new employee matches your ideal profile. Until HR is fully automated ML powered software will help HR professionals to improve decision-making and create more efficient ways to interact with employees.

When using machine learning for HR services be aware of bias issues in using datasets. Bias when hire new personal is for humans already difficult to handle. But you do not want a machine learning application that only selects people based on old paradigms in society.

Predicting services
^^^^^^^^^^^^^^^^^^^^

* Predicting services: Almost all predicting services in all business domains can benefit from the combination of large data sets and using machine learning algorithms. E.g. you can empower predicting services by using weather data (historical and new), financial and demographic data and local production data to find out in more detail how your next sales campaign will go. But prediction is also possible on failures on production lines, where historical data is combined with sensor data.

Software
^^^^^^^^^^^^
A holy grail for software developers is of course creating a machine learning algorithms that creates software for use cases that require expensive and complex human programming work.

The last years some real progress for using machine learning for creating software is made. Use cases seen are e.g.:

* Software code improvement: Manual programming is hard and error prone. By training machine learning on a large code base to learn the model what ‘bugs’ are, it is possible to use machine learning to prevent programming bugs in new developed software code. In this way code can not be committed since the automated checks provided spotted an error. Detecting a bug before software is tested and deployed is far cheaper than correcting errors in code when a program is already released. A game development company  has used this application of machine learning for real with success already. (reference http://www.wired.co.uk/article/ubisoft-commit-assist-ai )

* Creating new software programs: Based on a problem it is proven by different companies that software can be generated in stead of manual crafted (programmed). Feeding a algorithm massive inputs of examples programs it is possible to generate a new program based for your specific problem. Of course this application of machine learning is still in early phase. It is also questionable if this application of machine learning makes real sense since the new paradigm of machine learning is no longer program a solution but create a program outcome based on input data.


Security
^^^^^^^^^^

* Email spam filters. Although simple rules can and should be applied, the enormous creativity of spammers and the amount send good fighting spam is a solid use case for a supervised machine learning problem. 
* Network filtering. Due to the learning capability of machine learning network security devices are improved using machine learning techniques.
* Fraud detection. Fraud detection is possible using enormous data and searching for strange patterns.

Besides fraud detection machine learning can also applied for IT security detections since intrusion detection systems and virus scanners are more and more shipped with self learning algorithms. Also Complex financial fraud schemes can be easily detected using predictive machine learning models.

Privacy
^^^^^^^^^

Privacy can be protected using machine learning. E.g. images can be made invisible by using a machine learning enabled application. A scientific proof is demonstrated and the code is named ‘DeepPrivacy’. In the section with a collection of Computer Vision Building Blocks for more information on this SBB.) The technique used is based on Generative Adversarial Network (GAN) for face anonymization. It's far from perfect, but usable for most low quality images.

.. warning::

  Besides protecting privacy machine learning is still too often a privacy nightmare.




Risk and compliance
^^^^^^^^^^^^^^^^^^^^

* Evaluating risks can be done using large amounts of data. Natural language processing techniques can be used to validate highly automatic if your company meets regulations. Since audit and inspecting work is mostly based on standardized rules performed by knowledge workers this kind of work can be automated using machine learning techniques.

* Detecting danger and safety risks. E.g. for autonomous vehicles (robots). More and more machine learning software is developed to make transport safer for us humans.



.. sub chapter with real exiting business examples

.. include:: business-examples.rst



Business Challenges
--------------------

Applying machine learning for real business use cases is complex and difficult.  

Common business challenges when applying machine learning in business products or services are e.g.:

- Determining when applying machine learning is a good choice for solving a business problem.

- Getting the right data and preparation of the data to be used for training a machine learning model.

- Dealing with privacy, security and safety aspects.

- Engineering solid and maintainable machine learning applications. Designing, creating and debugging machine learning applications is specialized IT work.

- Dealing with terrible math and statistics foundations. Of course most software building blocks will keep this away from you, but you must make choices that require some more in depth knowledge of the foundations behind the chosen algorithms used. 

- Have access skilled IT engineers. Not only machine learning engineers are needed, but also good engineers that are skilled in setting up IT environments. This accounts for cloud and also for on premise environments. Choices that are possible for machine learning cloud environments are often not trivial, unless you have an unlimited credit card. 

The number one challenge is: How to integrate machine learning into your current business operations and products in order to really benefit from this technology?

Normal IT projects have a bad reputation. Projects are often delayed and do not deliver what was needed. Machine learning are still not different. In fact machine learning projects are still complex and risky IT projects. So an agile approach is recommended to reduce risks. 

Integration of machine learning software pipelines, especially when it also involves digital integration between companies and systems of different companies is known to be hard, complex and will make you poor if handled wrong. If you have a bad track record when it comes to executing traditional IT projects, machine learning projects will have the same challenges with a couple of new real high risks elements.

Machine is not a logical and intuitive way to solve problems. For many engineers and software programmers solving problems using a machine learning approach is against the learned and trained intuition. So training and building an intuition for what tool should be leveraged to solve a problem is needed for engineers involved. At a minimum engineers involved should be aware of available machine learning algorithms and machine learning building blocks (SBBs) and the trade-offs and constraints of each one. This publication contains an overview of the typical algorithms and an overview of diverse ML FOSS building blocks available. This will increase the insights and improves the awareness of available options. 


Machine learning needs trial and error before it works well. But debugging a machine learning application is a real complex challenge. An endless number of factors must be taken into account. Not only technical but even more from a business perspective. When are risks in outcomes acceptable? You need insights in the context where the results are used in order to evaluate if machine learning results are usable enough. When you want to improve the output you can face problems e.g. the following problems:

- Is there a bug in the used software framework?

- Is the data quality below an acceptable level?

- Is the chosen algorithm the right choice?

- Are other IT issues influencing the outcome, e.g. performance?

- With machine learning finding bugs and working on optimizations is almost ‘exponentially’  harder due to the complex nature of the various aspects involved. So to figure out what is wrong when things don’t work as expected can take far more time than available.

- Are the risks for business use acceptable? For live saving systems your will make other choices than for a marketing system. 


Business capabilities 
-----------------------

To take advantage of machine learning your organisation needs to have or develop the needed capabilities.
Before starting a proof of concept or project with machine learning you need to dive into the subject and options. *Warning*: Don’t fall for a vendor hype. So beware of demo's and courses of vendors who sell you perfect SaaS ML solutions. If a promise for new business innovation based on a new machine learning application seems too good to be true: It often is. 

Only you know your business systems, your requirements, your financial objectives, your customers and thus the right trade-offs to make. Good and simple tools can make the process for using machine learning easier. But tools are no magic bullet. You still need to have to integrate machine learning outcomes with your business products or services. 

To use the power of machine learning collaboration is needed. So the focus should also be on solving business problems and not only IT challenges.

The following capabilities are often needed to successfully apply machine learning for your business use case:

- Capability to experiment and learn. So a real learning culture. 

- Managers, architects, developers and engineers with an open mindset. So open for learning and experimenting.

- Descent knowledge of key quality aspects involved. E.g. privacy, safety and security. A must take these privacy, safety and security serious from the start. Do it by design. It can initially take some extra time, but once key safeguards are in place experimenting with data and machine learning outcomes will be possible with lower risks. So make sure you will also have some privacy and security experts involved from the start.

- Solid business innovation strategy, innovation management system (process and people) available. 

If your goal is to use machine learning to reduce cost by automating human workflows make sure everyone shares this goal upfront. 



Business ethics
---------------------

When machine learning algorithms make decisions that affect human lives, what standards of transparency, openness and accountability should apply to those decisions? If the decisions are "wrong", who is legally and ethically responsible?

There are always good and bad uses for any technology. This accounts also for machine learning technology. Working with machine learning can, will and must raise severe ethical questions. Machine learning can be used in many bad ways. Saying that you ‘Don't be evil’ , like the mission statement of Google (https://en.wikipedia.org/wiki/Don%27t_be_evil) was for decades, will not save you. Any business that uses machine learning should develop a process in order to handle ethical issues before they arrive. And ethical questions will arise.


A growing number of experts believe that a third revolution will occur during the 21st century, through the invention of machines with intelligence which surpasses our own intelligence. The rapid progress in machine learning technology turns out to be input for all kind of disaster scenarios. When the barriers to apply machine learning will be lowered one of the fears is that knowledge work and various mental tasks currently performed by humans will become obsolete. 

When machine learning develops and the border with artificial intelligence will be approached many more philosophical and ethical discussions will take place. One of the core question is: What is human intelligence? But the more important question is: Who is responsible for mistakes? The self learning algorithm? To put it in the context of machine learning: What is the real value of human intelligence when machine learning algorithms can take over many common mental tasks and control tasks of humans? Who is responsible for accidents with autonomous vehicles?

Many experts believe that there is a significant chance we will develop machines more intelligent than ourselves within a few decades. This could lead to large, rapid improvements in human welfare, or mass unemployment and poverty on large scale. History learns that there are good reasons to think that this could lead to disastrous outcomes for our current societies.  If machine learning research advances without enough research work going on security, safety on privacy, catastrophic accidents are likely to occur. Or if we look back at history: Incidents will occur since regulations are always developed afterwards with new technology.

With FOSS machine learning capabilities you should be able to take some control over the rapid pace machine learning driven software is hitting our lives. So instead of trying to stop developments and use, it is better to steer developments into a positive, safe, human centric direction. So apply machine learning using a decent machine learning architecture were also some critical ethical business questions are addressed. 

Advances within machine learning could lead to extremely positive developments, presenting solutions to now-intractable global problems. But applying machine learning without good architectures where ethical questions are also addressed, using machine learning at large can pose severe risks. Humanity’s superior intelligence is the sole reason that we are the dominant species on our planet. If technology with advanced machine learning algorithms surpass humans in intelligence, then just as the fate of gorillas currently depends on the actions of humans, the fate of humanity may come to depend more on the actions of machines than our own.

To address ethical questions for your machine learning solution architecture you can use the high level framework with ethical requirements below. All requirements are of equal importance, support each other, and should be implemented and evaluated throughout the system’s lifecycle.


.. image:: /images/ml-ethical-requirements.png
   :alt: ML Ethics requirements
   :align: center 

The framework of ethical requirements is part of the (draft)'Ethics Guidelines for Trustworthy Artificial Intelligence (AI)' from the Expert Group on Artificial Intelligence (AI HLEG)of the European Commission (https://ec.europa.eu/futurium/en/ai-alliance-consultation). 


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

The question regarding who is accountable for negative effects when you use machine learning technology is simple to answer. You are! 
Accountability is about holding individuals and organisations responsible for how any ML enabled application is being used. But this is not trivial: The outcome of a machine learning application system will not simply be the product of the software itself, or any single decision-maker. This is because the success or failure of a ML enabled system may be the product of one or several components. In most cases, a system failure will be the result of multiple factors, and responsibility will not be easily apportioned. 
So: If you do not understand the technology, the impact for your business and on society you should not use it. 


Regulations for applying machine learning are not yet developed. Although some serious thinking is already be done in the field regarding:

* Safety and
* Liability

Many governmental bodies promote adopting a risk-adapted regulatory approach when it comes to ethical issues regarding algorithmic systems (machine learning). History learns that risks based approaches that depend on human discipline, especially in areas where safety issues are clear, are fuel for disasters waiting to happen. It makes more sense to adopt an approach that bans the human factor and risks can be calculated using long proven scientific statistical methods. 

Government rules and laws will be formed during the transition the coming decade. Machine learning techniques are perfect to use for autonomous weapons. So drones will in near future decide based on hopefully predefined rules when to launch a missile and when not. But as with all technologies: Failures will happen! And we all hope it will not hit us.

Using machine learning comes with responsibilities. These responsibilities apply for all institutions that fund, develop, and deploy ML based systems.
