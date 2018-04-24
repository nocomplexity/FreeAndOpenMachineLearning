Machine learning reference architecture
==========================================

When you are going to apply machine learning for your business for real you should develop a solid architecture. A good architecture covers all crucial concerns like business concerns, data concerns, security and privacy concerns. And of course a good architecture should address technical concerns in order to minimize the risk of instant project failure.

Unfortunately it is still not a common practice for many companies to share architectures as open access documents. So most architectures you will find are more solution architectures published by commercial vendors. 

Architecture is a minefield. And creating a good architecture for new innovative machine learning systems and applications is an unpaved  road. Architecture is not by definition high level and sometimes relevant details are of the utmost importance. But getting details of the inner working on the implementation level of machine learning algorithms can be very hard. So a reference architecture on machine learning should help you in several ways. 

Unfortunately there is no de-facto single machine learning reference architecture. Architecture organizations and standardization organizations are never the front runners with new technology. So there are not yet many mature machine learning reference architectures that you can use. You can find vendor specific architecture blueprints, but these architecture mostly lack specific architecture areas as business processes needed and data architecture needed. Also the specific vendor architecture blueprints tend to steer you into a vendor specific solution. What is of course not always the most flexible and best fit for your business use case in the long run. 

In this section we will describe a (first) version of an open reference architecture for machine learning. Of course this reference architecture is an open architecture, so open for improvements and discussions. So all input is welcome to make it better! See section  :ref:`Help <Help>`. 

The scope and aim of this open reference architecture for machine learning is to enable you to create better and faster solution architectures and designs for your new machine learning driven systems and applications. 

You should also be aware of the important difference between:

* Architecture building Blocks and
* Solution building blocks

.. image:: /images/abb-sbb.png
   :width: 600px
   :alt: ML Architecture Building Blocks vs SBBs
   :align: center 


This reference architecture for machine learning describes architecture building blocks. So you could use this reference architecture and ask vendors for input on for delivering the needed solution building blocks. However in another section of this book we have collected numerous great FOSS solution building blocks so you can create an open architecture and implement it with FOSS solution building blocks only. 


Before describing the various machine learning architecture building blocks we briefly describe the machine learning process. This because in order to setup a solid reference architecture high level process steps are crucial to describe the most needed architecture needs. 

Applying machine learning for any practical use case requires beside a good knowledge of machine learning principles and technology also a strong and deep knowledge of business and IT architecture and design aspects. 

The machine learning process
------------------------------

Setting up an architecture for machine learning systems and applications requires a good insight in the various processes that play a crucial role. 
So to develop a good architecture you should have a solid insight in:

* The business process in which your machine learning system or application is used.
* The way humans interact or act (or not) with the machine learning system.
* The development and maintenance process needed for the machine learning system.
* Crucial quality aspects, e.g. security, privacy and safety aspects.

In its core a machine learning process exist of a number of typical steps. These steps are:

* Determine the problem you want to solve using machine learning technology
* Search and collect training data for your machine learning development process.
* Select a machine learning model
* Prepare the collected data to train the machine learning model
* Test your machine learning system using test data
* Validate and improve the machine learning model. Most of the time you will need to search for more training data within this iterative loop.


.. image:: /images/ml-process.png
   :width: 600px
   :alt: Machine Learning Process
   :align: center 

You will need to improve your machine learning model after a first test. Improving can be done using more training data or by making model adjustments. 


Machine Learning Architecture Building Blocks
-----------------------------------------------

This reference architecture for machine learning gives guidance for developing solution architectures where machine learning systems play a major role. Discussions on what a good architecture is, can be a senseless use of time. But input on this reference architecture is always welcome. This to make it more generally useful for different domains and different industries. Note however that the architecture as described in this section is technology agnostics. So it is aimed at getting the architecture building blocks needed to develop a solution architecture for machine learning complete. 

Every architecture should be based on a strategy. For a machine learning system this means an clear answer on the question: What problem must be solved using machine learning technology? Besides a strategy principles and requirements are needed. 

The way to develop a machine learning architecture is outlined in the figure below.

.. image:: /images/ml-architecture-process.png
   :width: 600px
   :alt: Machine Learning Architecture Process
   :align: center 


In essence developing an architecture for machine learning is equal as for every other system. But some aspects require special attention. These aspects are outlined in this reference architecture.

Principles are statements of direction that govern selections and implementations. That is, principles provide a foundation for decision making.

Principles are commonly used within business design and successful IT projects. A simple definition of a what a principle is:

* A principle is a qualitative statement of intent that should be met by the architecture.



The key principles that are used for this reference machine learning architecture are:

1. The most important machine learning aspects must be addressed.
#. The quality aspects: Security, privacy and safety require specific attention.
#. The reference architecture should address all architecture building blocks from development till hosting and maintenance.
#. Translation from architecture building blocks towards FOSS machine learning solution building blocks should be easily possible.
#. The machine learning reference architecture is technology agnostics. The focus is on the outlining the conceptual architecture building blocks that make a machine learning architecture. 

By writing down these principles is will be easier to steer discussions on this reference architecture and to improve this machine learning architecture. 

Machine learning architecture principles are used to translate selected alternatives into basic ideas, standards, and guidelines for simplifying and organizing the construction, operation, and evolution of systems.

Important concerns for this machine learning reference architecture are the aspects:

* Business aspects (e.g capabilities, processes, legal aspects, risk management)
* Information aspects (data gathering and processing, data processes needed)
* Machine learning applications and frameworks needed (e.g. type of algorithm, easy of use)
* Hosting (e.g. compute, storage, network requirements but also container solutions)
* Security, privacy and safety aspects
* Maintenance (e.g. logging, version control, deployment, scheduling)
* Scalability, flexibility and performance 



.. image:: /images/ml-reference-architecture.png
   :width: 600px
   :alt: Machine Learning Architecture Building Blocks
   :align: center 

Conceptual overview of machine learning reference architecture 

Since this simplified machine learning reference architecture is far from complete it is recommended to consider e.g. the following questions when you start creating your solution architecture where machine learning is part of:

* Do you just want to experiment and play with some machine learning models? 
* Do you want to try different machine learning frameworks and libraries in to discover what works best for your use case? Machine learning systems never work directly. You will need to iterate, rework and start all over again. Its innovation!
* Is performance crucial for your application? 
* Are human lives direct or indirect dependent of your machine learning system?


In the following sections a more in depth description of the various machine learning architecture building blocks is given. 


Business Processes
^^^^^^^^^^^^^^^^^^^^^^^^

To apply machine learning with success it is crucial that the core business processes of your organization that will be affected with this new technology are determined. In most cases secondary business processes will benefit more than primary processes. Think of marketing, sales and quality aspects that make your primary business processes better.

Business Services
^^^^^^^^^^^^^^^^^^

Business services are services that your company provides to customers, both internally and externally. When applying machine learning for business use you should create a map to outline what services are impacted, changed or disappear when using machine learning technology. Are customers directly impacted or will your customer experience indirect benefits?

Business Functions
^^^^^^^^^^^^^^^^^^^
A business function delivers business capabilities that are aligned to your organization, but not necessarily directly governed by your organization. For machine learning it is crucial that the information that a business function needs is known. Also the quality aspects of this information should be taken into account. To apply machine learning it is crucial to know how information is exactly processes and used in the various business functions.

People, Skills and Culture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Machine learning needs a culture where experimentation is allowed. When you start with machine learning you and your organization need to build up knowledge and experience. Failure will happen and must be allowed. Fail hard and fail fast. Take risks. However your organization culture should be open to such a risk based approach. IT projects in general fail often so doing an innovative IT project using machine learning will be a risk that must be able to cope with. 
To make a shift to a new innovative experimental culture make sure you have different types of people directly and indirectly involved in the machine learning project. Also make use of good temporary independent consultants. So consultants that have also a mind set of taking risks and have an innovative mindset. Using consultants for machine learning of companies who sell machine learning solutions as cloud offering do have the risk that needed flexibility in an early stage is lost. Also to be free on various choices make sure you are not forced into a closed machine learning SaaS solution too soon.
Since skilled people on machine learning with the exact knowledge and experience are not available you should use creative developers. Developers (not programmers) who are keen on experimenting using various open source software packages to solve new problems. 


Business organization
^^^^^^^^^^^^^^^^^^^^^^

Machine learning experiments need an organization structure that does not limit creativity. In general hierarchical organizations are not the perfect placed where experiments and new innovative business concepts can grow. Applying machine learning in an organization requires an organization that is data and IT driven. A perfect blueprint for a 100% good organization structure does not exist, but flexibility, learning are definitely needed. Depending on the impact of the machine learning project you are running you should make sure that the complete organization is informed and involved whenever needed. 

Partners
^^^^^^^^^^^
Since your business is properly not Amazon, Microsoft or Google you will need partners. Partners should work with you together to solve your business problems. If you select partners pure doing a functional aspect, like hosting, data cleaning ,programming or support and maintenance you will miss the needed commitment and trust. Choosing the right partners for your machine learning project is even harder than for ordinary IT projects, due to the high knowledge factor involved. Some rule of thumbs when selecting partners:
Big partners are not always better. With SMB partners who are committed to solve your business challenge with you governance structures are often easier and more flexible.
Be aware for vendor lock-ins. Make sure you can change from partners whenever you want. So avoid vendor specific and black-box approaches for machine learning projects. Machine learning is based on learning, and learning requires openness.
Trust and commitment are important factors when selecting partners. Commitment is needed since machine learning projects are in essence innovation projects that need a correct mindset.
Use the input of your created solution architecture to determine what kind of partners are needed when. E.g. when your project is finished you need stability and continuity in partnerships more than when you are in an innovative phase.


Risk management
^^^^^^^^^^^^^^^^^^

Running machine learning projects involves risk. Within your architecture it is crucial to address business and projects risks early. Especially when security, privacy and safety aspects are involved mature risks management is recommended. To make sure your machine learning project is not dead at launch, risk management requires a flexible and create approach for machine learning projects. Of course when your project is more mature openness and management on all risks involved is crucial. To avoid disaster machine learning projects it is recommended to create your:

* solution architecture using:
* Safety by design principles.
* Security by design principles and
* Privacy by design principles

In the beginning this will slow down your project, but doing security/privacy or safety later as ‘add-on’ requirements is never a real possibility and will take exponential more time and resources. 

