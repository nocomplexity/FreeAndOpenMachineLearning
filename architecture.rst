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

Development tools
^^^^^^^^^^^^^^^^^^^^

In order to apply machine learning you need good tools to do e.g.:

* Create experiments for machine learning fast.
* Create a solid solution architecture
* Create a data architecture
* Automate repetitive work (integration, deployment, monitoring etc)

Fully integrated tools that cover all aspects of your development process (business design and software and system design) are hard to find. Even in the OSS world. 
Many good architecture tools, like Arch for creating architecture designs are still usable and should be used. A good overview for general open architecture tools can be found here https://nocomplexity.com/architecture-playbook/.  
Within the machine learning domain the de facto development tool is ‘The Jupyter Notebook’. The Jupyter notebook is an web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text. A Jupyter notebook is perfect for various development steps needed for machine learning suchs as data cleaning and transformation, numerical simulation, statistical modeling, data visualization and testing/tuning machine learning models.
More information on the Jupyter notebook can be found here https://jupyter.org/ .


But do not fall in love with a tool too soon. You should be confronted with the problem first, before you can evaluate what tool makes your work more easy for you.


Data
^^^^^^

Data is the heart of the machine earning and many of most exciting models don’t work without large data sets. Data is the oil for machine learning. Data is transformed into meaningful and usable information. Information that can be used for humans or information that can be used for autonomous systems to act upon.

In normal architectures you make a clear separation when outlining your data architecture. Common view points for data domains are: business data, application data and technical data For any machine learning architecture and application data is of utmost importance. Not all data that you use to train your machine learning model needs can be originating from you own business processes. So sooner or later you will need to use data from other sources. E.g. photo collections, traffic data, weather data, financial data etc. Some good usable data sources are available as open data sources. 
For a open machine learning solution architecture it is recommended to strive to use open data. This since open data is most of the time already cleaned for privacy aspects. Of course you should take the quality of data in consideration when using external data sources. But when you use data retrieved from your own business processes the quality and validity should be taken into account too. 

Free and Open Machine learning needs to be feed with open data sources. Using open data sources has also the advantage that you can far more easily share data, reuse data, exchange machine learning models created and have a far easier task when on and off boarding new team members. Also cost of handling open data sources, since security and privacy regulations are lower are an aspect to take into consideration when choosing what data sources to use.

For machine learning you will need ‘big data’. Big data is any kind of data source that has one the following properties:

* Big data is data where the volume, velocity or variety of data is (too) great.So big is really a lot of data! 
* The ability to move that data at a high Velocity of speed.
* An ever-expanding Variety of data sources.
* Refers to technologies and initiatives that involve data that is too diverse, fast-changing or massive for conventional technologies, skills and infra- structure to address efficiently.


Every Machine Learning problem starts with data. For any project most of the time large quantities of training data are required. Big data incorporates all kind of data, e.g. structured, unstructured, metadata and semi-structured data from email, social media, text streams, images, and machine sensors (IoT devices).

Machine learning requires the right set of data that can be applied to a learning process. An organization does not have to have big data in order to use machine learning techniques; however, big data can help improve the accuracy of machine learning models. With big data, it is now possible to virtualise data so it can be stored in the most efficient and cost-effective manner whether on- premises or in the cloud.

Within your machine learning project you will need to perform data mining. The goal of data mining is to explain and understand the data. Data mining is not intended to make predictions or back up hypotheses. 

One of the challenges with machine learning is to automate knowledge to make predictions based on information (data). For computer algorithms everything processed is just data. Only you know the value of data. What data is value information is part of the data preparation process. Note that data makes only sense within a specific context. 

The more data you have, the easier it will be to apply machine learning for your specific use case.  With more data, you can train more powerful models. 

Some examples of the kinds of data machine learning practitioners often engage with:

* Images: Pictures taken by smartphones or harvested from the web, satellite images, photographs of medical conditions, ultrasounds, and radiologic images like CT scans and MRIs, etc.
* Text: Emails, high school essays, tweets, news articles, doctor’s notes, books, and corpora of translated sentences, etc.
* Audio: Voice commands sent to smart devices like Amazon Echo, or iPhone or Android phones, audio books, phone calls, music recordings, etc.
* Video: Television programs and movies, YouTube videos, cell phone footage, home surveillance, multi-camera tracking, etc.
* Structured data: Webpages, electronic medical records, car rental records, electricity bills, etc
* Product reviews (on Amazon, Yelp, and various App Stores)
* User-generated content (Tweets, Facebook posts, StackOverflow questions)
* Troubleshooting data from your ticketing system (customer requests, support tickets, chat logs)


When developing your solution architecture be aware that data is most of the time:

* Incorrect and
* useless.

So meta data and quality matters. Data only becomes valuable when certain minimal quality properties are met. For instance if you plan to use raw data for automating creating translating text you will soon discover that spelling and good use of grammar do matter. So the quality of the data input is an import factor of the quality of the output. E.g. automated Google translation services still struggle with many quality aspects, since a lot of data captures (e.g. captured text documents or emails) are full of style,grammar and spell faults.


Data science is a social process. Data is generated by people within a social context. Data scientists are social people who will have to do a lot of communication with all kind of business stakeholders. Data scientist should not work in isolation because the key thing is to find out what story is told within the data set and what import story is told over the data set.  


Data Tools
^^^^^^^^^^^^^

Without data machine learning stops. For machine learning you will be dealing with large complex data sets (maybe even big data) and the only way to make machine learning applicable is data cleaning and preparation. So  you need good tools to handle data.

The number of tools you will need will depend of the quality of your data sets, your experience, development environment and other choice you will have to make in your solution architecture. But a view use cases where good solid data tools will help are:

* Data visualization and viewer tools; Good data exploration tools give visual information about the data sets without a lot of custom programming.
* Data filtering, data transformation and data labelling;
* Data anonymiser tools;
* Data encryption / decryption tools
* Data search tools (analytics tools)

Without good data tools you are lost when doing machine learning for real. The good news is: There are a lot of OSS data tools you can use. Depending if you have raw csv, json or syslog data you will need other tools to prepare the dataset. The challenge is to choose tools that integrate good in your landscape and save you time when preparing your data for starting developing your machine learning models. Since most of the time when developing machine learning applications you will be fighting with data, it is recommended to try multiple tools. Most of the time you will learn that a mix of tools is the best option, since a single data tool will never cover all your needs. So leave some freedom within your architecture for your team members who will be dealing with data work (cleaning, preparation etc).

The field of ‘data analytics’ and ‘business intelligence’ is a mature field for decades within IT. So you will find many tools that are excellent for data analytics and/or reporting. But keep in mind that the purpose of fighting with data for machine learning is in essence only for data cleaning and feature extraction. So be aware of ‘old’  tools that are rebranded as new data science tools for machine learning. There is no magic data tool preparation of data for machine learning. Sometimes old-skool unix tool like awk or sed will just do the job. 

Besides tools that assist you with preparing the data pipeline, there are also good (open) tools for finding open datasets that you can use for your machine learning application. See the reference section for some tips.

To prepare your data working with the data within your browser seems a nice idea. You can visual connect data sources and e.g. create visuals by clicking on data. Or inspecting data in a visual way. There is however one major drawback: Despite the great progress made on very good and nice looking JavaScript frameworks for visualization, handling data within a browser DOM is and will take your browser over the limit. You can still expect hangups, indefinitely waits and very slow interaction. At least when not implemented well. But implementation of on screen data visualisation (Drag-and-Drop browser based) is requires an architecture and design approach that focus on performance and usability from day 1. Unfortunately many visual web based data visualization tools use an generic JS framework that is designed from another angle. So be aware that if you try to display all your data, it will eat all your resources(cpu, memory) and you will get a lot of frustration. So most of the time using a Jupyter Notebook will be a safe choice when preparing your data sets. 

