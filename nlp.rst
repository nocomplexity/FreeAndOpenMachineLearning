Natural Language Processing 
===========================

Introduction
-------------

A real tangible and still the most applied use case for machine learning is natural language processing (NLP). Many businesses use cases for machine learning are based on information. This can be input or processed information. In fact most business use cases that can benefit of machine learning have nothing to do with images or video. Most business do have information of customers in digital form available and want to use this data to develop more value added services for their customers. 

Everything that has to do with text processing and involves machine learning can be categorized as Natural Language Processing (NLP).

.. image:: /images/what-is-nlp.png  
   :alt: What is NLP 
   :align: center 

NLP is concerned with programming computers to process natural language. NLP is at the intersection of computer science, machine learning and linguistics. The most innovative form of NLP  are applications that use using the latest machine learning technologies to derive meaning from human languages. NLP is something that’s been part of our lives for decades. In fact, consumers from across the globe interact with NLP on a daily basis, without even realizing it. But with more FOSS machine learning building blocks available even with the latest machine learning algorithm, also NLP innovation is growing rapidly. 


NLP technology is important for scientific, economic, social, and cultural reasons. Computers that can communicate with humans as humans do are a holy grail. Including understanding context and emotions. Due to infinite number of cultures solving this problem has proven to be hard. Communication is not only verbal. Nonverbal is the significant part of communication. Nonverbal communication is the nonlinguistic transmission of information through visual, auditory, tactile, and kinesthetic (physical) channels. 

NLP is experiencing rapid growth as its theories and methods are deployed in a variety of new machine learning technologies. 

More and more NLP techniques are used products that have serious impact on our daily lives. A misinterpreted pronounced word like ‘stop’ can have several meanings when used within an autonomous driving car. Maybe you just meant to warn your children instead of stopping the vehicle on a dangerous intersection. 

Creating good NLP based applications using machine learning is hard. A simple test that gives an indication of the quality is to use a the sentence “Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo”. This sentence is correct. But many NLP algorithms and applications cannot handle this very well. Similar sentences exist in other languages. We humans are by nature good with complicated linguistic constructs, but many NLP algorithms still fail with this simple example. Some background information on this sentence can be found on Wikipedia (https://en.wikipedia.org/wiki/Buffalo_buffalo_Buffalo_buffalo_buffalo_buffalo_Buffalo_buffalo )



Basic NLP functions
--------------------

NLP is able to do all kinds of basic text functions. A short overview:

* Text Classification (e.g. document categorization).
* Finding words with the same meaning for search.
* Understanding how much time it take to read a text.
* Understanding how difficult it is to read is a text.
* Identifying the language of a text.
* Generating a summary of a text.
* Finding similar documents.
* Identifying entities (e.g., cities, people, locations) in a text.
* Translating a text.
* Text Generation.

Machine learning capabilities have proven to be very useful for improving typical NLP based use cases. This is due to the fact that text in digital format is widely available and machine learning algorithms typically perform good on large amounts of data.

NLP papers and NLP software comes with a typical terminology:

* Tokenizer: Splitting the text into words or phrases.
* Stemming and lemmatization. Normalizing words so that different forms map to the canonical word with the same meaning. For example, "running" and "ran" map to "run."
* Entity extraction: Identifying subjects in the text.
* Part of speech detection: Identifying text as a verb, noun, participle, verb phrase, and so on.
* Sentence boundary detection: Detecting complete sentences within paragraphs of text.

To create NLP enabled applications you need to set up a 'pipeline' for the various software building blocks. For each step in a NLP development pipeline another FOSS building block can be needed. The figure below shows a typical NLP pipeline. 

.. image:: /images/nlp-pipeline.png  
   :alt: Typical NLP Architecture 
   :align: center 


In the figure below a typical NLP architecture for providing input on common user questions. A lot of  Bot systems (Build–operate–transfer) or FAQ answering systems are created with no machine learning algorithms at all. Often simple keyword extraction is done and a simple match in a database is performed. More state-of-the-art NLP systems make intensive use of machine learning algorithms. The general rule is: If a application should be user friendly and value added than learning algorithms are needed, since it is no longer possible to program output based on given input.


.. image:: /images/nlp-architecture.png   
   :alt: Typical NLP Architecture 
   :align: center 

NLP Business challenges
--------------------------

When using NLP technology to extract information and insight from text, the starting point is typically the raw documents stored on websites, unstructured documents and structured documents.
Also the fact that documents are stored in a variety of formats, like PDF, MSword, TIFFs make that the time needed before text can be send towards an algorithm long and often manual intensive.
Even the most advanced web scraping techniques (software to store raw text of websites) is manual intensive. Unstructured text must be structured first before using these text for a NLP driven application is possible.

Privacy is a large concern when dealing with documents. To comply with the GDPR (in EU) using text with personal information of e.g. customers for other purposes is often not allowed without explicit permission of the owners of the personal data.



.. sub chapter with real NLP business examples

.. include:: nlp-examples.rst



