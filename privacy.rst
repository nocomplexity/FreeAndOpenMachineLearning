Privacy
--------------

Machine learning raises serious privacy concerns since machine learning is using massive amounts of data that contain often personal information. 

It is a common believe that personal information is needed for experimenting with machine learning before you can create good and meaningful applications. E.g. for health applications, travel applications, eCommerce and of course marketing application. Machine learning models are often loaded with massive amounts of personal data for training and to make in the end good meaningful predictions. 

The belief that personal data is needed for machine learning creates a tension between developers and privacy aware consumers. Developers want the ability to create innovative new products and services and need to experiment, while consumers and GDPR regulators are concerned for the privacy risks involved.

The applicability of machine learning models is hindered in settings where the risk of data leakage raises serious privacy concerns. Examples of such applications include scenar-
ios where clients hold sensitive private information, e.g., medical records, financial data, or location.

It is commonly believed that individuals must provide a copy of their personal information in order for AI to train or predict over it. This belief creates a tension between developers and consumers. Developers want the ability to create innovative products and services, while consumers want to avoid sending developers a copy of their data.

Machine learning models can be trained in environments that are not secure on data it never has access to. Secure machine learning that works on anonymized data sets is still an obscure and unpaved path. But some companies and organizations are already working on creating deep learning technology that work on encrypted data. Using encryption on data to train machine learning models raises the complexity in various ways. It is already hard to get inside the ‘black-box’ of the working of machine learning. Using advanced data encryption will require even more knowledge and competences for all engineers involved when developing machine learning applications. 

In the EU the use of personal data is protected by law in all countries by a single law. The EU General Data Protection Regulation (GDPR). This GDPR does not prohibit the use of machine learning. But when you use personal data you will have a severe challenge to explain to DPOs (Data Protection Officers) and consumers what you actually do with the data and how you comply with the GDPR. 

Machine learning systems must be data responsible. They should use only what they need and delete it when it is no longer needed (“data minimization”). They should encrypt data in transit and at rest, and restrict access to authorized persons (“access control”). Machine learning systems should only collect, use, share and store data in accordance with privacy and personal data laws and best practices. Since FOSS machine learning needs full transparency and reproducibility using private data should be avoided if possible.

When you apply machine learning for your business application you should consider the following questions:

* In what way will your customers be happy with their data usage for their and your benefit?
* Do you really have a clear and good overview of all GDPR implications when using personal data in your machine learning model? What happens if you invite other companies to use your model? 
* What are the ethical concerns when using massive amounts of data of your customers to develop new products? Is the way you use the data to train your model congruent with you business vision and moral?
* What are the privacy risks involved for your machine learning development chain and application?

Since security and privacy is complex to apply, frameworks are being developed to make this challenge easier. E.g. Tensorflow Encrypted aims to make privacy-preserving deep learning simple and approachable, without requiring expertise in cryptography, distributed systems, or high-performance computing. And PySyft is a Python library for secure, private Deep Learning. More on both frameworks can be found in the section on open ML software. 
