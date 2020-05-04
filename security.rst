Security
-----------

Using machine learning technology gives some serious new threads. More and more new ways for exploiting the technology are published. IT security is proven to be hard and complex to control and manage. But machine learning technology makes the problem of IT security even worse. This is due to the fact that the special created machine learning exploits are very hard to determine.

Machine learning challenges many current security measurements. This because machine learning software:

* Lowers the cost of applying current known attacks on all devices which depend on software. So almost all modern technology devices. 
* Machine learning software enables the easy creation of new threats and vulnerabilities on existing systems. E.g. you can take the CVE security vulnerability database (https://www.cvedetails.com/) and train a machine learning model how to create attack on the published omissions. 
* When machine learning software will be in hospitals, traffic control systems, chemical fabrics and IoT devices machine learning gives easier options to create a complete new attack surface as with traditional software. 

Security aspects for machine learning accounts for the application where machine learning is used, but also for the developed algorithms self. So machine learning security aspects are divided into the following categories:

* Machine learning attacks aimed to fool the developed machine learning systems. Since machine learning is often a ‘black-box’ these attacks are very hard to determine.

* System attacks special for machine learning systems. Machine learning offers new opportunities to break existing traditional software systems.

* Machine learning usage threats. The outcome of many machine learning systems is far from correct. If you base decisions or trust on machine learning applications you can make serious mistakes. This accounts e.g. for self driving vehicles, health care systems and surveillance systems. Machine learning systems are known for producing racially biased results often caused by using biased data sets. Think about problematic forms of "profiling" based on surveillance cameras with face detection. 

* Machine learning hosting and infrastructure security aspects. This category is not special for machine learning but is relevant for all IT systems. Protecting 'normal' software solutions was already a known challenge. But inspecting and protecting machine learning systems require besides already deep knowledge of cyber security also knowledge of nature of machine learning systems. And remember: Machine learning systems are not traditional software systems. A machine learning systems is a complete other paradigm that requires new knowledge of building a thread model to take measurements to reduce security risks. When manipulated training data is used when training your machine learning model it make results horrible and can be dangerous.

So key threads for machine learning system can be seen as:

- Attacks which compromise confidentiality
- Attacks which compromise integrity by manipulation of input.
- 'Traditional' attacks that have impact on availability.

Attack vectors for machine learning systems can be categorized in:

* Input manipulation

* Data manipulation

* Model manipulation

* Input extraction

* Data extraction

* Model extraction

* Environmental attacks (so the IT system used for hosting the machine learning algorithms and data)


Taxonomy and terminology of machine learning is not yet fully standardized. The US NIST publication 8269 (The National Institute of Standards and Technology) a taxonomy and terminology of Adversarial Machine Learning is proposed. See https://csrc.nist.gov/publications/detail/nistir/8269/draft. Adversarial Machine Learning (AML)introduces additional security challenges in training and testing (inference) phases of system operations. AML is concerned with the design of ML algorithms that can resist security challenges, the study of the capabilities of attackers, and the understanding of attack consequences. 

Top Machine Learning Security Risks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Adversarial attacks: The basic idea is to fool a machine learning system by providing malicious input that cause the system to make a false prediction or categorization. 

* Data poisoning: Machine learning systems learn directly from data. Intentionally manipulated data can compromise the machine learning application. If you want to make yourself e.g. invisible for face recognition you can create or buy special clothes. 

* Data confidentiality: An unique challenge in machine learning is protecting confidential data. 

* Data trustworthiness: Data integrity is essential.  Are the data suitable and of high enough quality to support machine learning? Are e.g. sensors to capture data reliable? How is data integrity preserved? Understanding machine learning data sources, both during training and during execution, is of critical importance. 

* Overfitting Attacks: Overfitting means the model fits the parameters too closely with regard to the particular observations in the training dataset, but does not generalize well to new data. Most of the time the model is too complex for the given training data. Overfit models are particularly easy to attack.

* Output integrity. If an attacker can interpose between a machine learning system and produced output, a direct attack on output is possible. The inscrutability of machine learning models (so not really understanding how they work) may make an output integrity attack easy and hard to spot. 


Some examples of machine learning exploits:

* Google's Cloud Computing service can be tricked into seeing things that are not there. In one test it perceived a rifle as a helicopter. 
* Fake videos made with help from machine learning software are spreading online, and the law can’t do much about it. E.g. videos with speeches given by political leaders created by machine learning software are created and spread online. E.g. a video where some president declares a war to another country is of course very dangerous. Even more dangerous is the fact that the fake machine learning created videos are very hard to diagnose as machine learning creations. This since besides machine learning a lot of common Hollywood special effects are also used to make it hard to distinguish real videos from fake video’s. Creating online fake porn video sites were you can use a photo of a celebrity or someone you do not like, is nowadays only just three mouse clicks away. And the reality is that you can do very little against these kinds of damaging threads. Even from a legal point of view.

Users and especially developers of machine learning applications must be more paranoid from a security point of view. But unfortunately security cost a lot of effort and money and a lot of special expertise is needed to minimize the risks.

