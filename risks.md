Security,Privacy and Safety
===========================

Introduction
------------

This section outlines security, privacy and safety concerns that keep
you awake when applying machine learning for real business use.

The complexity of ML technologies has fuelled fears that machine
learning applications causes harm in unforeseen circumstances, or that
they are manipulated to act in harmful ways. Think of a self driving car
with its own ethics or algorithms that make prediction based on your
personal data that really scare you. E.g. Predicting what diseases hit
you based on data from your grocery store.

As with any technology: Technology is never neutral. You have to think
before starting what values you implicitly use to design your new
technology. All technology can and will be misused. But it is up to the
designers to think of the risks when technology will be misused. On
purpose or by accident.

Machine learning systems should be operated reliably, safely and
consistently. Not only under normal circumstances but also in unexpected
conditions or when they are under attack for misuse.

Machine learning software differs from traditional software because:

-   The outcome is not easily predictable.
-   The used trained models are a black box, with very few options for
    transparency.
-   Logical reasoning (or cause and effect) is not present. Predictions
    are made based on statistical number crunching complex algorithms
    which are non linear.
-   Both Non IT people and trained IT people have a hard time figuring
    out machine learning systems, due to the new paradigms in use.

What makes security and safety more than normal aspects for machine
learning driven applications is that by design neural networks are not
designed to to make the inner workings easy to understand for humans and
quality and risk managers.

Without a solid background in mathematics and software engineering
evaluating the correct working of most machine learning application is
impossible for security researchers and safety auditors.

However more and more people dependent on the correct outcome of
decisions made by machine learning software. So we should ask some
critical questions:

-   Is the system making any mistakes?
-   How do you know what alternatives were considered?
-   What is the risk of trusting the outcome blind?

Understanding how output produced by machine learning software is
created will make more people comfortable with self-driving cars and
other safety critical systems that are machine learning enabled. In the
end systems that can kill you must be secure and safe to use. So how do
we get the process and trust chain to a level that we are not longer
depended of:

-   Software bugs
-   Machine learning experts
-   Auditors
-   A proprietary certification process that end with a stamp (if paid
    enough)

From other sectors, like finance or oil industry we know that there is
no simple solution. However regarding the risks involved only FOSS
machine learning applications have the right elements needed to start
working on processes that give enough trust to use machine learning
system for society at large.

To reduce risks for machine learning systems needed is:

-   Transparency: ML systems should be understandable. However they will
    never be. Computer science is a complex field. Only a fraction of
    the people are able to grasp the complete working of software and
    hardware in modern computer systems. So we need to find ways to
    manage and reduce risks in order to trust systems enabled by ML
    software. Transparency can be realized by using FOSS software (for
    everything). But beware that real trust requires that anyone with
    the needed expertise should be able to rebuild the software and
    retrain also the created machine learning model using the same
    training input. In open science for machine learning this is now
    becoming the new de-facto standard for scientific research.
-   Reproducible. All data and created models must be available so other
    research can verify the working independently.

Trusts means no security by obscurity. So open research, open science,
open software and open business innovation principles should be used
when machine learning applications are developed and deployed.

Security
--------

Using machine learning technology gives some serious new threads. More
and more new ways for exploiting the technology are published. IT
security is proven to be hard and complex to control and manage. But
machine learning technology makes the problem of IT security even worse.
This is due to the fact that the special created machine learning
exploits are very hard to determine.

Machine learning challenges many current security measurements. This
because machine learning software:

-   Lowers the cost of applying current known attacks on all devices
    which depend on software. So almost all modern technology devices.
-   Machine learning software enables the easy creation of new threats
    and vulnerabilities on existing systems. E.g. you can take the CVE
    security vulnerability database (<https://www.cvedetails.com/>) and
    train a machine learning model how to create attack on the published
    omissions.
-   When machine learning software will be in hospitals, traffic control
    systems, chemical fabrics and IoT devices machine learning gives
    easier options to create a complete new attack surface as with
    traditional software.

Security aspects for machine learning accounts for the application where
machine learning is used, but also for the developed algorithms self. So
machine learning security aspects are divided into the following
categories:

-   Machine learning attacks aimed to fool the developed machine
    learning systems. Since machine learning is often a 'black-box'
    these attacks are very hard to determine.
-   System attacks special for machine learning systems. Machine
    learning offers new opportunities to break existing traditional
    software systems.
-   Machine learning usage threats. The outcome of many machine learning
    systems is far from correct. If you base decisions or trust on
    machine learning applications you can make serious mistakes. This
    accounts e.g. for self driving vehicles, health care systems and
    surveillance systems. Machine learning systems are known for
    producing racially biased results often caused by using biased data
    sets. Think about problematic forms of \"profiling\" based on
    surveillance cameras with face detection.
-   Machine learning hosting and infrastructure security aspects. This
    category is not special for machine learning but is relevant for all
    IT systems. Protecting \'normal\' software solutions was already a
    known challenge. But inspecting and protecting machine learning
    systems require besides already deep knowledge of cyber security
    also knowledge of nature of machine learning systems. And remember:
    Machine learning systems are not traditional software systems. A
    machine learning systems is a complete other paradigm that requires
    new knowledge of building a thread model to take measurements to
    reduce security risks. When manipulated training data is used when
    training your machine learning model it make results horrible and
    can be dangerous.

So key threads for machine learning system can be seen as:

-   Attacks which compromise confidentiality
-   Attacks which compromise integrity by manipulation of input.
-   \'Traditional\' attacks that have impact on availability.

Attack vectors for machine learning systems can be categorized in:

-   Input manipulation
-   Data manipulation
-   Model manipulation
-   Input extraction
-   Data extraction
-   Model extraction
-   Environmental attacks (so the IT system used for hosting the machine
    learning algorithms and data)

Taxonomy and terminology of machine learning is not yet fully
standardized. The US NIST publication 8269 (The National Institute of
Standards and Technology) a taxonomy and terminology of Adversarial
Machine Learning is proposed. See
<https://csrc.nist.gov/publications/detail/nistir/8269/draft>.
Adversarial Machine Learning (AML)introduces additional security
challenges in training and testing (inference) phases of system
operations. AML is concerned with the design of ML algorithms that can
resist security challenges, the study of the capabilities of attackers,
and the understanding of attack consequences.

### Top Machine Learning Security Risks

-   Adversarial attacks: The basic idea is to fool a machine learning
    system by providing malicious input that cause the system to make a
    false prediction or categorization.
-   Data poisoning: Machine learning systems learn directly from data.
    Intentionally manipulated data can compromise the machine learning
    application. If you want to make yourself e.g. invisible for face
    recognition you can create or buy special clothes.
-   Data confidentiality: An unique challenge in machine learning is
    protecting confidential data.
-   Data trustworthiness: Data integrity is essential. Are the data
    suitable and of high enough quality to support machine learning? Are
    e.g. sensors to capture data reliable? How is data integrity
    preserved? Understanding machine learning data sources, both during
    training and during execution, is of critical importance.
-   Overfitting Attacks: Overfitting means the model fits the parameters
    too closely with regard to the particular observations in the
    training dataset, but does not generalize well to new data. Most of
    the time the model is too complex for the given training data.
    Overfit models are particularly easy to attack.
-   Output integrity. If an attacker can interpose between a machine
    learning system and produced output, a direct attack on output is
    possible. The inscrutability of machine learning models (so not
    really understanding how they work) may make an output integrity
    attack easy and hard to spot.

Some examples of machine learning exploits:

-   Google\'s Cloud Computing service can be tricked into seeing things
    that are not there. In one test it perceived a rifle as a
    helicopter.
-   Fake videos made with help from machine learning software are
    spreading online, and the law can't do much about it. E.g. videos
    with speeches given by political leaders created by machine learning
    software are created and spread online. E.g. a video where some
    president declares a war to another country is of course very
    dangerous. Even more dangerous is the fact that the fake machine
    learning created videos are very hard to diagnose as machine
    learning creations. This since besides machine learning a lot of
    common Hollywood special effects are also used to make it hard to
    distinguish real videos from fake video's. Creating online fake porn
    video sites were you can use a photo of a celebrity or someone you
    do not like, is nowadays only just three mouse clicks away. And the
    reality is that you can do very little against these kinds of
    damaging threads. Even from a legal point of view.

Users and especially developers of machine learning applications must be
more paranoid from a security point of view. But unfortunately security
cost a lot of effort and money and a lot of special expertise is needed
to minimize the risks.

Privacy
-------

Machine learning raises serious privacy concerns since machine learning
is using massive amounts of data that contain often personal
information.

It is a common believe that personal information is needed for
experimenting with machine learning before you can create good and
meaningful applications. E.g. for health applications, travel
applications, eCommerce and of course marketing applications. Machine
learning models are often loaded with massive amounts of personal data
for training and to make in the end good meaningful predictions.

The belief that personal data is needed for machine learning creates a
tension between developers and privacy aware consumers. Developers want
the ability to create innovative new products and services and need to
experiment, while consumers and GDPR regulators are concerned for the
privacy risks involved.

The applicability of machine learning models is hindered in settings
where the risk of data leakage raises serious privacy concerns. Examples
of such applications include scenar-ios where clients hold sensitive
private information, e.g., medical records, financial data, or location.

It is commonly believed that individuals must provide a copy of their
personal information in order for AI to train or predict over it. This
belief creates a tension between developers and consumers. Developers
want the ability to create innovative products and services, while
consumers want to avoid sending developers a copy of their data.

Machine learning models can be trained in environments that are not
secure on data it never has access to. Secure machine learning that
works on anonymized data sets is still an obscure and unpaved path. But
some companies and organizations are already working on creating deep
learning technology that works on encrypted data. Using encryption on
data to train machine learning models raises the complexity in various
ways. It is already hard or impossible to understand the inner working
of the 'black-box' machine learning models. Using advanced data
encryption will require even more knowledge and competences for all
engineers involved when developing machine learning applications.

In the EU the use of personal data is protected by law in all countries
by a single law. The EU General Data Protection Regulation (GDPR). This
GDPR does not prohibit the use of machine learning. But when you use
personal data you will have a severe challenge to explain to DPOs (Data
Protection Officers) and consumers what you actually do with the data
and how you comply with the GDPR.

Machine learning systems must be data responsible. They should use only
what they need and delete it when it is no longer needed ("data
minimization"). They should encrypt data in transit and at rest, and
restrict access to authorized persons ("access control"). Machine
learning systems should only collect, use, share and store data in
accordance with privacy and personal data laws and best practices. Since
FOSS machine learning needs full transparency and reproducibility using
private data should be avoided if possible.

When you apply machine learning for your business application you should
consider the following questions:

-   In what way will your customers be happy with their data usage for
    their and your benefit?
-   Do you really have a clear and good overview of all GDPR
    implications when using personal data in your machine learning
    model? What happens if you invite other companies to use your model?
-   What are the ethical concerns when using massive amounts of data of
    your customers to develop new products? Is the way you use the data
    to train your model congruent with you business vision and moral?
-   What are the privacy risks involved for your machine learning
    development chain and application?

Since security and privacy is complex to apply, frameworks are being
developed to make this challenge easier. E.g. Tensorflow Encrypted aims
to make privacy-preserving deep learning simple and approachable,
without requiring expertise in cryptography, distributed systems, or
high-performance computing. And PySyft is a Python library for secure,
private Deep Learning. More on both frameworks can be found in the
section on open ML software.

Safety
------

Machine learning is a powerful tool for businesses. But it can also lead
to unintended and dangerous consequences for users of systems powered by
machine learning software. The cause of safety issues is linked to the
people and data that train and deploy the machine learning software and
systems. Everyone involved in creating machine learning based systems
should be aware of possible safety risks come when using machine
learning technology.

Safety is a multifaceted area of research, with many sub-questions in
areas such as reward learning, robustness, and interpretability.

A machine learning driven system can currently only be as good as the
data it is given to work with. However you almost can never traceback to
the data that was used to train and develop the system. This makes that
the safety aspect should be kept in mind when dealing with security
aspects for systems that deal direct or indirect with humans.

To avoid dangerous bias or incorrect actions from systems, you should
develop machine learning system in the open and make the everything
reproducible from the start.

However safety risks will always be there: It is impossible to cover all
perspectives and variables for a machine learning system in development
before it is released. And the nature of machine learning systems means
that the outcome of machine learning is never perfect. Risks will always
be present. So not all use cases possible for machine learning are
acceptable from an ethical point of view.

The following activities will reduce safety risks and increase
reliability of machine learning systems:

-   Systematic evaluation: So evaluate the data and models used to train
    and operate machine learning based products and services.
-   Create processes for solid documenting and auditing operations.
-   Involve domain experts. Involvement of domain experts in the design
    process and operation of machine learning systems. Also involve real
    people in advance who are in the end targeted by outcomes of ml
    systems especially when decisions about people are made using
    machine learning applications.
-   Evaluation of when and how a machine learning system should seek
    human input during critical situations, and how a system controlled
    by a human in a manner that is meaningful and intelligible.
-   A robust feedback mechanism so that users can report issues they
    experience.
