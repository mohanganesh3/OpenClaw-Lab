[2112.01575] Towards Interactive Reinforcement Learning with Intrinsic Feedback

Towards Interactive Reinforcement Learning with Intrinsic Feedback

Benjamin Poole 1

Corresponding author

1 Department of Computer Science

University of North Carolina at Charlotte, 9201 University City Blvd, Charlotte NC, 28223

Minwoo Lee 1

1 Department of Computer Science

University of North Carolina at Charlotte, 9201 University City Blvd, Charlotte NC, 28223

({ bpoole16, Minwoo.Lee } @charlotte.edu )

Abstract

Reinforcement learning (RL) and brain-computer interfaces (BCI) have experienced significant growth over the past decade. With rising interest in human-in-the-loop (HITL), incorporating human input with RL algorithms has given rise to the sub-field of interactive RL. Adjacently, the field of BCI has long been interested in extracting informative brain signals from neural activity for use in human-computer interactions. A key link between these fields lies in the interpretation of neural activity as feedback such that interactive RL approaches can be employed. We denote this new and emerging medium of feedback as  intrinsic feedback . Despite intrinsic feedback’s ability to be conveyed automatically and even unconsciously, proper exploration surrounding this key link has largely gone unaddressed by both communities. Thus, to help facilitate a deeper understanding and a more effective utilization, we provide a tutorial-style review covering the motivations, approaches, and open problems of intrinsic feedback and its foundational concepts.

1  Introduction

The combination of reinforcement learning (RL) and human-in-the-loop (HITL) ideas through integration of human input into the RL framework has led to the emergence of the field of interactive RL  [ 131 ,  71 ,  9 ,  88 ] . Interactive RL can be seen as a method for facilitating human-to-agent knowledge transfer. This knowledge transfer inherently helps to alleviate various aptitude and alignment challenges an agent may face. Challenges encompassed by the term  aptitude  correspond to challenges for simply being able to learn. This includes ideas such as  robustness , the ability of an agent to perform a task (e.g., asymptotic performance) and generalize within/between environments of similar complexity;  scalability , the ability of an agent to scale up to more complex environments; and  aptness , the rate at which an agent can learn a desired behavior. On the other hand,  alignment  encompasses challenges concerned with learning as intended (i.e., learning to behave as a human dictates)  [ 4 ,  69 ,  25 ,  32 ,  99 ] . The hypothetical Paperclip Maximizer  [ 13 ]  is a classical example of misalignment where an agent maximizes its goal of making paperclips in an unintended manner by turning the world into a paperclip factory. While not as extreme, RL algorithms often display lesser forms of misalignment  [ 4 ,  69 ,  32 ] . Therefore, it is not unreasonable to anticipate an increase in misaligned behaviors as the problem complexity increases, i.e., the complexity of the environment and behaviors  [ 69 ] .

The field of brain-computer interfaces (BCI) aims to develop systems for monitoring the brain and utilizing neural activity as a direct means of communication between the brain and external devices. BCI research has primarily focused on medical applications such as fatigue detection, monitoring sleep, analyzing emotion, and facilitating sensory-motor rehabilitation. However, particular interest has been directed towards addressing paralysis conditions like amyotrophic lateral sclerosis and locked-in-state. These paralysis conditions necessitate innovative technologies that can assist patients in interacting with the external world. Therefore, traditional BCI applications have been primarily concentrated on control problems such as wheelchair and interface control  [ 98 ,  102 ] . Yet, as the medical BCI research has progressed, it has begun to spill out into the non-medical realm  [ 24 ] . Applications in robotics  [ 56 ,  126 ,  100 ] , VR and gaming  [ 51 ] , smart environments  [ 67 ] , bio-metrics and security  [ 38 ] , and brain-to-brain interfaces  [ 97 ,  49 ]  have provided opportunities to further explore this form of intrinsic communication. Moreover, there has been growing interest in using BCI to communicate with artificial agents to facilitate human-AI interactions. One promising avenue involves the integration of RL and BCI, where particular neural activity is interpreted as a reward in the RL framework  [ 45 ,  47 ,  26 ,  54 ] . Interestingly, as we will soon see, this fundamental integration can actually be more broadly formulated using interactive RL concepts.

In the realm of interactive RL, a crucial concern revolves around seamlessly integrating and effectively communicating human intentions to an agent  [ 59 ,  73 ] . To this end, human input is often categorized into two main channels of communication: advice and demonstrations  [ 88 ] . Of particular interest is a specific type of advice denoted as  feedback , which aims to convey the "goodness" or "badness" of an agent’s behavior. Classically, the de facto medium for conveying feedback has been what we refer to as  explicit feedback . This type of feedback involves the human communicating directly with the agent via a button push or very basic natural language  [ 61 ,  59 ,  8 ,  20 ] . More recently,  implicit feedback  provided through human social cues (e.g., facial expressions, body language, or vocal tone) has begun to see wider use in conveying feedback as it offers a more natural and potentially automatic means of communication  [ 74 ] .

Yet, both explicit and implicit feedback do not cover the entire range of potential human feedback mediums. Another untapped source of valuable information lies in intrinsically occurring and automatically elicited biological signals. These raw and largely unconsciously elicited biological signals have the potential to provide a wealth of information concerning a human’s internal state including feelings and even beliefs. Thus, we denote  intrinsic feedback  as a new medium for providing feedback where feedback is implied through a human’s intrinsically occurring biological signals  1

1  1 Originally termed as “intrinsic interactive RL” by  Kim et al. [ 54 ] , we now extend the term “intrinsic” to potentially encompass all biological signals, not just brain signals, for the sake of generalization.

. As recent works  [ 128 ,  80 ,  16 ,  121 ,  2 ]  have begun to recognize that neural activity can be interpreted more broadly as feedback, we will focus particularly on the brain, the central organ in the human nervous system, as it harbors a dense concentration of biological signals. For brevity, we will henceforth use the terms intrinsic feedback and neurological intrinsic feedback interchangeably.

Table 1 :

Summary of interactive RL reviews categorized by the human input categories of advice and demonstrations (denoted as “Demos”). Advice is further broken into categories of feedback and other (i.e., all other types of advice such as general advice, guidance, ect). Feedback is further broken into its three different mediums. A category with a check mark ✓indicates the review is  primarily  focused on a particular category of human input while no check mark indicates a given category is not discussed or only briefly mentioned. For the reader’s reference, we have included a single comprehensive learning from demonstrations review although many more exist.

Review

Summary

Advice

Demos

Feedback

Other

Intrinsic

Explicit

Implicit

Osa et al.  [  92  ]

(

2018

)

An in-depth technical review of various learning from demonstration approaches (e.g., behavioral cloning and inverse RL).

-

-

-

-

✓

Zhang et al.  [  131  ]

(

2019a

)

A high-level overview of basic interactive RL ideas and approaches.

-

✓

-

-

✓

Li et al.  [  71  ]

(

2019

)

A partially technical ov