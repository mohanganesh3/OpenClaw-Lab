[2212.07126] Explainability of Text Processing and Retrieval Methods: A Critical Survey

\setupctable

botcap, captionskip=2pt, mincapwidth=2in, framebg=1 1 1, framefg=0 0 0, framerule=0pt, framesep=0pt,

Explainability of Text Processing and Retrieval Methods: A Critical Survey

Sourav Saha

Indian Statistical Institute, Kolkata

India

sourav.saha_r@isical.ac.in

,

Debapriyo Majumdar

Indian Statistical Institute, Kolkata

India

debapriyo@isical.ac.in

and

Mandar Mitra

Indian Statistical Institute, Kolkata

India

mandar@isical.ac.in

Abstract.

Deep Learning and Machine Learning based models have become extremely
popular in text processing and information retrieval. However, the
non-linear structures present inside the networks make these models
largely inscrutable. A significant body of research has focused on
increasing the transparency of these models. This article provides a
broad overview of research on the explainability and interpretability of
natural language processing and information retrieval methods. More
specifically, we survey approaches that have been
applied to explain word embeddings, sequence modeling, attention
modules, transformers, BERT, and document ranking. The concluding section
suggests some possible directions for future research on this topic.

Explainability, Interpretability, Text Processing, Information Retrieval, Natural Language Processing, Machine Learning

†

†  conference:  ; ;

†

†  ccs:  Information systems Information retrieval

1.  Introduction

Given the volume of information that we routinely encounter, we have become
acutely dependent on technology that attempts to process the available
information, classify it on the basis of diverse criteria, select useful
information from this barrage, and present it in a form that is easy to
consume and assimilate. It is not enough for such technology to be good at
what it does: as humans, we would ideally also like to understand why our
tools behave the way they do. Naturally, a substantial body of research has
focused on methods for providing high-level, human-understandable
 interpretations  or  explanations  of the inner workings of, as
well as the results produced by, information processing tools and technologies.

Upto about 10 years ago, commonly used information processing tools
generally extracted a  feature vector  — a list of features along
with their associated numerical weights — from the form in which the
information was initially provided: structured or unstructured text,
images, audio, video, etc. These features typically corresponded to
human-understandable properties or attributes of the original information.
We confine our discussion in this article to textual information only. For
text, features would include words and phrases occurring in the text, their
counts, morphology, parts of speech, and various grammatical structures
appearing in the text. These features were chosen on the basis of
linguistic considerations; they were, in a sense, interpretable by design.
A number of studies (some of which are discussed in more detail in
Section

9  ) did focus on understanding feature
weights, their behaviour, and the mathematical / statistical models
underlying their computation and use, but they were not explicitly
designated as  explainability  /  intepretability  studies.

Over the past few years, advances in Deep Machine Learning (ML) techniques
have revolutionised the state of the art in textual information processing
(and, of course, in a number of other areas).
However, these improvements have come at a cost: text is no longer simply
represented in terms of easily interpretable features. Instead, early in
the information processing pipeline, text is converted to so-called
“dense” vectors in

ℝ  n

superscript  ℝ  𝑛

\mathbb{R}^{n}

; the dimensions of this space are no
longer easily understood, even in isolation. This representation is then
processed by ML systems that have large and complex architectures,
involving many parameters, often numbering in the millions.  As a consequence, in recent
times, the research community has devoted a good deal of attention to
providing high-level, intuitive explanations of these models and methods.

Thus, this seems to be an appropriate time to present an organised summary
of the major research efforts in the area of explainability in textual
information processing and retrieval. Indeed, several such general surveys
have already been
published  (Søgaard,  2021 ; Madsen et al.,  2022 ; Zini and Awad,  2022 ; Anand et al.,  2022 ) ,
in addition to more specialised surveys, such
as  (Belinkov and Glass,  2019 ; Dufter et al.,  2022 ) . Our objective is to
complement these existing surveys by discussing more recent work in areas
already covered there, and by focussing on areas not addressed by them.
Before discussing these surveys in more detail in
Section

2  , we first briefly describe how we collected
the set of papers summarised in this survey.

1.1.  Scope

Text processing and retrieval problems are studied by researchers from the
Information Retrieval (IR) and Natural Language Processing (NLP)
communities, as well as from the broader Machine Learning / Deep Learning
(DL) communities. Thus, to ensure comprehensive coverage of explainability
and interpretability studies relating to IR and NLP, we identified papers
from a wide range of highly-regarded venues including  SIGIR ,
 CIKM ,  WSDM ,  WWW , the  ACL  family of conferences
and journals,  AAAI ,  ICML ,  ICLR ,  NeurIPS ,  FAccT  and
 IPM  (Information Processing &amp; Management). From the longlisted set
of papers, a subset of 41 papers was selected for discussion in this
article.

While assessing and summarising research results in an empirical
discipline, it is important to keep in mind what datasets were used for
validation, as well as their limitations, if any. Accordingly, in addition
to discussing the method(s) adopted in each study, we make it a point to
include details about the datasets that were used for experimental
evaluation.

Organization of the survey:

The remainder of this survey is
organised as follows. Section

2

lists existing surveys
that cover explainability in textual IR and NLP, and describes the relation
between them and the current article. Section

3

introduces the basic terminology used in the literature on explainability.
Section

4

reviews some well-known approaches to
explainability that are most commonly used as baselines in recent work.
Sections

5

through

8

cover
explainability for the various elements of deep text processing and
retrieval pipelines, e.g., word embeddings, attention mechanisms,
transformer architectures, and are intended as supplements to existing
surveys (specifically those by  Madsen et al. ( 2022 )  and
 Zini and Awad ( 2022 ) ). Section

9

is devoted to
a detailed discussion of explainable document / passage ranking, and is
itself divided into subsections by topic. Within each of these sections and
subsections, different studies have been discussed in chronological order.
We conclude in Section

10

with a few research
questions that we believe merit further investigation.

2.  Related Work

In a recent monograph  (Lin et al.,  2021 ) ,

Lin et al.

provide a comprehensive survey of neural IR models, covering topics from
multi-stage ranking architectures to dense retrieval techniques. The
authors observe that IR models may be stratified into three fairly
well-separated levels on the basis of effectiveness: transformer-based
ranking strategies represent a definite advance over pre-transformer neural
retrieval models, which in turn improve significantly over
“traditional”  1

1  1 In this survey, we use this adjective to refer to
all IR models (including relatively recently proposed ones) from the
pre-neural era.

statistical ranking models. An intuitive (but precise)
grasp of what leads to these quantum jumps in effectiveness seems eminently
desirable.
This, along with a number of othe