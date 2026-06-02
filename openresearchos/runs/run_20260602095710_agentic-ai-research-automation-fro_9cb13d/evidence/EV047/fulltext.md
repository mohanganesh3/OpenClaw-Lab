[2606.02530] SafeSteer: Localized On-Policy Distillation for Efficient Safety Alignment

Skip to main content

Learn about arXiv becoming an independent nonprofit.

We gratefully acknowledge support from the Simons Foundation,  member institutions , and all contributors.

Donate

&gt;

cs

&gt;  arXiv:2606.02530

Help  |  Advanced Search

All fields

Title

Author

Abstract

Comments

Journal reference

ACM classification

MSC classification

Report number

arXiv identifier

DOI

ORCID

arXiv author ID

Help pages

Full text

Search

GO

quick links

Login

Help Pages

About

-->

Computer Science > Artificial Intelligence

arXiv:2606.02530  (cs)

[Submitted on 1 Jun 2026]

Title: SafeSteer: Localized On-Policy Distillation for Efficient Safety Alignment

Authors:  Hao Li ,  Jingkun An ,  Zijun Song ,  Pengyu Zhu ,  Rui Li ,  Hao Wang ,  Wendi Feng ,  Yesheng Liu ,  Lijun Li ,  Jin-Ge Yao ,  Lei Sha

View a PDF of the paper titled SafeSteer: Localized On-Policy Distillation for Efficient Safety Alignment, by Hao Li and 10 other authors

View PDF

HTML (experimental)

Abstract: Aligning Large Language Models (LLMs) with human values often degrades their general capabilities, termed the alignment tax. Existing methods mitigate this by balancing dual objectives, which heavily rely on massive general-purpose data or auxiliary reward models.
 In this paper, we argue that, because safety features are inherently sparse within the output distribution, alignment requires localized modifications rather than global trade-offs. To this end, we propose SafeSteer, which performs on-policy distillation confined to safety tokens. First, we construct a safety teacher via activation steering. Based on this teacher, we develop a safety token selection algorithm. Consequently, SafeSteer restricts the reverse KL penalty to these tokens during training to preserve general capabilities.
 Experimental results across diverse models show that our SafeSteer achieves a superior trade-off between safety and general capability compared with existing methods, attaining strong safety performance on seven safety benchmarks with only minimal degradation on five general capability benchmarks. Notably, SafeSteer requires only 100 harmful samples without using any general-purpose data, less than 1% of what previous baselines used, considerably reducing alignment cost. More details are on our project page at  this https URL .

Comments:

19 pages, 8 figures, 14 tables. Submitted to EMNLP 2026

Subjects:

Artificial Intelligence (cs.AI) ; Computation and Language (cs.CL)

Cite as:

arXiv:2606.02530  [cs.AI]

&nbsp;

(or

arXiv:2606.02530v1  [cs.AI]  for this version)

&nbsp;

https://doi.org/10.48550/arXiv.2606.02530

Focus to learn more

arXiv-issued DOI via DataCite (pending registration)

Submission history  From: Jingkun An [ view email ]

[v1]

Mon, 1 Jun 2026 17:38:12 UTC (2,088 KB)

Full-text links:

Access Paper:

View a PDF of the paper titled SafeSteer: Localized On-Policy Distillation for Efficient Safety Alignment, by Hao Li and 10 other authors

View PDF

HTML (experimental)

TeX Source

view license

Current browse context:

cs.AI

&lt;&nbsp;prev

&nbsp; | &nbsp;

next&nbsp;&gt;

new

|

recent

|

2026-06

Change to browse by:

cs

cs.CL

References &amp; Citations

NASA ADS

Google Scholar

Semantic Scholar

export BibTeX citation

Loading...

BibTeX formatted citation

&times;

loading...

Data provided by:

Bookmark

Bibliographic Tools

Bibliographic and Citation Tools

Bibliographic Explorer Toggle

Bibliographic Explorer

( What is the Explorer? )

Connected Papers Toggle

Connected Papers

( What is Connected Papers? )

Litmaps Toggle

Litmaps

( What is Litmaps? )

scite.ai Toggle

scite Smart Citations

( What are Smart Citations? )

Code, Data, Media

Code, Data and Media Associated with this Article

alphaXiv Toggle

alphaXiv

( What is alphaXiv? )

Links to Code Toggle

CatalyzeX Code Finder for Papers

( What is CatalyzeX? )

DagsHub Toggle

DagsHub

( What is DagsHub? )

GotitPub Toggle

Gotit.pub

( What is GotitPub? )

Huggingface Toggle

Hugging Face

( What is Huggingface? )

ScienceCast Toggle

ScienceCast

( What is ScienceCast? )

Demos

Demos

Replicate Toggle

Replicate

( What is Replicate? )

Spaces Toggle

Hugging Face Spaces

( What is Spaces? )

Spaces Toggle

TXYZ.AI

( What is TXYZ.AI? )

Related Papers

Recommenders and Search Tools

Link to Influence Flower

Influence Flower

( What are Influence Flowers? )

Core recommender toggle

CORE Recommender

( What is CORE? )

Author

Venue

Institution

Topic

About arXivLabs

arXivLabs: experimental projects with community collaborators

arXivLabs is a framework that allows collaborators to develop and share new arXiv features directly on our website.

Both individuals and organizations that work with arXivLabs have embraced and accepted our values of openness, community, excellence, and user data privacy. arXiv is committed to these values and only works with partners that adhere to them.

Have an idea for a project that will add value for arXiv's community?

Learn more about arXivLabs  .

Which authors of this paper are endorsers?  |

Disable MathJax  ( What is MathJax? )

About

Help

contact arXiv  Click here to contact arXiv

Contact

subscribe to arXiv mailings  Click here to subscribe

Subscribe

Copyright

Privacy Policy

Web Accessibility Assistance

arXiv Operational Status