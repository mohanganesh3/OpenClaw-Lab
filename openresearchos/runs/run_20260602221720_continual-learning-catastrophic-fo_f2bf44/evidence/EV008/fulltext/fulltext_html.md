[2605.19080] MANGO: Meta-Adaptive Network Gradient Optimization for Online Continual Learning

Skip to main content

Learn about arXiv becoming an independent nonprofit.

We gratefully acknowledge support from the Simons Foundation,  member institutions , and all contributors.

Donate

&gt;

cs

&gt;  arXiv:2605.19080

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

Computer Science > Machine Learning

arXiv:2605.19080  (cs)

[Submitted on 18 May 2026]

Title: MANGO: Meta-Adaptive Network Gradient Optimization for Online Continual Learning

Authors:  Ankita Awasthi ,  Marco Apolinario ,  Kaushik Roy

View a PDF of the paper titled MANGO: Meta-Adaptive Network Gradient Optimization for Online Continual Learning, by Ankita Awasthi and 2 other authors

View PDF

HTML (experimental)

Abstract: In Online Continual Learning (OCL), a neural network sequentially learns from a non-stationary data stream in a single-pass with access only to a limited memory replay buffer. This contrasts sharply with off-line continual learning where training is multiple epoch dependent on large datasets. The main challenge faced by OCL is to overcome catastrophic forgetting of past tasks (stability) while learning new ones efficiently (plasticity). Existing methods counter forgetting via replay-based rehearsal, output level distillation, fixed regularization, or meta-learning on the current data. However, these methods have limitations: rehearsal introduces a stored sample bias; distillation operates on output-distributions without modulating parameter updates; fixed-regularization penalizes parameters irrespective of sensitivity; stream-only meta-learning lacks a feedback controlled parameter update. We propose Meta-Adaptive Network Gradient Optimization (MANGO), an OCL framework that balances stability-plasticity via gradient-gating and meta-learned regularization. Gradient-gating scales parameter updates based on sensitivity, preventing destructive updates. Meta-learned regularization adapts stability coefficients, evaluating the effect of parameter update on replay. In MANGO, replay acts as both a training signal and a forgetting evaluator. We evaluated our method on three standard OCL benchmark datasets. MANGO outperforms strong baselines, achieving state-of-the-art results with consistent performance across replay sizes. In domain incremental learning on CLEAR-10 and class incremental learning on CIFAR-100 and Tiny-ImageNet, it achieves highest accuracy among all baselines and achieves positive Backward Transfer, overcoming forgetting on CLEAR-10.

Subjects:

Machine Learning (cs.LG) ; Artificial Intelligence (cs.AI)

Cite as:

arXiv:2605.19080  [cs.LG]

&nbsp;

(or

arXiv:2605.19080v1  [cs.LG]  for this version)

&nbsp;

https://doi.org/10.48550/arXiv.2605.19080

Focus to learn more

arXiv-issued DOI via DataCite

Submission history  From: Ankita Awasthi [ view email ]

[v1]

Mon, 18 May 2026 20:11:26 UTC (157 KB)

Full-text links:

Access Paper:

View a PDF of the paper titled MANGO: Meta-Adaptive Network Gradient Optimization for Online Continual Learning, by Ankita Awasthi and 2 other authors

View PDF

HTML (experimental)

TeX Source

view license

Current browse context:

cs.LG

&lt;&nbsp;prev

&nbsp; | &nbsp;

next&nbsp;&gt;

new

|

recent

|

2026-05

Change to browse by:

cs

cs.AI

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

IArxiv recommender toggle

IArxiv Recommender

( What is IArxiv? )

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