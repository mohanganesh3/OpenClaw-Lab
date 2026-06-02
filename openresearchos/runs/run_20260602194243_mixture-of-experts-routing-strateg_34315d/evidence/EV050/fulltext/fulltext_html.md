[2605.15484] When Does Sparse MoE Help in Vision? The Role of Backbone Compute Leverage in Sparse Routing

Skip to main content

Learn about arXiv becoming an independent nonprofit.

We gratefully acknowledge support from the Simons Foundation,  member institutions , and all contributors.

Donate

&gt;

cs

&gt;  arXiv:2605.15484

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

Computer Science > Computer Vision and Pattern Recognition

arXiv:2605.15484  (cs)

[Submitted on 15 May 2026]

Title: When Does Sparse MoE Help in Vision? The Role of Backbone Compute Leverage in Sparse Routing

Authors:  Libo Sun ,  Po-wei Harn ,  Peixiong He ,  Xiao Qin

View a PDF of the paper titled When Does Sparse MoE Help in Vision? The Role of Backbone Compute Leverage in Sparse Routing, by Libo Sun and 3 other authors

View PDF

HTML (experimental)

Abstract: Mixture-of-Experts (MoE) networks promise favorable accuracy-compute trade-offs, yet practical vision deployments are hindered by expert collapse and limited end-to-end efficiency gains. We study when sparse top-$k$ routing with hard capacity constraints helps in vision classification, evaluated under multi-seed protocols on four benchmarks (CIFAR-10/100, Tiny-ImageNet, ImageNet-1K). We observe a \emph{compute-leverage pattern}: positive accuracy gaps require a substantial fraction $\rho$ of total FLOPs to be routed; at ImageNet scale this is necessary but not sufficient, as multi-expert routing ($k \geq 2$) is additionally required. Two controlled experiments isolate these factors. A hidden-size sweep on CIFAR-10 yields both predicted sign reversals across standard and depthwise backbones, ruling out backbone family as the active variable. An ImageNet-1K ablation that varies only top-$k$ -- holding architecture, initialization, and $\rho$ fixed -- reverses the gap from positive to negative across all five seeds. A per-sample variant of Soft MoE that softmaxes over experts rather than the batch rescues CIFAR-100 above the dense baseline, identifying batch-axis dispatch as the dominant failure mode in per-sample CNN settings. Code and aggregate results:  this https URL .

Comments:

24 pages (main + appendix), 8 figures, 18 tables. Under review at TMLR. Code and aggregate results:  this https URL

Subjects:

Computer Vision and Pattern Recognition (cs.CV) ; Machine Learning (cs.LG)

Cite as:

arXiv:2605.15484  [cs.CV]

&nbsp;

(or

arXiv:2605.15484v1  [cs.CV]  for this version)

&nbsp;

https://doi.org/10.48550/arXiv.2605.15484

Focus to learn more

arXiv-issued DOI via DataCite

Submission history  From: Libo Sun [ view email ]

[v1]

Fri, 15 May 2026 00:01:11 UTC (4,516 KB)

Full-text links:

Access Paper:

View a PDF of the paper titled When Does Sparse MoE Help in Vision? The Role of Backbone Compute Leverage in Sparse Routing, by Libo Sun and 3 other authors

View PDF

HTML (experimental)

TeX Source

view license

Current browse context:

cs.CV

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

cs.LG

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