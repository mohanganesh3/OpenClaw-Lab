[2606.02559] From Layers to Submodules: Rethinking Granularity in Replacement-Based LLM Compression

Skip to main content

Learn about arXiv becoming an independent nonprofit.

We gratefully acknowledge support from the Simons Foundation,  member institutions , and all contributors.

Donate

&gt;

cs

&gt;  arXiv:2606.02559

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

Computer Science > Computation and Language

arXiv:2606.02559  (cs)

[Submitted on 1 Jun 2026]

Title: From Layers to Submodules: Rethinking Granularity in Replacement-Based LLM Compression

Authors:  Elia Cunegatti ,  Marcus Vukojevic ,  Erik Nielsen ,  Giovanni Iacca

View a PDF of the paper titled From Layers to Submodules: Rethinking Granularity in Replacement-Based LLM Compression, by Elia Cunegatti and 3 other authors

View PDF

HTML (experimental)

Abstract: Post-training compression of Large Language Models (LLMs) removes entire architectural components, either deleting them or replacing them with fitted modules. Existing replacement-based methods share two design constraints: full-layer granularity and contiguous selection. We argue that this is overly restrictive: in fact, redundancy in pretrained transformers is not confined to contiguous regions, nor does it evenly distribute between Attention and FeedForward outputs, implying that different strategies best approximate different submodule types and that removable components need not cluster within contiguous depth ranges. Based on this intuition, we introduce SubFit (Submodule-level Fitted residual replacement), which compresses LLMs at the submodule level: Attention and FeedForward submodules are selected non-contiguously, and each receives its own lightweight fitted residual bypass. SubFit operates post-training and requires only calibration data. Across ten LLMs (five base, five instruction-tuned), five sparsity levels from 12.5% to 37.5%, and four replacement-based baselines, SubFit achieves the best aggregate perplexity-accuracy trade-off across the evaluated sparsity levels, with larger gains under aggressive compression. At 25% sparsity, it retains 84.6% of dense downstream accuracy and incurs 2.42x perplexity degradation, against 81.6% and 4.34x for the strongest baselines, while delivering measurable inference speedup and KV-cache savings. Code is available at  this https URL .

Subjects:

Computation and Language (cs.CL) ; Artificial Intelligence (cs.AI)

Cite as:

arXiv:2606.02559  [cs.CL]

&nbsp;

(or

arXiv:2606.02559v1  [cs.CL]  for this version)

&nbsp;

https://doi.org/10.48550/arXiv.2606.02559

Focus to learn more

arXiv-issued DOI via DataCite (pending registration)

Submission history  From: Elia Cunegatti Mr. [ view email ]

[v1]

Mon, 1 Jun 2026 17:52:53 UTC (119 KB)

Full-text links:

Access Paper:

View a PDF of the paper titled From Layers to Submodules: Rethinking Granularity in Replacement-Based LLM Compression, by Elia Cunegatti and 3 other authors

View PDF

HTML (experimental)

TeX Source

view license

Current browse context:

cs.CL

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