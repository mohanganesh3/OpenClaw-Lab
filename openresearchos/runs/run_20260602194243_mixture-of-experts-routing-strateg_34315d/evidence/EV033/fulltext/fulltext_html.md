[2510.04286] SliceMoE: Routing Embedding Slices Instead of Tokens for Fine-Grained and Balanced Transformer Scaling

Skip to main content

Learn about arXiv becoming an independent nonprofit.

We gratefully acknowledge support from the Simons Foundation,  member institutions , and all contributors.

Donate

&gt;

cs

&gt;  arXiv:2510.04286

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

arXiv:2510.04286  (cs)

[Submitted on 5 Oct 2025]

Title: SliceMoE: Routing Embedding Slices Instead of Tokens for Fine-Grained and Balanced Transformer Scaling

Authors:  Harshil Vejendla

View a PDF of the paper titled SliceMoE: Routing Embedding Slices Instead of Tokens for Fine-Grained and Balanced Transformer Scaling, by Harshil Vejendla

View PDF

Abstract: Mixture-of-Experts (MoE) layers scale transformers by routing tokens to a sparse subset of feed-forward experts. Token-level routing, however, assigns an entire semantic spectrum to each expert, creating capacity bottlenecks, load-balancing pathologies, and limited specialization. We introduce SliceMoE, an architecture that routes contiguous slices of a token&#39;s hidden vector. A d-dimensional embedding is partitioned into S slices, and for each slice, a lightweight shared router predicts the top-k experts. Experts operate on their assigned slices independently, and outputs are reassembled, maintaining per-token FLOP efficiency. Because slices from different tokens interleave within an expert, utilization is naturally smoother. We propose a slice-level capacity loss, cross-slice dropout, and efficient fused batched GEMM kernels. Experiments on WikiText-103 language modeling, WMT En-De translation, and three text-classification datasets show SliceMoE attains up to 1.7x faster inference than dense baselines, 12 to 18 percent lower perplexity than parameter-matched token-MoE, and improved expert balance, with interpretable expertise over syntactic versus semantic subspaces.

Comments:

EMNLP 2025 Main, 8 pages, 9 figures

Subjects:

Computation and Language (cs.CL) ; Artificial Intelligence (cs.AI); Machine Learning (cs.LG)

Cite as:

arXiv:2510.04286  [cs.CL]

&nbsp;

(or

arXiv:2510.04286v1  [cs.CL]  for this version)

&nbsp;

https://doi.org/10.48550/arXiv.2510.04286

Focus to learn more

arXiv-issued DOI via DataCite

Submission history  From: Harshil Vejendla [ view email ]

[v1]

Sun, 5 Oct 2025 16:57:32 UTC (7,718 KB)

Full-text links:

Access Paper:

View a PDF of the paper titled SliceMoE: Routing Embedding Slices Instead of Tokens for Fine-Grained and Balanced Transformer Scaling, by Harshil Vejendla

View PDF

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

2025-10

Change to browse by:

cs

cs.AI

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