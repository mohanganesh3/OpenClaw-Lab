[2605.09516] Mixture of Layers with Hybrid Attention

Skip to main content

Learn about arXiv becoming an independent nonprofit.

We gratefully acknowledge support from the Simons Foundation,  member institutions , and all contributors.

Donate

&gt;

cs

&gt;  arXiv:2605.09516

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

arXiv:2605.09516  (cs)

[Submitted on 10 May 2026]

Title: Mixture of Layers with Hybrid Attention

Authors:  Ivan Ternovtsii ,  Yurii Bilak

View a PDF of the paper titled Mixture of Layers with Hybrid Attention, by Ivan Ternovtsii and 1 other authors

View PDF

HTML (experimental)

Abstract: Standard Mixture-of-Experts (MoE) transformers route tokens to expert subnetworks within each layer, but the layer structure itself remains monolithic. We introduce Mixture of Layers (MoL), which replaces full-width transformer blocks (d_model) with K parallel thin blocks at reduced dimensionality (d_thin &lt;&lt; d_model), connected via learned down/up projections and composed via top-k block routing. Scaling sparse block routing to many blocks creates an attention coverage problem, as each block sees fewer tokens. We address this by introducing hybrid attention, which pairs one shared softmax block for global context with Gated DeltaNet linear attention in routed blocks.

Subjects:

Machine Learning (cs.LG) ; Artificial Intelligence (cs.AI)

Cite as:

arXiv:2605.09516  [cs.LG]

&nbsp;

(or

arXiv:2605.09516v1  [cs.LG]  for this version)

&nbsp;

https://doi.org/10.48550/arXiv.2605.09516

Focus to learn more

arXiv-issued DOI via DataCite

Submission history  From: Ivan Ternovtsii [ view email ]

[v1]

Sun, 10 May 2026 12:53:28 UTC (42 KB)

Full-text links:

Access Paper:

View a PDF of the paper titled Mixture of Layers with Hybrid Attention, by Ivan Ternovtsii and 1 other authors

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