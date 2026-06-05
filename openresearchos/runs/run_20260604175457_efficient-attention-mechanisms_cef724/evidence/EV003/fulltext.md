[2606.03825] Dynamic Short Convolutions Improve Transformers

Skip to main content

Learn about arXiv becoming an independent nonprofit.

We gratefully acknowledge support from the Simons Foundation,  member institutions , and all contributors.

Donate

&gt;

cs

&gt;  arXiv:2606.03825

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

arXiv:2606.03825  (cs)

[Submitted on 2 Jun 2026]

Title: Dynamic Short Convolutions Improve Transformers

Authors:  Oliver Sieberling ,  Bharat Runwal ,  Rameswar Panda ,  Yoon Kim

View a PDF of the paper titled Dynamic Short Convolutions Improve Transformers, by Oliver Sieberling and 3 other authors

View PDF

HTML (experimental)

Abstract: Transformers have become the dominant architecture for large language models, largely due to the scalability and flexibility of attention, feed-forward layers, residual connections, and normalization. This paper introduces dynamic short convolutions as an additional neural network primitive for improving Transformers. Unlike static short convolutions, dynamic convolutions use input-dependent filters, which preserves the locality bias of convolution while increasing expressivity. Motivating experiments show that applying dynamic short convolutions to key, query, and value representations improves performance on challenging associative recall tasks compared with static convolutional variants. Across language-modeling experiments ranging from 150M to 2B parameters, dynamic convolutions consistently outperform standard Transformers and Transformers augmented with static short convolutions. Fitting scaling laws indicates a 1.33$\times$ compute advantage over compute-matched Transformers when dynamic convolutions are applied to the key, query, and value vectors, and a 1.60$\times$ advantage when adding dynamic convolutions after every linear layer. Dynamic convolutions also offer improvements on linear RNNs (Mamba-2/Gated DeltaNet) and mixture-of-experts architectures. We make these gains practical with custom Triton kernels that enable efficient training with a manageable end-to-end slowdown. These results suggest that dynamic short convolutions are a scalable, hardware-efficient, and expressive primitive for advancing Transformer-based language models.

Subjects:

Machine Learning (cs.LG) ; Computation and Language (cs.CL)

Cite as:

arXiv:2606.03825  [cs.LG]

&nbsp;

(or

arXiv:2606.03825v1  [cs.LG]  for this version)

&nbsp;

https://doi.org/10.48550/arXiv.2606.03825

Focus to learn more

arXiv-issued DOI via DataCite (pending registration)

Submission history  From: Oliver Sieberling [ view email ]

[v1]

Tue, 2 Jun 2026 16:07:55 UTC (88 KB)

Full-text links:

Access Paper:

View a PDF of the paper titled Dynamic Short Convolutions Improve Transformers, by Oliver Sieberling and 3 other authors

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