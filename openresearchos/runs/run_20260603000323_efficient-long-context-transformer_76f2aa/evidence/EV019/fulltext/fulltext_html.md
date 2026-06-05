[2605.20683] Layer-wise Token Compression for Efficient Document Reranking

Skip to main content

Learn about arXiv becoming an independent nonprofit.

We gratefully acknowledge support from the Simons Foundation,  member institutions , and all contributors.

Donate

&gt;

cs

&gt;  arXiv:2605.20683

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

Computer Science > Information Retrieval

arXiv:2605.20683  (cs)

[Submitted on 20 May 2026 ( v1 ), last revised 21 May 2026 (this version, v2)]

Title: Layer-wise Token Compression for Efficient Document Reranking

Authors:  Shengyao Zhuang ,  Zhichao Xu ,  Ivano Lauriola

View a PDF of the paper titled Layer-wise Token Compression for Efficient Document Reranking, by Shengyao Zhuang and 2 other authors

View PDF

HTML (experimental)

Abstract: Transformer-based document cross-encoder rerankers are a central component of modern information retrieval systems. Despite their success, these models suffer from high computational costs due to processing long query-document sequences at inference time. A known approach to improve efficiency is token compression, which consists of aggregating groups of tokens together in the initial embedding layer, reducing the effective number of tokens, and making the computation faster. While token compression has proven to be successful for bi-encoder retrievers, we empirically observed that this approach may be ineffective for cross-encoder rerankers. In this paper, we propose Layer-wise Token Compression (LTC), which applies adaptive token pooling at intermediate transformer layers. Through extensive ablation studies on MS MARCO passage and document ranking tasks, we demonstrate that compression at middle layers preserves ranking quality while increasing inference QPS by up to 25% for passage ranking and up to 116% for document ranking. We also extend LTC to listwise LLM rerankers and show that the same approach can be easily applied to long-context listwise reranking, where the QPS improvements are even greater. More surprisingly, when applying rerankers trained on short passages to long-document ranking tasks, models trained with compression outperform their uncompressed counterparts, suggesting that compression may act as a beneficial regularizer that encourages length-invariant representations.

Comments:

SIGIR2026 short paper

Subjects:

Information Retrieval (cs.IR)

Cite as:

arXiv:2605.20683  [cs.IR]

&nbsp;

(or

arXiv:2605.20683v2  [cs.IR]  for this version)

&nbsp;

https://doi.org/10.48550/arXiv.2605.20683

Focus to learn more

arXiv-issued DOI via DataCite

Related DOI :

https://doi.org/10.1145/3805712.3809871

Focus to learn more

DOI(s) linking to related resources

Submission history  From: Shengyao Zhuang [ view email ]

[v1]

Wed, 20 May 2026 03:52:31 UTC (1,356 KB)

[v2]

Thu, 21 May 2026 04:53:43 UTC (1,356 KB)

Full-text links:

Access Paper:

View a PDF of the paper titled Layer-wise Token Compression for Efficient Document Reranking, by Shengyao Zhuang and 2 other authors

View PDF

HTML (experimental)

TeX Source

view license

Additional Features

Audio Summary

Current browse context:

cs.IR

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