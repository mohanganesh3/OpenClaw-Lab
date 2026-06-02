[2605.28112] A Wolf in Sheep&#39;s Clothing: Targeted Routing Hijacking in Federated RAG

Skip to main content

Learn about arXiv becoming an independent nonprofit.

We gratefully acknowledge support from the Simons Foundation,  member institutions , and all contributors.

Donate

&gt;

cs

&gt;  arXiv:2605.28112

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

Computer Science > Cryptography and Security

arXiv:2605.28112  (cs)

[Submitted on 27 May 2026]

Title: A Wolf in Sheep&#39;s Clothing: Targeted Routing Hijacking in Federated RAG

Authors:  Junjie Mu ,  Qiongxiu Li

View a PDF of the paper titled A Wolf in Sheep&#39;s Clothing: Targeted Routing Hijacking in Federated RAG, by Junjie Mu and Qiongxiu Li

View PDF

HTML (experimental)

Abstract: Federated Retrieval-Augmented Generation (FedRAG) is attractive for privacy-sensitive applications because raw data remain local. As a result, routing must rely on client-provided semantic profiles, creating a new opportunity for manipulation. We introduce Routing Hijacking, a routing-stage attack in which a malicious client forges its profile to attract target queries despite having irrelevant underlying data. We show that this vulnerability is severe. Across three representative FedRAG routing architectures, Routing Hijacking consistently misroutes target queries and leads to downstream disruptions and failures, including missing evidence, poisoning, incorrect answers, and hallucinations. In a high-stakes MedQA-USMLE case study, we further show that poisoned retrieved evidence can mislead models across scales, leading to incorrect answers, hallucinations, and sycophantic failures. Existing defenses do not close this gap: encrypted routing preserves the exploited ranking, and Byzantine-robust Federated Learning (FL) rules transfer poorly to heterogeneous routing profiles. To address this gap, we propose a trust-aware post-routing framework that reweights clients using returned-evidence feedback, including retrieval relevance, profile consistency, and cross-client agreement; online experiments show that it suppresses persistent hijacking over recurring queries and transfers to a learned neural router. Our findings establish routing integrity as a new security challenge in FedRAG and highlight the need for stronger defenses for secure federated retrieval.

Comments:

Under review. Code available at  this https URL

Subjects:

Cryptography and Security (cs.CR) ; Computation and Language (cs.CL); Information Retrieval (cs.IR)

Cite as:

arXiv:2605.28112  [cs.CR]

&nbsp;

(or

arXiv:2605.28112v1  [cs.CR]  for this version)

&nbsp;

https://doi.org/10.48550/arXiv.2605.28112

Focus to learn more

arXiv-issued DOI via DataCite

Submission history  From: Junjie Mu [ view email ]

[v1]

Wed, 27 May 2026 08:06:10 UTC (1,713 KB)

Full-text links:

Access Paper:

View a PDF of the paper titled A Wolf in Sheep&#39;s Clothing: Targeted Routing Hijacking in Federated RAG, by Junjie Mu and Qiongxiu Li

View PDF

HTML (experimental)

TeX Source

view license

Current browse context:

cs.CR

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

cs.CL

cs.IR

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