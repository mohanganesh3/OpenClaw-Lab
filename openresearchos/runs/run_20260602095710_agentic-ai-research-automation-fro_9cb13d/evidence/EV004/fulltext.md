[2605.05287] Securing the Agent: Vendor-Neutral, Multitenant Enterprise Retrieval and Tool Use

Skip to main content

Learn about arXiv becoming an independent nonprofit.

We gratefully acknowledge support from the Simons Foundation,  member institutions , and all contributors.

Donate

&gt;

cs

&gt;  arXiv:2605.05287

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

arXiv:2605.05287  (cs)

[Submitted on 6 May 2026]

Title: Securing the Agent: Vendor-Neutral, Multitenant Enterprise Retrieval and Tool Use

Authors:  Francisco Javier Arceo ,  Varsha Prasad Narsing

View a PDF of the paper titled Securing the Agent: Vendor-Neutral, Multitenant Enterprise Retrieval and Tool Use, by Francisco Javier Arceo and 1 other authors

View PDF

HTML (experimental)

Abstract: Retrieval-Augmented Generation (RAG) and agentic AI systems are increasingly prevalent in enterprise AI deployments. However, real enterprise environments introduce challenges largely absent from academic treatments and consumer-facing APIs: multiple tenants with heterogeneous data, strict access-control requirements, regulatory compliance, and cost pressures that demand shared infrastructure.
 A fundamental problem underlies existing RAG architectures in these settings: retrieval systems rank documents by relevance--whether through semantic similarity, keyword matching, or hybrid approaches--not by authorization, so a query from one tenant can surface another tenant&#39;s confidential data simply because it scores highest. We formalize this gap and analyze additional shortcomings--including tool-mediated disclosure, context accumulation across turns, and client-side orchestration bypass--that arise when agentic systems conflate relevance with authorization. To address these challenges, we introduce a layered isolation architecture combining policy-aware ingestion, retrieval-time gating, and shared inference, enforced through server-side agentic orchestration. This approach centralizes security-critical operations--tool execution authorization, state isolation, and policy enforcement--on the server, creating natural enforcement points for multitenant isolation while allowing client-side frameworks to retain control over agent composition and latency-sensitive operations.
 We validate the proposed architecture through an open-source implementation in OGX, a vendor-neutral framework that implements an OpenAI-compatible, open-source Responses API with server-side multi-turn orchestration. We evaluate it empirically and show that ABAC gating eliminates cross-tenant leakage while introducing negligible overhead.

Comments:

11 pages, 2 figures, Published in ACM Conference on AI and Agentic Systems

Subjects:

Cryptography and Security (cs.CR) ; Artificial Intelligence (cs.AI); Information Retrieval (cs.IR); Software Engineering (cs.SE)

ACM &nbsp;classes:

I.2; H.3; K.6

Cite as:

arXiv:2605.05287  [cs.CR]

&nbsp;

(or

arXiv:2605.05287v1  [cs.CR]  for this version)

&nbsp;

https://doi.org/10.48550/arXiv.2605.05287

Focus to learn more

arXiv-issued DOI via DataCite (pending registration)

Journal&nbsp;reference:

ACM Conference on AI and Agentic Systems (ACM CAIS &#39;26), May 26-29, 2026, San Jose, CA, USA

Related DOI :

https://doi.org/10.1145/3786335.3813145

Focus to learn more

DOI(s) linking to related resources

Submission history  From: Francisco Javier Arceo [ view email ]

[v1]

Wed, 6 May 2026 17:59:21 UTC (62 KB)

Full-text links:

Access Paper:

View a PDF of the paper titled Securing the Agent: Vendor-Neutral, Multitenant Enterprise Retrieval and Tool Use, by Francisco Javier Arceo and 1 other authors

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

cs.AI

cs.IR

cs.SE

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