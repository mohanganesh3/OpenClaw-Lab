[2605.17626] Verifier-Guided Code Translation via Meta-Step Decoding

Skip to main content

Learn about arXiv becoming an independent nonprofit.

We gratefully acknowledge support from the Simons Foundation,  member institutions , and all contributors.

Donate

&gt;

cs

&gt;  arXiv:2605.17626

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

arXiv:2605.17626  (cs)

[Submitted on 17 May 2026]

Title: Verifier-Guided Code Translation via Meta-Step Decoding

Authors:  Tianyang Zhou ,  Somesh Jha ,  Mihai Christodorescu ,  Kirill Levchenko ,  Varun Chandrasekaran

View a PDF of the paper titled Verifier-Guided Code Translation via Meta-Step Decoding, by Tianyang Zhou and 4 other authors

View PDF

HTML (experimental)

Abstract: Test-time scaling is an important mechanism for improving large language models, especially on tasks with deterministic verifiers. Code translation is a canonical example: the source program constrains valid outputs, while compilers, type check- ers, and behavioral checks provide exact pass/fail feedback. Existing approaches typically apply these verifiers only after generation, which is inefficient because early errors corrupt the autoregressive context and are rarely corrected later. We introduce Decoding Time Verification (DTV), a framework that treats structural boundaries as meta steps for verifier-guided decoding. DTV interleaves generation with verifier calls under a state-machine controller that enforces valid prefixes, using structural-boundary checks and structure-aware rollback to prevent error propagation while reducing wasted tokens. We evaluate DTV on C-to-Rust and JavaScript-to-TypeScript translation. Using Qwen3-4B as the primary generator under matched token budgets, DTV improves pass rates from 72.3% to 82.0% on C-to-Rust and from 33.3% to 46.0% on JavaScript-to-TypeScript relative to matched self-refinement baselines, while using fewer tokens per case; the same trend largely transfers to Gemma-4-E4B. In the evaluated cost-matched grid, DTV achieves a more favorable pass-rate-cost tradeoff than post-hoc verification or sampling-based scaling. These results show that verifier-guided decoding is an effective use of inference-time compute for code translation.

Comments:

31 pages, 8 figures

Subjects:

Machine Learning (cs.LG) ; Software Engineering (cs.SE)

Cite as:

arXiv:2605.17626  [cs.LG]

&nbsp;

(or

arXiv:2605.17626v1  [cs.LG]  for this version)

&nbsp;

https://doi.org/10.48550/arXiv.2605.17626

Focus to learn more

arXiv-issued DOI via DataCite

Submission history  From: Tianyang Zhou [ view email ]

[v1]

Sun, 17 May 2026 19:47:07 UTC (341 KB)

Full-text links:

Access Paper:

View a PDF of the paper titled Verifier-Guided Code Translation via Meta-Step Decoding, by Tianyang Zhou and 4 other authors

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