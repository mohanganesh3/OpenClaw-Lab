[2605.02495] Efficient Preference Poisoning Attack on Offline RLHF

Skip to main content

Learn about arXiv becoming an independent nonprofit.

We gratefully acknowledge support from the Simons Foundation,  member institutions , and all contributors.

Donate

&gt;

cs

&gt;  arXiv:2605.02495

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

arXiv:2605.02495  (cs)

[Submitted on 4 May 2026 ( v1 ), last revised 25 May 2026 (this version, v2)]

Title: Efficient Preference Poisoning Attack on Offline RLHF

Authors:  Chenye Yang ,  Weiyu Xu ,  Lifeng Lai

View a PDF of the paper titled Efficient Preference Poisoning Attack on Offline RLHF, by Chenye Yang and 2 other authors

View PDF

HTML (experimental)

Abstract: Offline Reinforcement Learning from Human Feedback (RLHF) pipelines such as Direct Preference Optimization (DPO) train on a pre-collected preference dataset, which makes them vulnerable to preference poisoning attack. We study label flip attacks against log-linear DPO. We first illustrate that flipping one preference label induces a parameter-independent shift in the DPO gradient. Using this key property, we can then convert the targeted poisoning problem into a structured binary sparse approximation problem. To solve this problem, we develop two attack methods: Binary-Aware Lattice Attack (BAL-A) and Binary Matching Pursuit Attack (BMP-A). BAL-A embeds the binary flip selection problem into a binary-aware lattice and applies Lenstra-Lenstra-Lovász reduction and Babai&#39;s nearest plane algorithm; we provide sufficient conditions that enforce binary coefficients and recover the minimum-flip objective. BMP-A adapts binary matching pursuit to our non-normalized gradient dictionary and yields coherence-based recovery guarantees and robustness (impossibility) certificates for $K$-flip budgets. Experiments on synthetic dictionaries and the Stanford Human Preferences dataset validate the theory and highlight how dictionary geometry governs attack success.

Comments:

Accepted to ICML 2026

Subjects:

Machine Learning (cs.LG) ; Artificial Intelligence (cs.AI); Machine Learning (stat.ML)

Cite as:

arXiv:2605.02495  [cs.LG]

&nbsp;

(or

arXiv:2605.02495v2  [cs.LG]  for this version)

&nbsp;

https://doi.org/10.48550/arXiv.2605.02495

Focus to learn more

arXiv-issued DOI via DataCite

Submission history  From: Chenye Yang [ view email ]

[v1]

Mon, 4 May 2026 11:45:38 UTC (1,686 KB)

[v2]

Mon, 25 May 2026 07:21:41 UTC (1,791 KB)

Full-text links:

Access Paper:

View a PDF of the paper titled Efficient Preference Poisoning Attack on Offline RLHF, by Chenye Yang and 2 other authors

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

stat

stat.ML

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