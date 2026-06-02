[2606.02528] Auditing Asset-Specific Preferences in Financial Large Language Models: Evidence from Bitcoin Representations and Portfolio Allocation

Skip to main content

Learn about arXiv becoming an independent nonprofit.

We gratefully acknowledge support from the Simons Foundation,  member institutions , and all contributors.

Donate

&gt;

q-fin

&gt;  arXiv:2606.02528

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

Quantitative Finance > General Finance

arXiv:2606.02528  (q-fin)

[Submitted on 1 Jun 2026]

Title: Auditing Asset-Specific Preferences in Financial Large Language Models: Evidence from Bitcoin Representations and Portfolio Allocation

Authors:  Wenbin Wu

View a PDF of the paper titled Auditing Asset-Specific Preferences in Financial Large Language Models: Evidence from Bitcoin Representations and Portfolio Allocation, by Wenbin Wu

View PDF

Abstract: Large language models now power robo-advisors and trading agents, yet whether they carry built-in biases toward specific assets is largely untested. We ask three questions: do LLMs systematically prefer certain financial instruments; can an internal representation with causal leverage over those preferences be identified; and does that representation affect downstream financial decisions?
 We develop a three-level audit protocol and apply it to Bitcoin. First, a behavioral audit of eight frontier LLMs shows that Bitcoin&#39;s ranking among money-like instruments is frame-dependent: models place it around rank 5 of 8 as &#34;reliable money&#34; but near the top under crisis and autonomous-agent frames, and an attribute-swap experiment confirms rankings track functional properties, not names. Second, we open a model&#39;s internals: a search across thousands of sparse-autoencoder features in Gemma 3 identifies a dominant Bitcoin-selective feature. Amplifying it shifts the model toward the asset and suppressing it shifts the model away, even when &#34;Bitcoin&#34; never appears in the prompt. Third, we test financial consequences: amplification raises Bitcoin&#39;s portfolio share by 5.2 percentage points while suppression lowers it by 4.6 pp, with amplification reallocating within crypto and suppression cutting total crypto exposure.
 We characterize this as bounded behavioral leverage (leverage meaning causal influence over outputs, not financial leverage): an identifiable internal feature can be perturbed to move financial choices, but only within measurable limits. The framework links internal representations to external recommendations, validated with random controls and mechanism boundaries. As LLMs become autonomous financial agents, this is a first step toward a behavioral layer for emerging know-your-agent (KYA) standards: knowing what an agent prefers, and how far that preference can be moved.

Comments:

28 pages, 5 figures, 18 tables

Subjects:

General Finance (q-fin.GN) ; Computers and Society (cs.CY); Machine Learning (cs.LG)

Cite as:

arXiv:2606.02528  [q-fin.GN]

&nbsp;

(or

arXiv:2606.02528v1  [q-fin.GN]  for this version)

&nbsp;

https://doi.org/10.48550/arXiv.2606.02528

Focus to learn more

arXiv-issued DOI via DataCite (pending registration)

Submission history  From: Wenbin Wu [ view email ]

[v1]

Mon, 1 Jun 2026 17:36:06 UTC (2,228 KB)

Full-text links:

Access Paper:

View a PDF of the paper titled Auditing Asset-Specific Preferences in Financial Large Language Models: Evidence from Bitcoin Representations and Portfolio Allocation, by Wenbin Wu

View PDF

view license

Current browse context:

q-fin.GN

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

cs.CY

cs.LG

q-fin

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