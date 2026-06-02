[2605.01647] Beyond Perplexity: Character Distribution Signatures and the MDTA Benchmark for AI Text Detection

Skip to main content

Learn about arXiv becoming an independent nonprofit.

We gratefully acknowledge support from the Simons Foundation,  member institutions , and all contributors.

Donate

&gt;

cs

&gt;  arXiv:2605.01647

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

arXiv:2605.01647  (cs)

[Submitted on 3 May 2026]

Title: Beyond Perplexity: Character Distribution Signatures and the MDTA Benchmark for AI Text Detection

Authors:  Priyadarshan Narayanasamy ,  Swastik Agrawal ,  Klint Faber ,  Fardina Fathmiul Alam

View a PDF of the paper titled Beyond Perplexity: Character Distribution Signatures and the MDTA Benchmark for AI Text Detection, by Priyadarshan Narayanasamy and 3 other authors

View PDF

HTML (experimental)

Abstract: Training-free AI text detection methods primarily rely on model log-probabilities, achieving strong performance through approaches like Binoculars and DNA-DetectLLM. However, these methods face a fundamental ceiling as models are optimized through RLHF to produce human-like probability distributions. We introduce an alternative detection signal based on character distribution signatures. We provide theoretical foundations showing that AI models, trained on massive domain-balanced corpora, approximate global character patterns while humans exhibit domain-specialized distributions, creating a &#34;Wall of Separation&#34; where human-AI divergence significantly exceeds AI-AI divergence. To enable systematic evaluation, we construct the Models-Domains-Temperatures-Adversarials (MDTA) benchmark comprising 642,274 prompt-aligned samples across 4 models, 5 domains, 3 temperature settings, and 3 adversarial strategies, substantially expanding the HC3 dataset with modern model responses, temperature variation, and adversarial augmentation. We introduce the Letter Distribution Score (LD-Score), demonstrating low correlation (r = 0.08-0.13) with perplexity methods. When integrated with DNA-DetectLLM, Binoculars and FastDetectGPT via a non-linear classifier, LD-Score yields consistent improvements in AUROC and F1, with particularly pronounced gains in specialized domains where vocabulary constraints amplify the detection signal. The MDTA dataset can be accessed at:  this https URL .

Comments:

11 figures, 10 tables, 24 pages, Under Review at COLM 2026

Subjects:

Computation and Language (cs.CL)

Cite as:

arXiv:2605.01647  [cs.CL]

&nbsp;

(or

arXiv:2605.01647v1  [cs.CL]  for this version)

&nbsp;

https://doi.org/10.48550/arXiv.2605.01647

Focus to learn more

arXiv-issued DOI via DataCite

Submission history  From: Swastik Agrawal [ view email ]

[v1]

Sun, 3 May 2026 00:10:15 UTC (10,913 KB)

Full-text links:

Access Paper:

View a PDF of the paper titled Beyond Perplexity: Character Distribution Signatures and the MDTA Benchmark for AI Text Detection, by Priyadarshan Narayanasamy and 3 other authors

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