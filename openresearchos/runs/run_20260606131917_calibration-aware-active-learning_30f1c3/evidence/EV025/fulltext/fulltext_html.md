[2605.11551] VNDUQE: Information-Theoretic Novelty Detection using Deep Variational Information Bottleneck

Skip to main content

Learn about arXiv becoming an independent nonprofit.

We gratefully acknowledge support from the Simons Foundation,  member institutions , and all contributors.

Donate

&gt;

cs

&gt;  arXiv:2605.11551

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

arXiv:2605.11551  (cs)

[Submitted on 12 May 2026]

Title: VNDUQE: Information-Theoretic Novelty Detection using Deep Variational Information Bottleneck

Authors:  Aryan Gondkar ,  Hayder Radha ,  Yiming Deng

View a PDF of the paper titled VNDUQE: Information-Theoretic Novelty Detection using Deep Variational Information Bottleneck, by Aryan Gondkar and 2 other authors

View PDF

HTML (experimental)

Abstract: Detecting out-of-distribution (OOD) samples is critical for safe deployment of neural networks in safety-critical applications. While maximum softmax probability (MSP) provides a simple baseline, it lacks theoretical grounding and suffers from miscalibration. We propose VNDUQE (VIB-based Novelty Detection and Uncertainty Quantification for Nondestructive Evaluation), which investigates novelty detection through the Deep Variational Information Bottleneck (VIB), which explicitly constrains information flow through learned representations. We train VIB models on MNIST with held-out digit classes and evaluate OOD detection using information-theoretic metrics: KL divergence and prediction entropy. Our results reveal complementary detection signals: KL divergence achieves perfect detection (100\% AUROC on noise) on far-OOD samples (noise, domain shift), while prediction entropy excels at near-OOD detection (94.7\% AUROC on novel digit classes). A parallel detection strategy combining both metrics achieves 95.3\% average AUROC and 92\% true positive rate at 5\% false positive rate, which is a 32 percentage point improvement over baseline MSP (85.0\% AUROC, 60.1\% TPR). Compression via the information bottleneck principle ($\beta=10^{-3}$) reduces Expected Calibration Error by 38\%, demonstrating that information-theoretic constraints produce fundamentally more reliable uncertainty estimates. These findings directly support active learning with expensive computational oracles, where well-calibrated novelty detection enables principled threshold selection for oracle queries.

Comments:

6 pages, 3 figures, Fall 2025 version

Subjects:

Machine Learning (cs.LG) ; Computer Vision and Pattern Recognition (cs.CV); Information Theory (cs.IT)

Cite as:

arXiv:2605.11551  [cs.LG]

&nbsp;

(or

arXiv:2605.11551v1  [cs.LG]  for this version)

&nbsp;

https://doi.org/10.48550/arXiv.2605.11551

Focus to learn more

arXiv-issued DOI via DataCite (pending registration)

Submission history  From: Aryan Gondkar [ view email ]

[v1]

Tue, 12 May 2026 05:31:44 UTC (203 KB)

Full-text links:

Access Paper:

View a PDF of the paper titled VNDUQE: Information-Theoretic Novelty Detection using Deep Variational Information Bottleneck, by Aryan Gondkar and 2 other authors

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

cs.CV

cs.IT

math

math.IT

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