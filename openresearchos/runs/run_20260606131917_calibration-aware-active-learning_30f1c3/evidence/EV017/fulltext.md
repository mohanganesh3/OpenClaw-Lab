[2605.09790] Multi-Tier Labeling and Physics-Informed Learning for Orbital Anomaly Detection at Scale

Skip to main content

Learn about arXiv becoming an independent nonprofit.

We gratefully acknowledge support from the Simons Foundation,  member institutions , and all contributors.

Donate

&gt;

cs

&gt;  arXiv:2605.09790

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

Computer Science > Distributed, Parallel, and Cluster Computing

arXiv:2605.09790  (cs)

[Submitted on 10 May 2026]

Title: Multi-Tier Labeling and Physics-Informed Learning for Orbital Anomaly Detection at Scale

Authors:  Yong Fu

View a PDF of the paper titled Multi-Tier Labeling and Physics-Informed Learning for Orbital Anomaly Detection at Scale, by Yong Fu

View PDF

HTML (experimental)

Abstract: Detecting orbital anomalies, such as maneuvers, atmospheric decay, and attitude upsets, across the rapidly growing population of low-Earth-orbit (LEO) satellites is a prerequisite for collision avoidance, decay forecasting, and conjunction screening. The bottleneck is not modeling capacity but labels: there is no public ground-truth corpus of orbital anomalies, manual review does not scale to approximately 10^4 active satellites, and pure rule-based detectors trade recall for precision so aggressively that they are blind to most behavioral anomalies. We present a multi-tier labeling cascade that composes three weak supervision sources of increasing fidelity: a fast physics rule set (rule_v1), an Interacting Multiple Model Unscented Kalman Filter (IMM-UKF) bank, and a supplemental-element calibration step (supGP), to produce labels at a scale unavailable from any single source. Applied to 232M Two-Line Element (TLE) records spanning 60 years, the cascade yields 8.6M labeled sequences of length 50 (430M timesteps) over 11 features that include explicit time encoding and full mean-element state. On overlapping satellites, IMM-UKF surfaces 42.6x more anomalies than rule_v1 alone. We train a 6.5M-parameter Transformer in two stages, achieving a maneuver recall of 55.4% and decay recall of 62.8% on a held-out test set. An ablation on the time-delta feature alone yields a 107% relative improvement in decay recall. We frame the resulting model as a high-recall triage classifier whose role is to surface candidate events for downstream filtering, not to issue final attributions, and discuss the path toward a Neural-ODE-based orbital world model.

Subjects:

Distributed, Parallel, and Cluster Computing (cs.DC) ; Artificial Intelligence (cs.AI); Machine Learning (cs.LG)

Cite as:

arXiv:2605.09790  [cs.DC]

&nbsp;

(or

arXiv:2605.09790v1  [cs.DC]  for this version)

&nbsp;

https://doi.org/10.48550/arXiv.2605.09790

Focus to learn more

arXiv-issued DOI via DataCite (pending registration)

Submission history  From: Yong Fu [ view email ]

[v1]

Sun, 10 May 2026 22:20:52 UTC (19 KB)

Full-text links:

Access Paper:

View a PDF of the paper titled Multi-Tier Labeling and Physics-Informed Learning for Orbital Anomaly Detection at Scale, by Yong Fu

View PDF

HTML (experimental)

TeX Source

view license

Current browse context:

cs.DC

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

cs.LG

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