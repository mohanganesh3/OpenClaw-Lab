[2605.29395] Low Rank for Rank: Uncertainty-Aware Task-Specific LLM Ranking under Sparse Pairwise Comparisons

Skip to main content

Learn about arXiv becoming an independent nonprofit.

We gratefully acknowledge support from the Simons Foundation,  member institutions , and all contributors.

Donate

&gt;

stat

&gt;  arXiv:2605.29395

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

Statistics > Methodology

arXiv:2605.29395  (stat)

[Submitted on 28 May 2026]

Title: Low Rank for Rank: Uncertainty-Aware Task-Specific LLM Ranking under Sparse Pairwise Comparisons

Authors:  Jiachun Li ,  David Simchi-Levi ,  Will Wei Sun

View a PDF of the paper titled Low Rank for Rank: Uncertainty-Aware Task-Specific LLM Ranking under Sparse Pairwise Comparisons, by Jiachun Li and 2 other authors

View PDF

HTML (experimental)

Abstract: Pairwise human-preference platforms such as Chatbot Arena have become central to large language model (LLM) evaluation, yet reliable task-specific ranking remains challenging. Global leaderboards mask task heterogeneity, while ranking each fine-grained task independently is unstable under sparse, imbalanced comparisons. We propose a low-rank framework for task-specific LLM ranking from sparse pairwise comparisons, modeling the task-by-model ability matrix $\Theta^\star \in \mathbb{R}^{d_t \times d_m}$ as low rank so that information is shared across related tasks while task-specific differences are preserved. We first develop a max-norm ($\ell_\infty$) accurate estimator for the latent scores, combining a convex initializer with alternating-minimization refinement, and prove task-wise top-$K$ recovery guarantees under sparse sampling. Our main contribution is an uncertainty quantification framework for task-specific ranking. We construct cross-fitted one-step debiased estimators for fixed score contrasts -- such as the task-specific ability gap between two models -- yielding asymptotically valid confidence intervals that attain the semiparametric efficiency bound. We then extend the inference to the high-dimensional ranking regime, where per-task ranks and top-$K$ membership are determined by many dependent score-gap hypotheses. Using Gaussian and multiplier-bootstrap calibration, we obtain simultaneous confidence sets for per-task ranks and valid top-$K$ membership tests across many tasks and models. Experiments on synthetic data and Chatbot Arena show that low-rank sharing improves sample efficiency over independent task-wise Bradley-Terry estimation and produces tighter, better-calibrated ranking certificates, with the largest gains in the sparse regime typical of real LLM benchmarks.

Subjects:

Methodology (stat.ME) ; Machine Learning (stat.ML)

Cite as:

arXiv:2605.29395  [stat.ME]

&nbsp;

(or

arXiv:2605.29395v1  [stat.ME]  for this version)

&nbsp;

https://doi.org/10.48550/arXiv.2605.29395

Focus to learn more

arXiv-issued DOI via DataCite

Submission history  From: Jiachun Li [ view email ]

[v1]

Thu, 28 May 2026 05:44:43 UTC (132 KB)

Full-text links:

Access Paper:

View a PDF of the paper titled Low Rank for Rank: Uncertainty-Aware Task-Specific LLM Ranking under Sparse Pairwise Comparisons, by Jiachun Li and 2 other authors

View PDF

HTML (experimental)

TeX Source

view license

Current browse context:

stat.ME

&lt;&nbsp;prev

&nbsp; | &nbsp;

next&nbsp;&gt;

new

|

recent

|

2026-05

Change to browse by:

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