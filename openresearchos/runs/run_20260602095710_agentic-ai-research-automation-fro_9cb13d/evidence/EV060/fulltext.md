[2606.02475] Physics-Informed Residuals for Adaptive Mesh Refinement in Finite-Difference PDE Solvers

Skip to main content

Learn about arXiv becoming an independent nonprofit.

We gratefully acknowledge support from the Simons Foundation,  member institutions , and all contributors.

Donate

&gt;

math

&gt;  arXiv:2606.02475

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

Mathematics > Numerical Analysis

arXiv:2606.02475  (math)

[Submitted on 1 Jun 2026]

Title: Physics-Informed Residuals for Adaptive Mesh Refinement in Finite-Difference PDE Solvers

Authors:  Henry Kasumba ,  Ronald Katende

View a PDF of the paper titled Physics-Informed Residuals for Adaptive Mesh Refinement in Finite-Difference PDE Solvers, by Henry Kasumba and Ronald Katende

View PDF

HTML (experimental)

Abstract: Classical finite-difference solvers remain reliable tools for partial differential equations, but their efficiency depends on where mesh resolution is placed. Uniform refinement can waste degrees of freedom when solution difficulty is localised near sharp gradients, fronts, oscillations, or constraint-sensitive regions. This paper studies a hybrid strategy in which a physics-informed neural network (PINN) is used not as the final solver, but as an off-grid residual probe for adaptive mesh refinement. The PINN residual is sampled over the domain, converted into cellwise indicators, and used to guide refinement before the final approximation is computed by a finite-difference solver.
 The method is evaluated on three benchmarks. The main full-solver validation uses the one-dimensional viscous Burgers equation with a nonuniform finite-difference solve on the adapted meshes. PINN-threshold refinement attains final relative $L^2$ error $0.021067$ with $60$ degrees of freedom, compared with $0.022617$ for uniform refinement with $192$ degrees of freedom. At matched mesh size, PINN-threshold reduces the error by about $67.5\%$. PINN-D&#34;orfler refinement gives similar performance, with error $0.021264$ using $58$ degrees of freedom. A gradient indicator remains slightly more accurate, so the result supports usefulness rather than universal superiority. Manufactured 2D and 3D proxy tests, based on a nonlinear Schr&#34;odinger equation and an incompressible Navier--Stokes system, show that PINN residuals can organise structured refinement and improve over random refinement, although they do not consistently outperform gradient or uniform baselines. The results support PINN-guided AMR as a residual-indicator strategy for transferring physics-informed diagnostic information into finite-difference mesh adaptation while preserving the classical solver as the final approximation engine.

Comments:

17 pages, 5 tables, 5 figures

Subjects:

Numerical Analysis (math.NA) ; Computational Engineering, Finance, and Science (cs.CE); Machine Learning (cs.LG)

MSC  classes:

65M06, 65M22, 65M50, 65M60, 35Q30, 35Q55, 76D05

Cite as:

arXiv:2606.02475  [math.NA]

&nbsp;

(or

arXiv:2606.02475v1  [math.NA]  for this version)

&nbsp;

https://doi.org/10.48550/arXiv.2606.02475

Focus to learn more

arXiv-issued DOI via DataCite (pending registration)

Submission history  From: Ronald Katende [ view email ]

[v1]

Mon, 1 Jun 2026 16:47:21 UTC (1,104 KB)

Full-text links:

Access Paper:

View a PDF of the paper titled Physics-Informed Residuals for Adaptive Mesh Refinement in Finite-Difference PDE Solvers, by Henry Kasumba and Ronald Katende

View PDF

HTML (experimental)

TeX Source

view license

Current browse context:

math.NA

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

cs.CE

cs.LG

cs.NA

math

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