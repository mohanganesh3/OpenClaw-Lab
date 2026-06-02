[2110.06751] Improving the sample-efficiency of neural architecture search with reinforcement learning

Skip to main content

Learn about arXiv becoming an independent nonprofit.

We gratefully acknowledge support from the Simons Foundation,  member institutions , and all contributors.

Donate

&gt;

cs

&gt;  arXiv:2110.06751

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

arXiv:2110.06751  (cs)

[Submitted on 13 Oct 2021]

Title: Improving the sample-efficiency of neural architecture search with reinforcement learning

Authors:  Attila Nagy ,  Ábel Boros

View a PDF of the paper titled Improving the sample-efficiency of neural architecture search with reinforcement learning, by Attila Nagy and 1 other authors

View PDF

Abstract: Designing complex architectures has been an essential cogwheel in the revolution deep learning has brought about in the past decade. When solving difficult problems in a datadriven manner, a well-tried approach is to take an architecture discovered by renowned deep learning scientists as a basis (e.g. Inception) and try to apply it to a specific problem. This might be sufficient, but as of now, achieving very high accuracy on a complex or yet unsolved task requires the knowledge of highly-trained deep learning experts. In this work, we would like to contribute to the area of Automated Machine Learning (AutoML), specifically Neural Architecture Search (NAS), which intends to make deep learning methods available for a wider range of society by designing neural topologies automatically. Although several different approaches exist (e.g. gradient-based or evolutionary algorithms), our focus is on one of the most promising research directions, reinforcement learning. In this scenario, a recurrent neural network (controller) is trained to create problem-specific neural network architectures (child). The validation accuracies of the child networks serve as a reward signal for training the controller with reinforcement learning. The basis of our proposed work is Efficient Neural Architecture Search (ENAS), where parameter sharing is applied among the child networks. ENAS, like many other RL-based algorithms, emphasize the learning of child networks as increasing their convergence result in a denser reward signal for the controller, therefore significantly reducing training times. The controller was originally trained with REINFORCE. In our research, we propose to modify this to a more modern and complex algorithm, PPO, which has demonstrated to be faster and more stable in other environments. Then, we briefly discuss and evaluate our results.

Subjects:

Machine Learning (cs.LG)

Cite as:

arXiv:2110.06751  [cs.LG]

&nbsp;

(or

arXiv:2110.06751v1  [cs.LG]  for this version)

&nbsp;

https://doi.org/10.48550/arXiv.2110.06751

Focus to learn more

arXiv-issued DOI via DataCite

Submission history  From: Attila Nagy [ view email ]

[v1]

Wed, 13 Oct 2021 14:30:09 UTC (1,414 KB)

Full-text links:

Access Paper:

View a PDF of the paper titled Improving the sample-efficiency of neural architecture search with reinforcement learning, by Attila Nagy and 1 other authors

View PDF

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

2021-10

Change to browse by:

cs

References &amp; Citations

NASA ADS

Google Scholar

Semantic Scholar

DBLP  - CS Bibliography

listing  |  bibtex

Attila Nagy

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