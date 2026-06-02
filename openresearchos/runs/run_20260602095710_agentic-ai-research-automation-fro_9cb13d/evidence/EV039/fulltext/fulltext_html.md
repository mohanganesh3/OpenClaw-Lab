[2606.02569] AdaCodec: A Predictive Visual Code for Video MLLMs

Skip to main content

Learn about arXiv becoming an independent nonprofit.

We gratefully acknowledge support from the Simons Foundation,  member institutions , and all contributors.

Donate

&gt;

cs

&gt;  arXiv:2606.02569

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

Computer Science > Computer Vision and Pattern Recognition

arXiv:2606.02569  (cs)

[Submitted on 1 Jun 2026]

Title: AdaCodec: A Predictive Visual Code for Video MLLMs

Authors:  Haowen Hou ,  Zhen Huang ,  Zheming Liang ,  Qingyi Si ,  Chenglin Li ,  Shuai Dong ,  Kele Shao ,  Ruilin Li ,  Dianyi Wang ,  Nan Duan ,  Jiaqi Wang

View a PDF of the paper titled AdaCodec: A Predictive Visual Code for Video MLLMs, by Haowen Hou and 10 other authors

View PDF

HTML (experimental)

Abstract: Video is temporally redundant: adjacent frames usually share most objects, background, and layout. Yet existing video multimodal large language models (video MLLMs) usually encode each sampled frame as an independent RGB image, causing visual tokens to repeat content already present in earlier frames. This suggests a more direct video interface: send a full reference frame only when the scene cannot be predicted well from prior context, and otherwise transmit a compact description of inter-frame changes. We call this interface a \emph{predictive visual code}, and instantiate it for video MLLMs as \textbf{AdaCodec}. AdaCodec spends full visual tokens on a reference frame only when its conditional predictive cost is high; otherwise, it encodes inter-frame changes, including motion and prediction residuals, as compact P-tokens. Across all eleven benchmarks, AdaCodec improves over the Qwen3-VL-8B per-frame RGB baseline at a matched visual-token budget. Even at $1/7$ the budget, AdaCodec with 32k tokens surpasses the 224k baseline on all long-video benchmarks; on five general-video benchmarks, it raises the average score while substantially cutting time-to-first-token from 9.26s to 1.62s.

Comments:

23 pages

Subjects:

Computer Vision and Pattern Recognition (cs.CV) ; Artificial Intelligence (cs.AI); Computation and Language (cs.CL)

Cite as:

arXiv:2606.02569  [cs.CV]

&nbsp;

(or

arXiv:2606.02569v1  [cs.CV]  for this version)

&nbsp;

https://doi.org/10.48550/arXiv.2606.02569

Focus to learn more

arXiv-issued DOI via DataCite (pending registration)

Submission history  From: Zhen Huang [ view email ]

[v1]

Mon, 1 Jun 2026 17:56:35 UTC (742 KB)

Full-text links:

Access Paper:

View a PDF of the paper titled AdaCodec: A Predictive Visual Code for Video MLLMs, by Haowen Hou and 10 other authors

View PDF

HTML (experimental)

TeX Source

view license

Current browse context:

cs.CV

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

cs.AI

cs.CL

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