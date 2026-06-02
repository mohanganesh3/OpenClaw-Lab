[2606.02522] Moment-Video: Diagnosing Temporal Fidelity of Video MLLMs on Momentary Visual Events

Skip to main content

Learn about arXiv becoming an independent nonprofit.

We gratefully acknowledge support from the Simons Foundation,  member institutions , and all contributors.

Donate

&gt;

cs

&gt;  arXiv:2606.02522

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

arXiv:2606.02522  (cs)

[Submitted on 1 Jun 2026]

Title: Moment-Video: Diagnosing Temporal Fidelity of Video MLLMs on Momentary Visual Events

Authors:  Xiaolin Liu ,  Yilun Zhu ,  Xiangyu Zhao ,  Xuehui Wang ,  Yan Li ,  Xin Li ,  Haoyu Cao ,  Xing Sun ,  Shaofeng Zhang ,  Xu Yang ,  Zhihang Zhong ,  Xue Yang

View a PDF of the paper titled Moment-Video: Diagnosing Temporal Fidelity of Video MLLMs on Momentary Visual Events, by Xiaolin Liu and 11 other authors

View PDF

HTML (experimental)

Abstract: Video multimodal large language models (MLLMs) have made rapid progress on general and long-form video understanding, yet their ability to preserve brief answer-critical visual evidence remains underexplored. Many practical questions are determined by momentary visual events: localized actions or state transitions that may last only a few frames. Such evidence can be skipped by sparse frame sampling, suppressed by visual-token compression, or diluted by coarse temporal aggregation, causing failures that language-side reasoning cannot reliably recover. We introduce Moment-Video, a benchmark for diagnosing the temporal fidelity of video MLLMs through momentary visual event understanding. Each question is grounded in a localized, visually observable, and sampling-sensitive event, requiring models to notice, count, describe, or reason about transient evidence rather than rely on persistent objects, global scene context, or language priors. Moment-Video contains 1,000 human-verified video-QA pairs across 7 domains and 25 fine-grained subcategories, covering four task types: Temporal Occurrence, Temporal Counting, Action Description, and Temporal Reasoning. We evaluate 33 proprietary and open-source MLLMs on Moment-Video. The best-performing model, Seed-2.0-Pro, achieves only 39.6% overall accuracy, while most open-source models remain below 25%, revealing a substantial gap in momentary visual event understanding. Diagnostic analyses show that denser frame sampling improves some models but does not eliminate the bottleneck, and longer videos introduce stronger temporal-localization challenges. These findings suggest that current video MLLMs still lack temporally faithful representations for capturing, preserving, and using brief but decisive visual evidence.

Comments:

28 pages, 10 figures, 11 tables

Subjects:

Computer Vision and Pattern Recognition (cs.CV) ; Artificial Intelligence (cs.AI)

Cite as:

arXiv:2606.02522  [cs.CV]

&nbsp;

(or

arXiv:2606.02522v1  [cs.CV]  for this version)

&nbsp;

https://doi.org/10.48550/arXiv.2606.02522

Focus to learn more

arXiv-issued DOI via DataCite (pending registration)

Submission history  From: Yilun Zhu [ view email ]

[v1]

Mon, 1 Jun 2026 17:32:20 UTC (8,465 KB)

Full-text links:

Access Paper:

View a PDF of the paper titled Moment-Video: Diagnosing Temporal Fidelity of Video MLLMs on Momentary Visual Events, by Xiaolin Liu and 11 other authors

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