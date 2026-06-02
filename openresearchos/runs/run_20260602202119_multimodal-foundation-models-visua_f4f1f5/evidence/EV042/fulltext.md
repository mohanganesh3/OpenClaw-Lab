[2605.02912] Reasoning-Guided Grounding: Elevating Video Anomaly Detection through Multimodal Large Language Models

Skip to main content

Learn about arXiv becoming an independent nonprofit.

We gratefully acknowledge support from the Simons Foundation,  member institutions , and all contributors.

Donate

&gt;

cs

&gt;  arXiv:2605.02912

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

arXiv:2605.02912  (cs)

[Submitted on 7 Apr 2026]

Title: Reasoning-Guided Grounding: Elevating Video Anomaly Detection through Multimodal Large Language Models

Authors:  Sakshi Agarwal ,  Aishik Konwer ,  Ankit Parag Shah

View a PDF of the paper titled Reasoning-Guided Grounding: Elevating Video Anomaly Detection through Multimodal Large Language Models, by Sakshi Agarwal and 2 other authors

View PDF

HTML (experimental)

Abstract: Video Anomaly Detection (VAD) has traditionally been framed as binary classification or outlier detection, providing neither interpretable reasoning nor precise spatial localization of anomalous events. While Vision-Language Models (VLMs) offer rich scene understanding, they struggle with reliable spatial grounding - often producing hallucinated or geometrically invalid bounding boxes when asked to localize objects. We propose VANGUARD (Video Anomaly Understanding through Reasoning and Grounding), a framework that unifies anomaly classification, spatial grounding, and chain-of-thought reasoning within a single VLM. VANGUARD introduces a three-stage curriculum that progressively layers training objectives: (1) classifier warmup on frozen backbone features, (2) LoRA-adapted spatial grounding, and (3) chain-of-thought generation. To overcome the sparse annotation typical of VAD benchmarks, we employ a teacher-student annotation pipeline in which a VLM (Qwen3-VL-4B) generates structured per-subclip reasoning trajectories based on manual annotations available from the UCA Dataset. Further, GroundingDINO provides bounding box supervision. On UCF-Crime, VANGUARD achieves 94% ROC-AUC with 84% F1 while simultaneously producing interpretable chain-of-thought explanations and spatial grounding of anomalous objects - capabilities absent from prior VAD methods. Ablations confirm that staged training outperforms monolithic optimization, and that structured reasoning acts as an implicit regularizer yielding more balanced predictions than classification-only fine-tuning. Zero-shot transfer to XD-Violence and ShanghaiTech demonstrates cross-domain generalization without target-domain adaptation.

Comments:

under review at conference

Subjects:

Computer Vision and Pattern Recognition (cs.CV) ; Artificial Intelligence (cs.AI); Machine Learning (cs.LG)

Cite as:

arXiv:2605.02912  [cs.CV]

&nbsp;

(or

arXiv:2605.02912v1  [cs.CV]  for this version)

&nbsp;

https://doi.org/10.48550/arXiv.2605.02912

Focus to learn more

arXiv-issued DOI via DataCite

Submission history  From: Aishik Konwer [ view email ]

[v1]

Tue, 7 Apr 2026 20:15:15 UTC (6,482 KB)

Full-text links:

Access Paper:

View a PDF of the paper titled Reasoning-Guided Grounding: Elevating Video Anomaly Detection through Multimodal Large Language Models, by Sakshi Agarwal and 2 other authors

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