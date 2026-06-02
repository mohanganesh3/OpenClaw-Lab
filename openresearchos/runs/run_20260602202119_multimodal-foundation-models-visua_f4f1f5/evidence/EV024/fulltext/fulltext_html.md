[2605.29881] Mitigating Hallucination in Vision-Language Models through Barrier-Regulated Adaptive Closed-form Steering

Skip to main content

Learn about arXiv becoming an independent nonprofit.

We gratefully acknowledge support from the Simons Foundation,  member institutions , and all contributors.

Donate

&gt;

cs

&gt;  arXiv:2605.29881

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

arXiv:2605.29881  (cs)

[Submitted on 28 May 2026]

Title: Mitigating Hallucination in Vision-Language Models through Barrier-Regulated Adaptive Closed-form Steering

Authors:  Soumyadeep Jana ,  Pulkit Mittal ,  Sanasam Ranbir Singh

View a PDF of the paper titled Mitigating Hallucination in Vision-Language Models through Barrier-Regulated Adaptive Closed-form Steering, by Soumyadeep Jana and 2 other authors

View PDF

HTML (experimental)

Abstract: Large vision-language models (LVLMs) often hallucinate objects that are not present in the input image, largely because visual grounding weakens as decoding progresses. Existing inference-time mitigation methods modify logits or hidden states throughout generation, but they suffer from three key limitations: they lack an explicit grounding objective, intervene even when the model is already well-grounded, and use fixed correction strengths that do not adapt to the severity of grounding failure. We propose BRACS (Barrier-Regulated Adaptive Closed-form Steering), a training-free steering framework that addresses these issues through barrier-regulated adaptive closed-form steering. BRACS monitors the model&#39;s own attention to measure visual grounding and applies corrections to the hidden states only when grounding deteriorates. The corrective update is computed analytically in closed form, requiring no training of auxiliary networks or model retraining. Experiments on LLaVA-1.5-7B and Qwen-VL-Chat show that BRACS consistently outperforms prior methods on hallucination benchmarks, reducing CHAIR$_s$ by 9.4 points and improving POPE F1 by 2.7 points, while matching or improving performance on four general multimodal benchmarks. BRACS also remains efficient, operating at 80% of greedy decoding throughput and achieving 1.3 times higher speed on average than the baselines.

Subjects:

Computer Vision and Pattern Recognition (cs.CV) ; Artificial Intelligence (cs.AI)

Cite as:

arXiv:2605.29881  [cs.CV]

&nbsp;

(or

arXiv:2605.29881v1  [cs.CV]  for this version)

&nbsp;

https://doi.org/10.48550/arXiv.2605.29881

Focus to learn more

arXiv-issued DOI via DataCite (pending registration)

Submission history  From: Soumyadeep Jana [ view email ]

[v1]

Thu, 28 May 2026 13:07:01 UTC (706 KB)

Full-text links:

Access Paper:

View a PDF of the paper titled Mitigating Hallucination in Vision-Language Models through Barrier-Regulated Adaptive Closed-form Steering, by Soumyadeep Jana and 2 other authors

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