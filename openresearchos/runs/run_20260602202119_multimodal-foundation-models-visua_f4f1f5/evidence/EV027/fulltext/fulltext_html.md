[2605.14621] Do We Really Need External Tools to Mitigate Hallucinations? SIRA: Shared-Prefix Internal Reconstruction of Attribution

Skip to main content

Learn about arXiv becoming an independent nonprofit.

We gratefully acknowledge support from the Simons Foundation,  member institutions , and all contributors.

Donate

&gt;

cs

&gt;  arXiv:2605.14621

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

arXiv:2605.14621  (cs)

[Submitted on 14 May 2026]

Title: Do We Really Need External Tools to Mitigate Hallucinations? SIRA: Shared-Prefix Internal Reconstruction of Attribution

Authors:  Tian Qin ,  Junzhe Chen ,  Yuqing Shi ,  Tianshu Zhang ,  Qiang Ju ,  Lijie Wen

View a PDF of the paper titled Do We Really Need External Tools to Mitigate Hallucinations? SIRA: Shared-Prefix Internal Reconstruction of Attribution, by Tian Qin and 5 other authors

View PDF

HTML (experimental)

Abstract: Large vision-language models (LVLMs) often hallucinate when language priors dominate weak or ambiguous visual evidence. Existing contrastive decoding methods mitigate this problem by comparing predictions from the original image with those from externally perturbed visual inputs, but such references can introduce off-manifold artifacts and require costly extra forward passes. We propose SIRA, a training-free internal contrastive decoding framework that constructs a counterfactual reference inside the same LVLM by exploiting the staged information flow of multimodal transformers. Instead of removing visual information from the input, SIRA first lets image and text tokens interact through a shared prefix, forming an aligned multimodal state that preserves prompt interpretation, decoding history, positional structure, and early visual grounding. It then forks a counterfactual branch in later transformer layers, where attention to image-token positions is masked. This branch retains the shared multimodal context but lacks continued access to fine-grained visual evidence, yielding a language-prior-dominated internal reference for token-level contrast. During decoding, SIRA suppresses tokens that remain strong without late visual access and favors predictions whose advantage depends on the full visual pathway. Experiments on POPE, CHAIR, and AMBER with Qwen2.5-VL and LLaVA-v1.5 show that SIRA consistently reduces hallucinations while preserving descriptive coverage and incurring lower overhead than two-pass contrastive decoding. SIRA requires no training, external verifier, or perturbed input, and applies to open-weight LVLMs with white-box inference access.

Subjects:

Computer Vision and Pattern Recognition (cs.CV) ; Artificial Intelligence (cs.AI); Computation and Language (cs.CL)

Cite as:

arXiv:2605.14621  [cs.CV]

&nbsp;

(or

arXiv:2605.14621v1  [cs.CV]  for this version)

&nbsp;

https://doi.org/10.48550/arXiv.2605.14621

Focus to learn more

arXiv-issued DOI via DataCite

Submission history  From: Tian Qin [ view email ]

[v1]

Thu, 14 May 2026 09:37:55 UTC (1,804 KB)

Full-text links:

Access Paper:

View a PDF of the paper titled Do We Really Need External Tools to Mitigate Hallucinations? SIRA: Shared-Prefix Internal Reconstruction of Attribution, by Tian Qin and 5 other authors

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