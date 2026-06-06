[2408.08982] Deep Generative Classification of Blood Cell Morphology

Deep Generative Classification of Blood Cell Morphology

Simon Deltadahl 1  &amp;Julian Gilbey 1  &amp;Christine Van Laer  2,3

&amp;Nancy Boeckx  2,4

&amp;Mathie Leers 5  &amp;Tanya Freeman 6  &amp;Laura Aiken 6  &amp;Timothy Farren 6  &amp;Matthew Smith 6  &amp;Mohamad Zeina  7

&amp;BloodCounts! consortium &amp;Concetta Piazzese  6,8

&amp;Joseph Taylor  6,8

&amp;Nicholas Gleadall 9  &amp;Carola-Bibiane Schönlieb 1  &amp;Suthesh Sivapalaratnam  6,8,†

&amp;Michael Roberts  1,10,†

&amp;Parashkev Nachev  7,†

Abstract

Accurate classification of haematological cells is critical for diagnosing blood disorders, but presents significant challenges for machine automation owing to the complexity of cell morphology, heterogeneities of biological, pathological, and imaging characteristics, and the imbalance of cell type frequencies. We introduce CytoDiffusion, a diffusion-based classifier that effectively models blood cell morphology, combining accurate classification with robust anomaly detection, resistance to distributional shifts, interpretability, data efficiency, and superhuman uncertainty quantification. Our approach outperforms state-of-the-art discriminative models in anomaly detection (AUC 0.976 vs. 0.919), resistance to domain shifts (85.85% vs. 74.38% balanced accuracy), and performance in low-data regimes (95.88% vs. 94.95% balanced accuracy). Notably, our model generates synthetic blood cell images that are nearly indistinguishable from real images, as demonstrated by a Turing test in which expert haematologists achieved only 52.3% accuracy (95% CI: [50.5%, 54.2%]). Furthermore, we enhance model explainability through the generation of directly interpretable counterfactual heatmaps. Our comprehensive evaluation framework, encompassing these multiple performance dimensions, establishes a new benchmark for medical image analysis in haematology, ultimately enabling improved diagnostic accuracy in clinical settings. Our code is available at  https://github.com/Deltadahl/CytoDiffusion .

1 Department of Applied Mathematics and Theoretical Physics, University of Cambridge, Cambridge, UK

2 Department of Laboratory Medicine, UZ Leuven, Leuven, Belgium

3 Department of Cardiovascular Sciences, Center for Molecular and Vascular Biology, University of Leuven, Leuven, Belgium

4 Department of Oncology, KU Leuven, Leuven, Belgium

5 Zuyderland Medical Center, Sittard-Geleen, Netherlands

6 Barts Health NHS Trust, London, United Kingdom

7  Queen Square Institute of Neurology, University College London, London, UK

8 Queen Mary University of London, London, United Kingdom

9 Department of Haematology, University of Cambridge, Cambridge, UK

10  Department of Medicine, University of Cambridge, Cambridge, UK

†  Equal contribution

ϵ  ∼

𝒩  ​

(  0  ,  I  )

similar-to  bold-italic-ϵ

𝒩

0  𝐼

\boldsymbol{\epsilon}\sim\mathcal{N}(0,I)

Input

𝒙  0

subscript  𝒙  0

\boldsymbol{x}_{0}

ℰ

ℰ

\mathcal{E}

𝒛

t  ∼

[  1  ,  T  ]

subscript  𝒛

similar-to  𝑡

1  𝑇

\boldsymbol{z}_{t\sim[1,T]}

Q

𝑄

Q

K

𝐾

K

V

𝑉

V

Q

𝑄

Q

K

𝐾

K

V

𝑉

V

Conditioning

c

𝑐

c

Diffusion model

arg  ​  min

𝑐

​

𝔼

ϵ  ,  t

​

[

w  t

​

‖

ϵ  −

ϵ  θ

​

(

𝒛  t

,  t  ,  c  )

⏟

‖

2  2

]

𝑐

arg  min

subscript  𝔼

bold-italic-ϵ  𝑡

delimited-[]

subscript  𝑤  𝑡

subscript

superscript

norm

bold-italic-ϵ

⏟

subscript  bold-italic-ϵ  𝜃

subscript  𝒛  𝑡

𝑡  𝑐

2

2

\underset{c}{\mathrm{arg\,min}}\,\mathbb{E}_{\boldsymbol{\epsilon},t}\left[w_{t}\left\|\boldsymbol{\epsilon}-\underbrace{\boldsymbol{\epsilon}_{\theta}(\boldsymbol{z}_{t},t,c)}\right\|^{2}_{2}\right]

ϵ  θ

subscript  bold-italic-ϵ  𝜃

\boldsymbol{\epsilon}_{\theta}

Figure 1:  Representation of the diffusion-based classification process. An input image

𝒙  0

subscript  𝒙  0

\boldsymbol{x}_{0}

is first encoded into a latent space using an encoder

ℰ

ℰ

\mathcal{E}

. Gaussian noise

ϵ  ∼

𝒩  ​

(  0  ,  𝑰  )

similar-to  bold-italic-ϵ

𝒩

0  𝑰

\boldsymbol{\epsilon}\sim\mathcal{N}(0,\boldsymbol{I})

is then added to create a noisy latent representation

𝒛  t

subscript  𝒛  𝑡

\boldsymbol{z}_{t}

. This noisy representation is fed through a diffusion model for each possible class condition

c

𝑐

c

. The model predicts the noise

ϵ  θ

subscript  bold-italic-ϵ  𝜃

\boldsymbol{\epsilon}_{\theta}

for each condition. The classification decision is made by selecting the class that minimises the error between the predicted noise

ϵ  θ

subscript  bold-italic-ϵ  𝜃

\boldsymbol{\epsilon}_{\theta}

and the true noise

ϵ

bold-italic-ϵ

\boldsymbol{\epsilon}

.

The haematological is amongst the most complex of physiological systems, and uniquely intertwined with all others. Though often quantified by simple “blood counts” of cell class frequencies, its characteristics are both supremely rich and highly variable within and across individuals  [ 1 ] . A cardinal aspect is the morphology of individual blood cells as seen on light microscopy of blood smears  [ 2 ] . The richness of morphological appearances—and their complex modulation by diverse biological, pathological, and instrumental factors—demand human expert visual description in commensurately expressive terms. The allocation of a cell to its major morphological type, e.g. lymphocyte, is only the crudest form of description, on which finer fractionation into subtypes, across a wide spectrum of (ab)normality, is overlaid.

Indeed, the task of morphological characterisation is both open-ended and lacks a definitive ground truth: there may be morphological patterns whose subtlety has concealed great clinical significance, and some morphological classes are purely expert-determined visual phenotypes with no means of objective corroboration. Moreover, pathological appearances may be highly unusual or unique, precluding classification into any class, even at the simplest level of description, and requiring metacognitive awareness of its impossibility. The difficulty is commonly compounded by interactions with irrelevant biological features with variable representation across the population, and instrumental variations of technical origin  [ 3 ,  4 ] . The challenge, in short, is one human experts can only imperfectly meet, inevitably exhibiting marked variation with skill and experience  [ 5 ,  6 ] .

So framed, the task of automating blood morphological analysis has a different aim from that commonly assumed, needs broader validation, and is harder to accomplish. The primary aim is not to approximate a human expert in a more cost-effective, reproducible, and scalable alternative, but to capture the space of possible morphological appearances with potentially superhuman fidelity, flexibility, and metacognitive awareness. It cannot plausibly be achieved by a discriminative model trained to classify cells into standard morphological classes, for such an approach can only approximate human fidelity, and is ill-equipped to deal with the domain shifts, class imbalances and anomalies, complex interactions between biological, pathological, and instrumental features, and metacognitive demands, that the foregoing implies. Nor can success in the task be evaluated by standard measures of in-distribution performance, for they are remote from real-world experience.

Here we therefore introduce a deep generative approach founded on diffusion-based classifiers, and a comprehensive evaluative framework that aspires to real-world ecological validity. Our approach, which we call CytoDiffusion, addresses the key challenges, exemplified by blood morphological analysis but not unique to it, of machine vision models of medical data, namely:

1.

Domain Shift : Models trained on data from a specific biological, pathological, and instrumental context often struggle when applied to data with a different distribution of characteristics, limiting their practical u