[2601.11108] Simple Models, Rich Representations: Visual Decoding from Primate Intracortical Neural Signals

Simple Models, Rich Representations: Visual Decoding from Primate Intracortical Neural Signals

Matteo Ciferri

University of Rome, Tor Vergata
 Department of Biomedicine and Prevention

matteo.ciferri@students.uniroma2.eu

&amp;Matteo Ferrante *

University of Rome, Tor Vergata
 Department of Biomedicine and Prevention

matteo.ferrante@uniroma2.it

&amp;Nicola Toschi

University of Rome, Tor Vergata
 Department of Biomedicine and Prevention
 A.A. Martinos Center for Biomedical Imaging

Harvard Medical School/MGH, Boston (US)

These authors contributed equally to this work

Abstract

Understanding how neural activity gives rise to perception is a central challenge in neuroscience. We address the problem of decoding visual information from high-density intracortical recordings in primates, using the THINGS Ventral Stream Spiking Dataset. We systematically evaluate the effects of model architecture, training objectives, and data scaling on decoding performance. Results show that decoding accuracy is mainly driven by modeling temporal dynamics in neural signals, rather than architectural complexity. A simple model combining temporal attention with a shallow MLP achieves up to 70% top-1 image retrieval accuracy, outperforming linear baselines as well as recurrent and convolutional approaches. Scaling analyses reveal predictable diminishing returns with increasing input dimensionality and dataset size. Building on these findings, we design a modular generative decoding pipeline that combines low-resolution latent reconstruction with semantically conditioned diffusion, generating plausible images from 200 ms of brain activity. This framework provides principles for brain-computer interfaces and semantic neural decoding.

1  Introduction

A complete account of perception and behavior must bridge neural representations with mental states, linking spikes and field potentials to the contents of subjective experience and overt action. Recent progress in cognitive science and computational neuroscience has been catalyzed by three intertwined developments. First, community-driven efforts now release large, meticulously curated datasets that pair rich sensory stimulation with high-resolution neural recordings  (Allen  et al. ,  2022 ; Horikawa and Kamitani,  2017 ; Chang  et al. ,  2019 ; Hebart  et al. ,  2023 ) . Second, advances in machine learning—particularly deep generative modeling and scalable optimization—provide expressive function classes capable of capturing the complex structure of brain-world mappings  (Antonello  et al. ,  2023 ; Oota  et al. ,  2023 ; Banville  et al. ,  2025 ) . Third, experimental practice is changing from  wide  surveys of many individuals to  deep , longitudinal studies that expose a few subjects to tens of thousands of stimuli, drastically increasing statistical power  (Kupers  et al. ,  2024 ) .

These factors have revived bidirectional modeling of the stimulus–brain relationship.  Encoding  models predict neural responses from sensory features, helping us understand the functional organization of the cortex, whereas  decoding  models seek to reconstruct stimuli - or latent variables relevant to the task - from brain activity, a line of work central to basic science, as well as emerging brain–computer interfaces. Successes span multiple modalities (EEG, MEG, fMRI, ECoG, and Utah array recordings) and cognitive domains, including language comprehension, speech production, music, and vision  (Bazeille  et al. ,  2021 ; Oota  et al. ,  2023 ) . However, even with invasive data, key questions persist: What properties of intracortical spike trains carry the information necessary for high-fidelity decoding? How do architectural choices—linear versus nonlinear models, temporal aggregation windows, loss functions—shape performance limits? And how do these factors interact with scale, both in terms of training data and in terms of the dimensionality of neural input?

We address these questions through the lens of visual decoding. Our study leverages the recently released THINGS Ventral-Stream Spiking Dataset (TVSD)  (Papale  et al. ,  2025 ) , in which two macaques viewed

∼

25  ​  k

\sim 25\,{\rm k}

natural images drawn from 1,854 object categories while they recorded multi-unit activity (MUA) from

∼

2  ,  000

\sim\!2{,}000

channels distributed across V1, V4 and IT at 30 kHz. Each image in the training partition was shown once, while 100 held-out images were repeated 30 times to boost signal-to-noise ratio and enable stringent cross-validation. In this work, we tackle the following research questions:

1.

Temporal versus architectural complexity.  What role does temporal structure play in neural decoding, and what types of models are best suited to capture it? While our experiments do not quantify the exact contribution of millisecond-scale timing, they suggest that modeling temporal dynamics is crucial for high-level decoding tasks. To explore this, we trained various models to decode neural data into semantic image representations obtained using a frozen CLIP model. We systematically compared architectures ranging from simple linear models to recurrent neural networks capable of capturing complex nonlinear dynamics. We show that the key driver of semantic decoding performance is the capacity to model the temporal structure in neural responses. Nonlinear models do improve performance, but our experiments reveal that their benefits are largely attributable to better temporal aggregation rather than to increased architectural complexity or spatial modeling. The evaluation was carried out by measuring the accuracy of image retrieval on held out test data, specifically quantifying the model’s ability to identify exact images from neural activity using top-1 and top-5 retrieval metrics.

2.

Objective functions.  We compare two ways of predicting vector representations from brain activity: mean squared error loss and representation alignment with contrastive learning. Similarly to the previous point, we used retrieval performance as a probe of the quality of decoded embeddings.

3.

Scaling laws.  We chart the performance of our best model in the retrieval task as a function of (i) the number of trials and (ii) the number of principal components derived from neural channels, revealing predictable regimes of diminishing returns that inform experimental design.

4.

Generative Decoding  We introduce a two-stage decoder that samples candidate images from a frozen generative prior (Stable Diffusion  (Podell  et al. ,  2023 ) ) and performs rejection sampling guided by a learned neural likelihood, achieving near-photorealistic reconstructions from

≈

200  ​  ms

\approx 200\,\mathrm{ms}

windows of activity. See Figure

1

for an overview of our generative decoding pipeline.

To address the first two research questions, we used zero-shot image retrieval as a proxy for assessing the quality of the mapping between brain activity and semantic visual representations. Crucially, we avoid generative models at this stage to minimize the confounding influence of strong image priors. When a generative model produces a high-quality image, it becomes difficult to discern whether the result reflects successful decoding or simply the model’s inherent ability to produce photorealistic samples  (Shirakawa  et al. ,  2024 ) . Retrieval-based evaluation offers a more transparent and interpretable benchmark: the model must identify the correct image from a fixed candidate pool based solely on the neural signal, allowing precise measurement of top-1 and top-5 accuracy.

However, retrieval has its limitations: mainly, its reliance on a predefined candidate set, which restricts generalization. For this reason, once we validated that our model achieved strong performance in this constrained setting, we turned to the more ambitious goal of gener