[2510.01250] Notebook for the PAN Lab at CLEF 2025

\copyrightclause

Copyright for this paper by its authors.
Use permitted under Creative Commons License Attribution 4.0
International (CC BY 4.0).

\conference

CLEF 2025 Working Notes, 9 – 12 September 2025, Madrid, Spain

[email=cls364@alumni.ku.dk,
url=https://github.com/ducanhdt/ducanhhbtt-detoxification2025/,
]

[email=plh618@alumni.ku.dk,
]
 \cormark [1]

\cortext

[1]Corresponding author.

Notebook for the PAN Lab at CLEF 2025

Trung Duc Anh Dang

Ferdinando Pio D’Elia

Centre for Language Technology, University of Copenhagen, Denmark

(2025)

Abstract

As social-media platforms emerge and evolve faster than the regulations meant to oversee them, automated detoxification might serve as a timely tool for moderators to enforce safe discourse at scale. We here describe our submission to the PAN 2025 Multilingual Text Detoxification Challenge, which rewrites toxic single-sentence inputs into neutral paraphrases across 15 typologically diverse languages. Building on a 12B-parameter Gemma-3 multilingual transformer, we apply parameter-efficient LoRA SFT fine-tuning and prompting techniques like few-shot and Chain-of-Thought. Our multilingual training corpus combines 3 600 human-authored parallel pairs, 21 600 machine-translated synthetic pairs, and model-generated pairs filtered by Jaccard thresholds. At inference, inputs are enriched with three LaBSE-retrieved neighbors and explicit toxic-span annotations. Evaluated via Style Transfer Accuracy, LaBSE-based semantic preservation, and xCOMET fluency, our system ranks first on high-resource and low-resource languages. Ablations show +0.081 joint score increase from few-shot examples and +0.088 from basic CoT prompting. ANOVA analysis identifies language resource status as the strongest predictor of performance (

η  2

\eta^{2}

= 0.667, p &lt; 0.01).

Warning: This paper contains offensive and potentially triggering texts that only serve as illustrative examples.

keywords:

Multilingual text detoxification  \sep Large Language Models  \sep Parameter‐efficient fine-tuning  \sep Chain-of-Thought  \sep Data augmentation

1  Introduction

The widespread use of digital communication and social media platforms has prompted the need, rather urgent, for moderation and content detoxification strategies that can be proved effective and easy to implement. Toxic language, hate speech, and harassment jeopardize safety and pluralism of online spaces, motivating the research community to fortify automated methods that could intervene at scale. The PAN 2025 Multilingual Detoxification Challenge  [ 1 ]  [ 2 ]  offers a shared testbed for systems that aim to rephrase user posts into safer language while keeping their original meaning, and across multiple languages.

Large‑scale detoxification is not a purely technical exercise; it carries substantial social, political, and practical ramifications. In real-world scenarios, excessive detoxification interventions may be perceived as censorship, eroding trust and discouraging open dialogue; interventions may strip away the original message of its core purpose. The stakes are even higher for minorities and marginalized communities, who nowadays often rely on these platforms to voice political and social dissent and might consequently feel censored: as far as dissenting voices are concerned, detoxification could silence the very people it aims to protect. While we set these broader implications aside for the present study, they remain a key source of motivation for our work.

The Multilingual Text Detoxification task at PAN 2025 challenges participants to rewrite a toxic piece of text into a non‑toxic form while preserving as much of the original meaning as possible across 15 typologically diverse languages, ranging from English and Spanish to Tatar and Hinglish. Formally, the input is a single‑sentence text that contains at least one instance of toxic language, and the system must produce a semantically faithful paraphrase with neutral tone.
Notably, the competition constrains the notion of toxicity to explicit toxicity: obscene or offensive lexicon in which meaningful neutral content is still present. Implicit toxicity, such as sarcasm, coded hate speech, or passive‑aggressive formulations, is excluded  1

1  1 It warrants mention that such subtle toxicity is just as dangerous as a social phenomenon, and ample research has shown how ordinary terms and euphemisms can smuggle dehumanizing ideas into everyday speech, gradually normalizing hatred and exclusion.  [ 3 ]  Yet, any problem is such only when we can carve it into a series of smaller and more manageable steps; this is our starting point.

.
We here present our approach, which builds directly on the 2024 paradigm [ 4 ]  of fine-tuning large-scale multilingual pretrained transformer models for detoxification. We opted for a model with extensive multilingual pre-training yet lightweight features, and focused on maximizing the possibilities of Chain-of-Thought prompting as well as data augmentation.

The paper is organized as follows: in Section 2 we review related work; Section 3 describes our methodology; Section 4 details the experimental setup; Section 5 presents our automatic and qualitative results; and Section 6 discusses limitations and future directions. Our experiments show that, while our approach achieves strong performance across all languages, the gains from data augmentation are especially pronounced for the low-resource languages for which they were designed, narrowing their gap with high-resource counterparts.

2  Related work

Text detoxification, a specialized subfield of Text Style Transfer (TST), involves transforming toxic texts into neutral versions while maintaining semantic integrity and linguistic fluency. Initial research in detoxification was largely driven by prominent competitions such as the Jigsaw/Conversation AI Kaggle challenges (2018–2021)  [ 5 ] , providing substantial datasets and significantly advancing toxicity detection methods. Early models, such as  [ 6 ] , employed unsupervised encoder-decoder architectures with cycle consistency losses for style transfer. Subsequent approaches, such as CondBERT and ParaGedi  [ 7 ] , introduced unsupervised conditional masked language modeling and paraphrasing, setting benchmarks at the time. Further progress was marked by the ParaDetox corpus  [ 8 ] , which demonstrated that parallel data substantially improved the detoxification performance over purely unsupervised techniques.

Recently, transformer-based large language models (LLMs) have been largely utilized in detoxification research due to their powerful text generation capabilities. Notably, GPT-based models, including GPT-2, GPT-3, and GPT-4, demonstrated considerable efficacy when fine-tuned or used with few-shot prompting strategies  [ 9 ] . Models such as GPT-DETOX have explored innovative in-context learning techniques including zero-shot, few-shot, and ensemble prompting, significantly outperforming earlier supervised and unsupervised methods.

Parallel corpus availability remains critical for enhancing detoxification methods. The creation of new datasets such as the multilingual MultiParaDetox corpus  [ 10 ]  has significantly expanded detoxification research to include languages previously underrepresented, such as Hindi, Arabic, Chinese, and Amharic. Furthermore, sophisticated prompt-engineering frameworks, like CO-STAR  [ 11 ] , have shown promising results by strategically guiding LLMs to enhance contextual and semantic coherence during detoxification. Recent advancements have also emphasized explainability and interpretability in detoxification processes. Very recently, a first automated explainable analysis across multiple languages was published  [ 12 ] , revealing common patterns and language-specific toxicity traits, and implemented Chain-of-Thought reasoning techniques to enhance the detoxification accuracy of LLMs