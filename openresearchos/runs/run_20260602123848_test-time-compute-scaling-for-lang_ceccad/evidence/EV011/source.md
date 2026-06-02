# EV011: GemDetox at TextDetox CLEF 2025: Enhancing a Massively Multilingual Model for Text Detoxification on Low-resource Languages

URL: https://www.semanticscholar.org/paper/00475b7f2906b127df83d45587c82e498be4c3a0
Year: 2025
Source: semantic_scholar
Arxiv: 2510.01250

## Abstract

As social-media platforms emerge and evolve faster than the regulations meant to oversee them, automated detoxification might serve as a timely tool for moderators to enforce safe discourse at scale. We here describe our submission to the PAN 2025 Multilingual Text Detoxification Challenge, which rewrites toxic single-sentence inputs into neutral paraphrases across 15 typologically diverse languages. Building on a 12B-parameter Gemma-3 multilingual transformer, we apply parameter-efficient LoRA SFT fine-tuning and prompting techniques like few-shot and Chain-of-Thought. Our multilingual training corpus combines 3,600 human-authored parallel pairs, 21,600 machine-translated synthetic pairs, and model-generated pairs filtered by Jaccard thresholds. At inference, inputs are enriched with three LaBSE-retrieved neighbors and explicit toxic-span annotations. Evaluated via Style Transfer Accuracy, LaBSE-based semantic preservation, and xCOMET fluency, our system ranks first on high-resource and low-resource languages. Ablations show +0.081 joint score increase from few-shot examples and +0.088 from basic CoT prompting. ANOVA analysis identifies language resource status as the strongest predictor of performance ($\eta^2$ = 0.667, p<0.01).
