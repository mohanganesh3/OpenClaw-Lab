[2602.23817] Footprint-Guided Exemplar-Free Continual Histopathology Report Generation

1

1  institutetext:  Faculty of Informatics and Data Science, University of Regensburg, Germany

2

2  institutetext:  Indian Institute of Information Technology Bhopal, India

@  @

Correspondence (Pratibha.Kumari@ur.de)

Footprint-Guided Exemplar-Free Continual Histopathology Report Generation

Pratibha Kumari

@  @

Daniel Reisenbüchler

Afshin Bozorgpour

yousef Sadegheih

Priyankar Choudhary

Dorit Merhof

Abstract

Rapid progress in vision-language modeling has enabled pathology report generation from gigapixel whole-slide images, but most approaches assume static training with simultaneous access to all data. In clinical deployment, however, new organs, institutions, and reporting conventions emerge over time, and sequential fine-tuning can cause catastrophic forgetting. We introduce an exemplar-free continual learning framework for WSI-to-report generation that avoids storing raw slides or patch exemplars. The core idea is a compact domain footprint built in a frozen patch-embedding space: a small codebook of representative morphology tokens together with slide-level co-occurrence summaries and lightweight patch-count priors. These footprints support generative replay by synthesizing pseudo-WSI representations that reflect domain-specific morphological mixtures, while a teacher snapshot provides pseudo-reports to supervise the updated model without retaining past data. To address shifting reporting conventions, we distill domain-specific linguistic characteristics into a compact style descriptor and use it to steer generation. At inference, the model identifies the most compatible descriptor directly from the slide signal, enabling domain-agnostic setup without requiring explicit domain identifiers.
Evaluated across multiple public continual learning benchmarks, our approach outperforms exemplar-free and limited-buffer rehearsal baselines, highlighting footprint-based generative replay as a practical solution for deployment in evolving clinical settings.

1  Introduction

Gigapixel whole slide images (WSIs) contain rich morphological information that underlies diagnostic and prognostic decisions in surgical pathology. In routine workflows, this information is summarized in free-text reports that describe salient findings and frequently follow semi-structured conventions (for example, diagnosis, grade, stage, and margin status). Automating report generation from WSIs is therefore an important step toward reducing documentation burden and enabling more consistent reporting. Recent vision-language models have begun to translate WSIs into coherent diagnostic narratives by combining slide encoders with large language models (LLMs), enabling multi-scale reasoning over billions of pixels and producing clinically meaningful text  [ 4 ,  24 ,  24 ] .

Despite this progress, most WSI report-generation systems are developed in a static training regime that assumes simultaneous access to multiple datasets spanning different organs, institutions, scanners, and patient populations. In practice, deployment is inherently continual, with new datasets (domains) arriving over time. Repeated cumulative retraining on all observed data can become computationally costly and may be infeasible when prior data cannot be retained due to storage and governance constraints  [ 21 ] . Naïve fine-tuning on incoming data can lead to catastrophic forgetting (CF), where performance degrades on previously learned domains  [ 22 ,  8 ,  10 ] . This challenge is pronounced in report generation when domain shifts affect both the input distribution and the report language, requiring CL methods that preserve input-output drift over time.

A widely used mechanism to reduce forgetting is rehearsal, where past examples are interleaved with current-domain training  [ 20 ,  17 ,  1 ] . In computational pathology, however, rehearsal is challenging because WSIs are naturally represented as variable-length sets of patch embeddings, making storage costly even when retaining only intermediate features rather than raw pixels; in addition, long-term retention of patient data may be restricted in some settings  [ 11 ] . Beyond exemplar replay, prompt-based CL for LLMs learns task-specific soft prompts while keeping the backbone frozen, but it typically assumes task or domain identity at inference to apply the appropriate prompt sequence  [ 18 ] . Relatedly, recent work on continual radiology report generation uses parameter-efficient tuning, learning a small set of domain-specific parameters while freezing the LLM backbone  [ 23 ] ; however, such strategies still maintain domain-specific components whose overhead grows as domains accumulate. In computational pathology, domain identifiers or metadata may be missing, inconsistent across sites, or insufficient to reflect shifts in reporting conventions, which can reduce robustness at deployment. Together, these considerations motivate a CL framework for WSI report generation that is storage-efficient, avoids retaining raw exemplars, and supports domain-agnostic inference.

In this work, we introduce a footprint-based generative replay framework for continual WSI report generation. Building on a HistoGPT-style vision-language generator  [ 24 ] , we address the core constraint of continual deployment: retaining prior-domain competence without keeping WSIs or large feature archives. Our key idea is to compress each domain into a compact domain footprint in a frozen embedding space, combining a representative codebook with lightweight slide-level composition statistics. These footprints enable exemplar-free rehearsal by synthesizing pseudo-WSIs, while pseudo reports are obtained from an immediate teacher to provide consistent supervision under the same next-token objective as current-domain training.
Since latent generation operates in this frozen space, it avoids generator drift across domains.
To further cope with shifts in reporting conventions, we incorporate per-domain  report-style prototype  and condition the LLM through a lightweight style-prefix. At inference, we infer the most compatible footprint directly from slide content, enabling domain-agnostic generation. We evaluate our approach under domain-incremental scenarios, measuring report quality and retention across domains.

Contributions.

We propose a CL framework for WSI report generation that
(i) represents each domain with compact visual footprints in a frozen embedding space, enabling rehearsal without storing raw WSIs or features,
(ii) enables exemplar-free replay via footprint-based pseudo-WSIs and immediate-teacher pseudo reports, with performance comparable to exemplar replay, and
(iii) supports domain-agnostic, style-conditioned generation using per-domain report-style prototypes.

Figure 1:  Overview of the proposed continual WSI report generation framework.

2  Method

We study continual WSI report generation in a domain-incremental setting. Training proceeds over domains, also termed as episodes

{

E  t

}

t  =  1

T

\{E_{t}\}_{t=1}^{T}

, where at episode

t  t

the model has access only to the current dataset

𝒟  t

=

{

(  𝐗  ,  y  )

}

\mathcal{D}_{t}=\{(\mathbf{X},y)\}

and cannot revisit samples from earlier episodes

{

𝒟  1

,  …  ,

𝒟

t  −  1

}

\{\mathcal{D}_{1},\dots,\mathcal{D}_{t-1}\}

. The goal is to learn each new domain while maintaining performance on previously learned domains. Our framework (Figure

1  ) has three components: (i) compact per-domain footprints that summarize morphology in a frozen embedding space, (ii) footprint-based replay that synthesizes pseudo-WSIs and pseudo reports to rehearse prior domains, and (iii) a per-domain report-style prototype injected into the language model to handle shifts in reporting conventions. The proposed CL strategy is model-agnostic and can be attached to any WSI-to-report generator that consumes patch embeddings