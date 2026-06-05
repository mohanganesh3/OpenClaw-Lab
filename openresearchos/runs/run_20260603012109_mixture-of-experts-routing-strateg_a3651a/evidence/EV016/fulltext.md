[2603.14147] An Alternative Trajectory for Generative AI

An Alternative Trajectory for Generative AI

Margarita Belova

Princeton University

margarita.bel@princeton.edu

&amp;Yuval Kansal  1

1  footnotemark:

1

Princeton University

yuvalkansal@princeton.edu

&amp;Yihao Liang  1

1  footnotemark:

1

Princeton University

yhliang@princeton.edu

&amp;Jiaxin Xiao  1

1  footnotemark:

1

Princeton University

jx0800@princeton.edu

&amp;Niraj K. Jha

Princeton University

jha@princeton.edu

Equal contributions: These author names are listed in alphabetical order.

Abstract

The generative artificial intelligence (AI) ecosystem is undergoing rapid transformations that threaten its sustainability. As models transition from research prototypes to high-traffic products, the dominant energetic burden has shifted from one-time training to recurring, unbounded inference—a phenomenon exacerbated by the emergence of “reasoning models” that inflate compute costs by orders of magnitude per query. The prevailing trajectory, i.e, pursuit of artificial general intelligence through continued scaling of monolithic models, is colliding with hard physical constraints: localized grid failures, prohibitive water consumption, and diminishing returns on data scaling. Whereas this trajectory yields models with impressive factual recall, it struggles in domains that require in-depth verifiable reasoning. A possible cause is a lack of sufficient abstractions in the underlying training data.

Current large language models (LLMs) exhibit genuine reasoning depth only in domains like mathematics and coding, where rigorous, pre-existing abstractions provide structural grounding. In other fields that lack such abstractions, the current approach fails to generalize well. We propose an alternative trajectory based on  domain-specific superintelligence (DSS) . We argue that to achieve robust reasoning in open-world domains, it is advantageous to first construct explicit symbolic abstractions, such as knowledge graphs, ontologies, and formal logic. These abstractions can form the basis for generation of synthetic curricula, training on which can enable small language models to master complex domain-specific reasoning, without encountering the model collapse problem that is typical of current synthetic data methods that use an LLM to supplement training data for next-generation LLMs.

Rather than training a single generalist giant model, we envision “societies of DSS models”: dynamic ecosystems where orchestration agents route tasks to distinct DSS back-ends. This paradigm shift can help decouple capability from size, enabling intelligence to migrate from energy-intensive data centers to secure, on-device experts. By aligning algorithmic progress with physical constraints, DSS societies can help move generative AI away from an environmental liability to a sustainable force for economic empowerment.

1  Introduction

Currently, the dominant paradigm in the generative artificial intelligence (AI) industry is a “top-down” approach: pursuit of artificial general intelligence (AGI) through massive scaling of generalist large language models (LLMs). The prevailing belief is that by training ever-larger models on the entirety of Internet-scale data, we will eventually converge towards an AI system capable of human-level reasoning across all domains. This vision relies heavily on scaling laws, which suggest that increasing model size and data automatically yield proportional gains in intelligence  (Kaplan et al.,  2020 ) . However, this “bigger-is-better” trajectory incurs ever-increasing costs. Empirical scaling laws document predictable gains from larger models and datasets but do not instill deeper forms of understanding or reliable reasoning on complex, compositional tasks in LLMs. In addition, the current trajectory faces various headwinds in terms of training and inference energy costs, data quality, and, most critically, depth of reasoning.

More concretely, three interrelated problems make the current top-down paradigm that the AI industry is following a risky foundation to build future AI systems.

First, both training and deploying large foundation models demand vast compute and energy resources. Modern deep learning (DL) workflows entail substantial financial and environmental costs, which scale quickly as models grow in size. The need for massive computational resources restricts AI development to a few well-funded entities. These costs create practical, economic, and sustainability constraints  (Strubell et al.,  2019 ) .

Second, whereas large models often excel at surface-level pattern matching and retrieval, they still struggle with robust, compositional reasoning and practical problem solving. Techniques like chain-of-thought (CoT) prompting have been shown to unlock stronger reasoning capabilities in language models, but largely rely on scale and special prompting rather than on principled, abstraction-driven reasoning mechanisms. Recent studies show suboptimal performance of LLMs on tasks that require combining small components into larger, novel solutions  (Dedhia et al.,  2025 ; Kansal and Jha,  2026 ; Zhao and Zhang,  2024 ) . These studies suggest that mere scaling is not likely to produce the kinds of structured reasoning needed for many real-world domains  (Wei et al.,  2022 ) .

Third, data quality matters. The Internet-scale text corpora used to train generalist models are large but highly unstructured and heterogeneous in quality and coverage. Good reasoning and trustworthy behavior depend on high-quality well-curated training data and on targeted curricula that teach models how to use abstractions. The trajectory of intelligent reasoning, one can argue, is predicated on a specific sequence: abstraction followed by generalization. In this view, the first step is to form structured mental models (abstractions) of the world, understand rules, relationships, and causalities, and then generalize these models to new, unseen situations. This data-centric view of AI argues that improving data quality, curation, and structure often yields larger gains than continually increasing model size alone  (Jakubik et al.,  2024 ; Halevy et al.,  2009 ) . Current LLM training approaches either invert or bypass this step altogether. They attempt to achieve generalization purely through the statistical breadth of knowledge acquisition. This results in the well-documented brittleness of LLMs  (Haller et al.,  2025 ; Su et al.,  2025 ; Mohsin et al.,  2025 ) . To move forward, it may be advantageous to restore abstraction as the prerequisite for generalization.

We propose an alternative trajectory for generative AI: a “bottom-up” approach centered on domain-specific superintelligence (DSS). By superintelligence, we do mean super-human, but qualified by its adjective: domain-specific. The AI industry defines superintelligence as being post-AGI since it is pursuing generalist superintelligence across all domains, rather than in specific domains. However, omniscience in a single LLM monolith may be an overrated attribute. We base our proposed trajectory on DSS because it is practically achievable and also because DSS societies can enable significant success in applications that require inter-domain expertise. Thus, rather than relying on a single generalist LLM monolith, we envision a future built on specialized small language models (SLMs): smaller, focused models trained on high-quality domain data, paired with explicit abstractions [e.g., knowledge graphs (KGs), formal semantics, inductive program libraries] and modular reasoning engines. This vision of a family of specialist models is highly scalable and mirrors how human societies are organized: Individuals acquire expertise in narrow domains and cooperate to solve complex problems that require inter-domain expertise. This idea has deep roots in cognitive science: Marvin Minsky’s Society of Mind framed intelligence as the coordin