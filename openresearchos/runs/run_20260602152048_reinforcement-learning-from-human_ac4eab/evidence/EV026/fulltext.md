[2510.14616] Beyond Correctness: Evaluating Subjective Writing Preferences Across Cultures

\contribution

Full author list in Contributions

Beyond Correctness: Evaluating Subjective Writing Preferences Across Cultures

( November 5, 2025 )

Abstract

Current preference learning methods achieve high accuracy on standard benchmarks but
exhibit significant performance degradation when objective quality signals are removed.
We introduce  WritingPreferenceBench , a dataset of 1,800 human-annotated
preference pairs (1,200 English, 600 Chinese) across 8 creative writing genres, where
responses are matched for objective correctness, factual accuracy, and length. On this
benchmark, sequence-based reward models—the standard architecture for RLHF—achieve only
52.7% mean accuracy, while zero-shot language model judges perform at 53.9%.
In contrast, generative reward models that produce explicit reasoning chains achieve 81.8% accuracy. We observe high within-model variance across genres: individual models range from 18.2% to 81.8% accuracy across different writing categories, with standard deviations averaging 10.1%.
This variance persists regardless of model scale, with 27B parameter models showing no consistent improvement over 8B variants. Our results suggest that current RLHF methods primarily learn to detect objective errors rather than capture subjective quality preferences (e.g., creativity, stylistic flair, and emotional resonance), and that successful preference modeling may require intermediate reasoning representations rather than direct classification.

\correspondence

Ge Zhang at

\checkdata [Project Page] https://WritingPreferenceBench.github.io/

1  Introduction

Reinforcement learning from human feedback (RLHF) has become the dominant paradigm for aligning language models with human values  [ 9 ,  23 ,  3 ] . Reward models trained via RLHF achieve 95% accuracy on RewardBench’s objective tasks  [ 14 ] , which emphasize safety violations, factual errors, and instruction-following. However, our benchmark reveals a critical limitation: when we systematically remove objective quality signals (grammatical errors, factual mistakes, length differences), sequence-based reward models—the dominant architecture in production RLHF systems—collapse to 52.7% accuracy on writing preference tasks, barely above random chance. This 42-percentage-point degradation indicates that current preference learning primarily optimizes for error detection rather than recognition of subjective creative quality—a fundamental misalignment between training objectives and the aesthetic judgment required for creative tasks.

However, writing tasks constitute over 40% of language model interactions  [ 22 ,  2 ] , spanning creative fiction, persuasive essays, and personal expression where subjective quality matters more than objective correctness. Yet our evaluation infrastructure remains anchored in verifiable metrics. RewardBench  [ 14 ]  conflates safety with preference; WritingBench mixes creative with functional tasks  [ 31 ] ; LitBench uses Reddit upvotes as quality proxies  [ 11 ] . Moreover, existing benchmarks predominantly focus on English, leaving cross-lingual preference evaluation, particularly for languages with distinct rhetorical traditions like Chinese, largely unexplored. Recent theoretical work warns of “reward hacking” where models exploit spurious correlations rather than learning genuine preferences  [ 24 ] .

Figure 1 :

WritingPreferenceBench  isolates subjective writing quality by neutralizing objective confounds (grammar, factuality, length). Across 1,800 human-validated preference pairs, standard sequence classifiers (SC-RM) perform near-randomly while generative reward models (GenRM) achieve 30% higher accuracy—but both architectures exhibit catastrophic instability across genres, exposing the brittleness of current preference learning.

We introduce  WritingPreferenceBench , a cross-lingual dataset of 1,800 human-annotated preference pairs (1,200 English, 600 Chinese) across 8 creative writing genres
where both responses are grammatically correct, factually accurate, and length-matched. We focus on three dimensions of creative quality:  creativity  (original ideas and novel perspectives),  stylistic sophistication  (narrative techniques and linguistic elegance), and  emotional resonance  (capacity to evoke authentic responses). By neutralizing objective confounds, our benchmark tests whether models can recognize these aesthetic qualities that distinguish compelling from merely competent writing.

Our evaluation of 21 models, comprising 7 reward models and 14 language models as judges spanning open-source and proprietary families, reveals fundamental limitations in current preference learning approaches. Sequence-based reward models, the dominant architecture in production RLHF systems  [ 25 ] , achieve only 52.7% mean accuracy across both languages, while zero-shot language model judges  [ 32 ]  reach 53.9%. Both results are statistically indistinguishable from random chance. In stark contrast, generative reward models that produce explicit reasoning chains  [ 6 ]  achieve 81.8% accuracy. This 29-percentage-point gap indicates that subjective preference modeling requires structured intermediate reasoning rather than direct pattern matching. Moreover, all architectures exhibit severe genre instability (mean standard deviation 10.9%), with individual models ranging from 18.2% to 81.8% accuracy across categories, suggesting reliance on superficial heuristics rather than generalizable aesthetic principles. Notably, these failures persist across model scales: 27B parameter models show no consistent improvement over 8B variants, and reasoning-enhanced LLMs (Claude-4-Opus-thinking, OpenAI-o3) provide no advantage over standard architectures.

Contributions.  We make three contributions to understanding preference learning:

•

Benchmark : WritingPreferenceBench provides 1,800 validated preference pairs with systematic signal isolation across English and Chinese, enabling reproducible cross-lingual evaluation of subjective preference modeling.

•

Empirical findings : Comprehensive evaluation establishes that (i) sequence classifiers fail systematically on subjective tasks, (ii) generative reward models with reasoning achieve 30% higher accuracy, and (iii) zero-shot LLM judges cannot reliably assess creative quality despite instruction tuning.

•

Architectural insights : Evidence that successful preference learning requires intermediate reasoning representations, not just pattern matching, with implications for next-generation RLHF systems.

2  WritingPreferenceBench

The fundamental challenge in evaluating subjective writing is not merely collecting human judgments, but ensuring those judgments isolate genuine aesthetic and stylistic preference from objective quality signals. We present  WritingPreferenceBench , a meticulously constructed benchmark that addresses this challenge through 1,800 preference pairs spanning English and Chinese creative writing. The construction process, illustrated in Figure

2  , was guided by rigorous design principles and implemented through a human-in-the-loop pipeline designed to systematically eliminate confounding variables.

Figure 2 :

The data curation pipeline of  WritingPreferenceBench . Our multi-stage process begins with expert-crafted queries across 51 genres, generates diverse responses using 20 state-of-the-art models, and culminates in rigorous human evaluation by trained annotators. Quality control mechanisms operate throughout to ensure preference pairs reflect genuine subjective quality distinctions rather than objective differences.

2.1  Benchmark Construction Pipeline

We implemented a multi-stage pipeline that translates our design principles into a concrete, reproducible workflow. This process, depicted in Figure

2  , first generates a diverse and culturally-rich set of candidate responses and then app