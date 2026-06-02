[2602.00425] Segment-Level Attribution for Selective Learning of Long Reasoning Traces

Segment-Level Attribution for Selective Learning of Long Reasoning Traces

Siyuan Wang, Yanchen Liu, Xiang Ren
 University of Southern California

{sw_641,liuyanch,xiangren}@usc.edu

Abstract

Large Reasoning Models (LRMs) achieve strong reasoning performance by generating long chains of thought (CoTs), yet only a small fraction of these traces meaningfully contributes to answer prediction, while the majority contains repetitive or truncated content. Such output redundancy is further propagated after supervised finetuning (SFT), as models learn to imitate verbose but uninformative patterns, which can degrade performance.
To this end, we incorporate integrated gradient attribution to quantify each token’s influence on final answers and aggregate them into two segment-level metrics: (1)  attribution strength  measures the overall attribution magnitude; and (2)  direction consistency  captures whether tokens’ attributions within a segment are uniformly positive or negative (high consistency), or a mixture of both (moderate consistency).
Based on these two metrics, we propose a segment-level selective learning framework to identify important segments with high attribution strength but moderate consistency that indicate reflective rather than shallow reasoning.
The framework then applies selective SFT on these important segments while masking loss for unimportant ones.
Experiments across multiple models and datasets show that our approach improves accuracy and output efficiency, enabling more effective learning from long reasoning traces

1

1  1 Code and data are available at  https://github.com/SiyuanWangw/SegmentSelectiveSFT .

.

1  Introduction

Recent Large Reasoning Models (LRMs)  (Jaech  et al. ,  2024 ; Guo  et al. ,  2025 ; Yang  et al. ,  2025 )  have demonstrated strong capabilities in solving complex problems. Their effectiveness is largely attributed to test-time scaling by increasing computation during inference to produce extended chains of thought (CoT)  (Wei  et al. ,  2022 )  that include detailed problem understanding, step-by-step solution processes, and comprehensive verification. These long-CoT trajectories have also become valuable supervisory resources for cold-start supervised finetuning (SFT)  (Muennighoff  et al. ,  2025 ) .

However, current reasoning CoTs often span thousands of tokens, of which only a small fraction meaningfully contributes to reaching the correct answer or improving confidence  (Sui  et al. ,  2025 ) . A substantial portion consists of redundant repetition or incomplete truncated thoughts  (Wang  et al. ,  2025d ) , as illustrated in Fig.

1

(left).
More critically, verbosity without substance actively degrade reasoning performance as CoT length increases  (Wu  et al. ,  2025 ; Huang  et al. ,  2025 ) . This phenomenon is also evident by the right-top panel of Fig.

1

where incorrect CoTs are typically correlated with more segments and tokens than correct CoTs for the same queries. Training LLMs on verbose CoT supervision with sparse positive contribution further exacerbates these issues. Models learn to imitate redundant behaviors, waste learning capacity on trivial continuations, and fail to prioritize the crucial high-impact parts of reasoning sequences  Lin  et al.  ( 2024 ) . As a result, finetuned models tend to achieve limited accuracy gains and generate inefficient outputs.

Figure 1:

Left : An illustrative CoT with important (green blocks) and redundant segments (gray blocks). Our metrics distinguish important from redundant segments (repetitions, truncations, superficial clarifications) with low strength and extremely high consistency. “Attribution Stre.” denotes normalized strength across all segments.  Right-top : Segment and token counts in correct vs. incorrect CoTs for the same queries.  Right-bottom : Cumulative distribution function (CDF) of normalized segment strength in correct and incorrect CoTs, with segment ordered in descending strength.

Prior studies have explored various strategies to identify important parts in long reasoning chains to construct compressed CoT supervision for efficiency purposes. However, they either focus on fine-grained token-level analysis  (Xia  et al. ,  2025b )  that neglects semantic integrity and fails to interpret redundancy in terms of meaningful reasoning units, or rely on segment-level perplexity  (Cui  et al. ,  2025b )  or entropy  (Li  et al. ,  2025b )  calculations. These indirect metrics provide not entirely consistent signals of importance and are prone to both false positives and false negatives. False positives occur when methods overemphasize superficial scaffolding text (e.g., “So, let’s calculate step by step”) that contributes little to actual reasoning yet serves as linguistic bridges whose removal disrupts subsequent textual coherence. False negatives arise when methods filter out independent verification or intermediate conclusions that, while exhibiting low-entropy and their removal not affecting linguistic fluency, significantly enhance the probability of reaching correct final answers.
Consequently, existing methods cannot accurately and comprehensively distinguish truly important segments from various forms of redundant content that meaninglessly contribute to reasoning accuracy.

In this work, we systematically identify important segments that directly contribute to correct answer prediction within long CoTs, and show that unimportant segments cluster into distinctive redundant patterns. Specifically, we utilize integrated gradient (IG) attribution  (Sundararajan  et al. ,  2017 )  to calculate each token’s direct influence on improving correct answer prediction and aggregate token-level attributions at the segment level to obtain two metrics:
(1)  Attribution strength  quantifies the overall magnitude of a segment’s influence on the model’s prediction by summing absolute IG values within each segment with length normalization.
(2)  Attribution direction consistency  is defined as the ratio between the absolute sum of signed IG attributions and the sum of absolute IG attributions, captures how uniformly a segment contributes in one direction (either positively or negatively toward the correct answer). Extremely high consistency often reflects shallow or skewed reasoning, such as segments with uniformly positive token IGs that merely provide superficial clarification (see the penultimate segments in Fig.

1

left), or uniformly negative token IGs that severely distort the reasoning process toward incorrect conclusions. In contrast, moderate consistency indicates more reflective reasoning that mixes supportive and corrective attribution within a segment, which is more critical for problem solving.

We first verify significant redundancy in long CoTs using  attribution strength , showing that 30

∼  \sim

40% of segments accumulate over 80% of total attribution in both correct and incorrect CoTs (Fig.

1  , right-bottom).
Building on this insight, we propose a segment-level selective learning framework that learns the most critical parts of long CoTs leveraging  attribution strength  and  direction consistency . It identifies segments with high strength but moderate consistency as important, filtering out redundancies like repeated content, truncated thoughts, and dispensable clarifications with minimal gains on correct answer probability. Unlike pruning-based SFT methods that compress CoT supervision but compromise accuracy, our framework applies selective SFT  (Lin  et al. ,  2024 )  that trains only on important segments while masking loss for unimportant ones. This acts as implicit regularization by preventing overfitting to uninformative content. Experiments across multiple models and datasets, using both self-generated and reference long CoTs, show that our method outperforms full-CoT SFT by improving reasoning efficacy (up to 4.7%) wh