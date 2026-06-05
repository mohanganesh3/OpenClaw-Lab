[2506.05344] SparseMM: Head Sparsity Emerges from Visual Concept Responses in MLLMs

SparseMM: Head Sparsity Emerges from Visual Concept Responses in MLLMs

Jiahui Wang 1 ,  Zuyan Liu 1,2

1

1  footnotemark:

1

,  Yongming Rao 2,1 ,  Jiwen Lu 1

†

1  Tsinghua University

2  Tencent Hunyuan Research

Authors contributed equally to this research.

† Corresponding author.

Abstract

Multimodal Large Language Models (MLLMs) are commonly derived by extending pre-trained Large Language Models (LLMs) with visual capabilities. In this work, we investigate how MLLMs process visual inputs by analyzing their attention mechanisms. We reveal a surprising sparsity phenomenon: only a small subset (approximately less than 5%) of attention heads in LLMs actively contribute to visual understanding, termed  visual heads . To identify these heads efficiently, we design a training-free framework that quantifies head-level visual relevance through targeted response analysis. Building on this discovery, we introduce  SparseMM , a KV-Cache optimization strategy that allocates asymmetric computation budgets to heads in LLMs based on their visual scores, leveraging the sparity of visual heads for accelerating the inference of MLLMs. Compared with prior KV-Cache acceleration methods that ignore the particularity of visual, SparseMM prioritizes stress and retaining visual semantics during decoding. Extensive evaluations across mainstream multimodal benchmarks demonstrate that SparseMM achieves superior accuracy-efficiency trade-offs. Notably, SparseMM delivers 1.38× real-time acceleration and 52% memory reduction during generation while maintaining performance parity on efficiency test. Our project is open sourced at  https://github.com/CR400AF-A/SparseMM .

1  Introduction

Figure 1 :

Head Sparsity Emerges from Visual Concept Responses.  We observe the visual-relevant heads are sparse in various MLLMs. Based on this observation, we devise a KV-Cache optimization strategy that allocates asymmetric budgets to LLM heads based on their importance for visual tokens, achieving better trade-off under limited computational resources.

Autoregressive large language models (LLMs)  [  39  ,

37  ,

14  ,

6  ,

45  ]  have revolutionized artificial intelligence with their exceptional instruction-following capabilities and expansive knowledge repositories. Building upon this foundation, researchers have extended LLMs to multimodal domains, particularly in vision-language integration, creating multimodal large language models (MLLMs)  [  21  ,

3  ,

26  ,

7  ,

53  ,

46  ]  that process both textual and visual inputs. Current approaches typically augment pre-trained LLMs by incorporating visual encoders (e.g., CLIP  [  42  ]  or SigLIP  [  55  ] ) paired with lightweight adapters to project visual features into the language model’s hidden space. While these architectures demonstrate remarkable multimodal reasoning abilities, how LLMs fundamentally acquire visual comprehension during supervised fine-tuning remains poorly understood. This knowledge gap constrains our ability to recognize cross-modal alignment and risks undervaluing visual semantics during multi-modal relevant tasks and applications, which may potentially leading to suboptimal architecture designs and inefficient computational resource allocation.

To this end, we present the first systematic investigation into how visual concepts are processed within LLMs. Through rigorous analysis of attention mechanisms, we uncover a critical phenomenon that only a small subset of attention heads (termed  visual heads ) disproportionately drive visual content understanding, while the majority remain text-specialized. Specifically, our experiments reveal two critical properties of these visual heads: (1)  Sparsity : Less than 5% of attention heads are intrinsically visual-active across layers, even in models trained with extensive multimodal data; (2)  Universality : Visual heads emerge consistently across diverse LLM architectures(e.g., Vicuna  [  8  ]  and Qwen2  [  41  ] ) and generalize to multiple attention paradigms such as multi-head attention (MHA)  [  49  ]  and grouped query attention (GQA)  [  2  ] .

To systematically identify these visual heads, we propose a training-free framework that quantifies the visual relevance of attention heads through targeted cross-modal response analysis. Specifically, our approach leverages OCR as an anchor task to establish precise correspondence between text outputs and visual inputs: for each generated word, we trace its activation back to spatially aligned image patches, enabling direct measurement of how specific attention heads mediate visual-text alignment. By analyzing and recoding the attention score of all the attention heads across a certain amount of samples, we compute visual scores that rank heads by their visual responsiveness. Crucially, while our identification mechanism relies on OCR’s granular spatial grounding, we demonstrate that the detected visual heads exhibit task-agnostic generalizability—they remain dominant in diverse vision-language tasks including object recognition and scene understanding.

Building on these insights, to demonstrate the effectiveness of visual heads on practical multi-modal tasks, we introduce  SparseMM , a KV-Cache optimization framework that exploits visual head sparsity to achieve accelerated inference. As multimodal inputs grow in complexity—spanning multi-turn dialogues  [  52  ,

19  ,

20  ] , high-resolution interleaved images  [  51  ,

9  ] , and dense video/3D sequences  [  12  ,

23  ,

16  ] —the computational overhead of maintaining full KV-Caches becomes prohibitive. Existing compression methods, however, treat all attention heads uniformly, disregarding the critical role of sparse visual heads in encoding visual semantics.

SparseMM addresses this by asymmetrically allocating KV-Cache budgets: visual heads receive prioritized retention based on their precomputed visual scores, while non-visual heads undergo aggressive compression via a hybrid strategy combining 1)  Score-Preferred Cache  (allocating cache budget based on visual head scores), 2)  Uniform-Based Cache  (preserving minimal budget for all the heads), and 3)  Local Window Cache  (preserving cache budget for recent tokens). This mixed approach ensures better accuracy-efficiency trade-offs, such that visual heads retain more computational cost while other heads are dynamically throttled.

Extensive experimental results demonstrate that SparseMM outperforms other strong baselines across multiple datasets, including DocVQA  [  35  ] , OCRBench  [  29  ] , TextVQA  [  44  ] , MMBench  [  28  ] , etc. For instance, on DocVQA, LLaVA-NeXT-Vicuna-7B  [  27  ]  achieves the same level of accuracy while using only 20% of the cache, and Qwen2-VL-7B-Instruct  [  41  ]  achieves equivalent performance with just 5.3% of the cache. These findings suggest that our method effectively captures visual information while compressing redundancies. Furthermore, the reduction in cache requirements enables our method to achieve lower decoding latency and peak memory usage. For example, LLaVA-NeXT-Mistral-7B  [  27  ]  maintains nearly constant decoding latency with 32K input tokens, resulting in almost a 50% acceleration compared to the full model, and reduces memory usage by 5GB.

2  Related Works

Architectures in MLLMs.

The predominant architecture for Multi-Modal Large Language Models (MLLMs) consists of three key components: a visual encoder, an adapter, and a LLM. By leveraging alignment training techniques and subsequent fine-tuning, this integrated framework has achieved remarkable performance on various multi-modal understanding tasks  [  38  ,

19  ,

24  ,

1  ,

30  ,

32  ,

10  ] . In typical implementations, the visual encoder is realized using models like CLIP  [  42  ]  or SigLIP  [  55  ] , which are adept at extracting rich visual representations. The adapter component