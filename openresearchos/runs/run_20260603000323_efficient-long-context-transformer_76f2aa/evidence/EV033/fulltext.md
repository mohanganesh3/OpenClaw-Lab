[2603.07572] This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible. TS-MLLM: A Multi-Modal Large Language Model-based Framework for Industrial Time-Series Big Data Analysis

This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.

TS-MLLM: A Multi-Modal Large Language Model-based Framework for Industrial Time-Series Big Data Analysis

Haiteng Wang

, 
Yikang Li

, 
Yunfei Zhu

, 
Jingheng Yan

, 
Lei Ren

, 
and Laurence T. Yang

The research is supported by The NSFC (National Science Foundation of China) project No.62225302, 623B2014, 62173023. (Corresponding authors: Lei Ren.)
Haiteng Wang, Yikang Li, and Jingheng Yan are with the School of Automation Science and Electrical Engineering, Beihang University, Beijing 100191, China (email: wanghaiteng@buaa.edu.cn, liyikang@buaa.edu.cn, yjh967@buaa.edu.cn).
Yunfei Zhu is with the School of Software, Beihang University, Beijing 100191, China (email: zhuyunfei@buaa.edu.cn).
Lei Ren is with the School of Automation Science and Electrical Engineering, Beihang University, Beijing 100191, China, also with the Hangzhou International Innovation Institute, Beihang University, Hangzhou 311115, China, and also with the State Key Laboratory of Intelligent Manufacturing System Technology, Beijing 100854, China (e-mail: renlei@buaa.edu.cn).
Laurence T. Yang is with the School of Computer and Artificial Intelligence, Zhengzhou University, Zhengzhou 450001, China, and also with the Department of Computer Science, St. Francis Xavier University, Antigonish, NS B2G 2W5, Canada (e-mail: ltyang@ieee.org).

Abstract

Accurate analysis of industrial time-series big data is critical for the Prognostics and Health Management (PHM) of industrial equipment. While recent advancements in Large Language Models (LLMs) have shown promise in time-series analysis, existing methods typically focus on single-modality adaptations, failing to exploit the complementary nature of temporal signals, frequency-domain visual representations, and textual knowledge information. In this paper, we propose TS-MLLM, a unified multi-modal large language model framework designed to jointly model temporal signals, frequency-domain images, and textual domain knowledge. Specifically, we first develop an Industrial time-series Patch Modeling branch to capture long-range temporal dynamics. To integrate cross-modal priors, we introduce a Spectrum-aware Vision-Language Model Adaptation (SVLMA) mechanism that enables the model to internalize frequency-domain patterns and semantic context. Furthermore, a Temporal-centric Multi-modal Attention Fusion (TMAF) mechanism is designed to actively retrieve relevant visual and textual cues using temporal features as queries, ensuring deep cross-modal alignment. Extensive experiments on multiple industrial benchmarks demonstrate that TS-MLLM significantly outperforms state-of-the-art methods, particularly in few-shot and complex scenarios. The results validate our framework’s superior robustness, efficiency, and generalization capabilities for industrial time-series prediction.

I

Introduction

Industrial time-series big data is the cornerstone of Prognostics and Health Management (PHM), essential for ensuring equipment reliability [ 6 ] . Rather than simple 1D signals, industrial time-series data involves the integration of multi-modal derived information, such as raw signals, frequency spectrum images, and textual semantic context. The analysis of big data from industrial sensors enables the precise capture of dynamic state variations and the proactive identification of latent failures, thereby significantly enhancing equipment reliability [ 10 ,  16 ] .

Deep learning methods, including RNNs [ 35 ,  37 ] , CNNs [ 12 ] , Transformers [ 19 ] , and Diffusion models [ 21 ,  24 ] , have been widely explored for PHM tasks. These methods leverage advanced techniques such as frequency decomposition, signal feature extraction, and multi-scale learning to model complex temporal dynamics. However, despite their high accuracy, these models suffer from limited generalization, especially in few-shot and zero-shot scenarios.

Pre-trained Large Language Models (LLMs) [ 9 ,  31 ,  8 ]  have emerged as a promising solution. Benefiting from training on vast corpora, these models exhibit robust generalization capabilities, making them helpful for complex time-series analysis. This paradigm can be further divided into two strategies: encoder-based and prompt-based methods. In the former, LLMs are treated as high-capacity feature encoders to directly process raw or patched time-series data  [ 32 ,  39 ] , effectively utilizing the model’s pre-trained attention mechanisms to extract universal temporal representations. The latter strategy, known as prompt-based modeling (e.g., PromptCast  [ 30 ] ), reformulates continuous data into discrete textual descriptions, attempting to activate the LLM’s inherent zero-shot reasoning through language prompts.

Building upon these foundations, recent research has shifted towards leveraging multi-modal large language models (MLLMs) to bridge the gap between time-series data and high-level semantic fusion. This evolution has led to two distinct augmented paradigms: text-augmented and vision-augmented modeling. Text-augmented frameworks, such as Time-LLM  [ 11 ] , align temporal patches with textual domain knowledge, enabling the model to incorporate expert priors into its prognostic reasoning. Simultaneously, vision-augmented approaches (e.g., Vision-TS  [ 3 ] , DiagLLM  [ 26 ] ) transform 1D sequences into 2D spectrograms or recurrence plots, tapping into the superior feature extraction power of visual-language encoders to capture morphological fault signatures.

Although recent progress has been made in incorporating textual and visual information into modeling, the joint modeling of industrial time-series data with these modalities remains scarcely investigated. Current studies are typically limited to single-modality adaptations, failing to exploit the intrinsic complementarity between temporal, visual, and textual information. Specifically, time-series models capture fine-grained dynamics but miss global morphological patterns, while vision-based models grasp structural signatures but lose temporal resolution. Furthermore, the representation misalignment between continuous signals and discrete tokens remains a significant hurdle. Consequently, there is an urgent need for a unified multi-modal framework that can synergize these diverse information to enhance robustness and generalization in complex industrial environments.

To bridge this gap, we propose TS-MLLM, a unified multi-modal framework that exploits MLLMs to jointly model temporal, visual, and textual information. In our paradigm, text conveys domain knowledge, visual features (derived from frequency-domain representations) capture spectral patterns, and time-series encoding represents continuous temporal dynamics. Specifically, TS-MLLM first employs an industrial time-series patch modeling branch to adaptively capture long-range temporal evolutions through patch-based Transformer blocks. To leverage cross-modal priors, a Spectrum-aware Vision-Language Model Adaptation (SVLMA) module is introduced to align frequency-domain spectrum images with textual domain knowledge within a pre-trained LLM space. Finally, a Temporal-centric Multi-modal Attention Fusion (TMAF) mechanism utilizes temporal features as queries to actively retrieve and integrate supportive cues from the MLLM via a multi-modal attention mechanism. This deep alignment ensures robust performance in complex industrial tasks, such as remaining useful life (RUL) estimation. The main contributions of this work are:

1) We propose TS-MLLM, a multi-modal large langua