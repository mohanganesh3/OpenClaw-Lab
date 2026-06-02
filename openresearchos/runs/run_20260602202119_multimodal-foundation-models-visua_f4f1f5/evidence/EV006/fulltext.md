[2511.17952] Multi-speaker Attention Alignment for Multimodal Social Interaction

Multi-speaker Attention Alignment for Multimodal Social Interaction

Liangyang Ouyang  Yifei Huang  Mingfang Zhang

Caixin Kang  Ryosuke Furuta  Yoichi Sato
 The University of Tokyo, Tokyo, Japan

{oyly, hyf, mfzhang, cxkang, furuta, ysato}@iis.u-tokyo.ac.jp

Abstract

Understanding social interaction in video requires reasoning over a dynamic interplay of verbal and non-verbal cues: who is speaking, to whom, and with what gaze or gestures.
While Multimodal Large Language Models (MLLMs) are natural candidates, simply adding visual inputs yields surprisingly inconsistent gains on social tasks.
Our quantitative analysis of cross-modal attention inside state-of-the-art MLLMs reveals a core failure mode: in multi-speaker scenes, visual and textual tokens lack speaker-consistent alignment, exhibiting substantially weaker cross-modal attention than in object-centric images.
To address this, we propose a multimodal multi-speaker attention alignment method that can be integrated into existing MLLMs. First, we introduce dynamic cross-modal head selection to identify attention heads most responsible for grounding.
Then, an adaptive social-aware attention bias, computed from existing attention patterns and speaker locations, is injected into the attention mechanism.
This bias reinforces alignment between a speaker’s visual representation and their utterances without introducing trainable parameters or architectural changes.
We integrate our method into three distinct MLLMs (LLaVA-NeXT-Video, Qwen2.5-VL, and InternVL3) and evaluate on three benchmarks (TVQA+, MMSI, OnlineMMSI). Across four social tasks, results demonstrate that our approach improves the ability of MLLMs and achieves state-of-the-art results.
Attention visualizations confirm our method successfully focuses the model on speaker-relevant regions, enabling more robust multi-party social reasoning. Our implementation and model will be available at  https://github.com/ut-vision/SocialInteraction .

1  Introduction

Understanding social interaction requires modeling multi-party human behaviors through both verbal and non-verbal cues, including dialogue  [ kang2025can ] , gestures  [ cao2025socialgesture ] , gaze  [ zhou2024detecting ] , and facial expressions  [ hyun2024smile ] .
To study these interactions, prior works have proposed a variety of tasks and benchmarks, such as video question answering (VQA), speaking target detection, mentioned player prediction, and pronoun coreference resolution  [ lei2020tvqa+ ,  lee2024modeling ] .
Beyond serving as evaluation platforms, these tasks underpin socially intelligent AI agents that operate in real-world multi-party scenarios like board games, daily conversations, and meetings.

Given their ability to comprehend both verbal and non-verbal information, multimodal large language models (MLLMs) are natural candidates for these tasks  [ lee2024modeling ,  li2025towards ,  park2025assessing ] .
However, our analysis reveals a critical limitation: the addition of visual information does not consistently improve, and can even degrade their performance in multi-person settings.
For example, on OnlineMMSI  [ li2025towards ] , supplying video frames to Qwen2.5-VL  [ bai2025qwen2 ]  input yields no gain on the mentioned player prediction task, while LLaMA-3.2-Vision  [ dubey2024llama ]  sees its performance drop on the pronoun coreference resolution task  [ li2025towards ] .
These observations suggest that current MLLMs struggle to effectively exploit multimodal cues in complex multi-person social settings.

To better understand why MLLMs fail to leverage multimodal cues, we conduct a systematic quantitative analysis of cross-modal attention weights inside state-of-the-art MLLMs  [ bai2025qwen2 ] .
By measuring the attention weights between a speaker’s textual tokens and their corresponding visual region ( i.e. , their bounding boxes), we uncover a stark deficiency. We find that the cross-modal alignment in multi-person videos is significantly weaker and less focused compared to the alignment observed in general object-centric datasets like COCO  [ lin2014microsoft ] .
This limitation results in inconsistent alignment between visual and textual modalities, thereby constraining the effectiveness of MLLMs in multi-person social tasks.

Figure 1 :

We propose a multimodal multi-speaker attention alignment method for MLLMs to understand social interactions in videos. Visualization of cross-attention weights in transformer layers confirms that our approach strengthens the model’s focus on areas relevant to the active speaker.

To address this misalignment problem, we propose a multimodal multi-speaker attention alignment method.
Our approach intervenes directly within the transformer’s cross-attention layers.
We first propose a  dynamic cross-modal head selection  strategy that identifies attention heads most responsible for visual-text grounding.
We then apply an  adaptive social-aware attention bias  to these heads, which amplifies the attention scores between the visual and textual tokens belonging to the same speaker.
As illustrated in

Fig.

1  , this mechanism explicitly guides the model to associate the correct visual features with the corresponding dialogue. Crucially, this mechanism requires no architectural changes or additional trainable parameters, making it a lightweight and generalizable solution.

We evaluate our method on three multimodal social interaction benchmarks (TVQA+  [ lei2020tvqa+ ] , MMSI  [ lee2024modeling ] , and OnlineMMSI  [ li2025towards ] ) across four representative tasks.
Integrated into three modern MLLMs, LLaVA-NeXT-Video  [ zhang2024llavanextvideo ] , Qwen2.5-VL  [ bai2025qwen2 ]  and InternVL3  [ zhu2025internvl3 ] , our method consistently outperforms their respective baselines, yielding an average accuracy improvement of 3% across multiple datasets and tasks.
It achieves state-of-the-art performance on three task settings and remains highly competitive on the remaining one.
Attention visualizations further confirm that our approach successfully guides the model to focus on speaker-relevant regions in videos.

Our main contributions are summarized as follows:

•

We are the first to systematically quantify and identify the cross-modal attention misalignment in MLLMs as a key bottleneck for understanding multi-party interactions.

•

We propose a novel attention alignment method that dynamically reinforces the association between speakers’ visual and textual representations without additional trainable parameters.

•

Extensive experiments demonstrate that our method effectively guides model attention to speaker-relevant regions, thereby improving performance in diverse multimodal social interaction tasks.

2  Related Works

2.1  Multimodal Social Interaction

Multimodal social interaction refers to human communication across multiple modalities, including spoken language, facial expressions  [ hyun2024smile ] , gaze  [ zhou2024detecting ,  liu2024pnp ,  liu2025gaze ] , gestures  [ cao2025socialgesture ,  caotoward ] , and body movements  [ balazia2022bodily ] .
Prior research has proposed a variety of related tasks and benchmarks, such as video question answering (VQA)  [ lei2018tvqa ,  zadeh2019social ,  hyun2024smile ,  mathur2025social ,  kong2025siv ] , conversational modeling  [ ryan2023egocentric ,  lee2024modeling ,  jia2024audio ,  chang2025multimodal ] , speaker prediction  [ northcutt2020egocom ,  muller2021multimediate ] , and social behavior classification  [ lai2023werewolf ,  cao2025socialgesture ,  ouyang2025leadership ] . These tasks hold strong potential for enabling AI agents to operate in multi-party social scenarios, including board games  [ lai2023werewolf ,  grauman2022ego4d ,  zhang2025multimind ] , daily conversations  [ northcutt2020egocom ] , and multi-person meetings  [ muller2018detecting ,  kraaij2005ami ] . Leveraging