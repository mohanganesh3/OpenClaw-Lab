[2512.05112] DraCo: Draft as CoT for Text-to-Image Preview and Rare Concept Generation

DraCo : Draft as CoT for Text-to-Image Preview and Rare Concept Generation

Dongzhi Jiang  ∗1  , Renrui Zhang  ∗†1

🖂 , Haodong Li  4  , Zhuofan Zong  1  , Ziyu Guo  2  , Jun He  3  , Claire Guo  5  ,

Junyan Ye  3  , Rongyao Fang  1  , Weijia Li  3  , Rui Liu  1

🖂 , Hongsheng Li  1

🖂

1 CUHK MMLab  2 CUHK IMIXR  3 Sun Yat-Sen University  4 SCUT  5 CUHK (Shenzhen)

{dzjiang, renruizhang}@link.cuhk.edu.hk

∗

Equal Contribution

† Project Leader

🖂 Corresponding author

Abstract

Recent unified multimodal large language models (MLLMs) have shown impressive capabilities, incorporating chain-of-thought (CoT) reasoning for enhanced text-to-image generation. However, existing approaches remain limited, either treating the model merely as a standalone generator or relying on abstract textual planning.
To this end, we propose  Draft-as-CoT (DraCo) , a novel interleaved reasoning paradigm that fully leverages both textual and visual contents in CoT for better planning and verification.
Our method first generates a low-resolution draft image as preview, providing more concrete and structural visual planning and guidance.
Then, we employ the model’s inherent understanding capability to verify potential semantic misalignments between the draft and input prompt, and performs refinement through selective corrections with super-resolution.
In this way, our approach addresses two fundamental challenges:
the coarse-grained nature of textual planning and the difficulty in generating rare attribute combinations.
To support training, we curate DraCo-240K, aiming to enhance three atomic capabilities spanning general correction, instance manipulation, and layout reorganization.
Supported by DraCo-CFG, a specialized classifier-free guidance (CFG) strategy for interleaved reasoning, DraCo achieves a tremendous increase on GenEval (+8%), Imagine-Bench (+0.91), and GenEval++ (+3%), significantly outperforming direct generation and other generation methods empowered by CoT.
The project is at  https://github.com/CaraJ7/DraCo .

1  Introduction

Figure 1 :

Conceptual Comparison of CoT Reasoning for T2I Generation.  (a) Generation without reasoning. (b) Employing exterior reward models to guide generation. (c) Generating Text CoT before producing image. (d)  DraCo : Producing visual draft for detailed planning and verify it with text reasoning, then correct and refine the draft for final output.

Figure 2 :

Visualization of  DraCo  Output.  For each example, the larger image represents the final output, while the smaller image is the visual draft. The corresponding prompt is located in the corner of each set.

Recently, unified multimodal large language models (MLLMs)  [ bagel ,  wu2024janus ,  wu2024vila ,  zhou2024transfusion ,  chen2025blip3o ,  emu2 ,  xiao2025omnigen ,  team2024chameleon ,  liquid ,  qu2024tokenflow ,  lin2025uniworld ]  have emerged as powerful architectures that integrate both visual understanding and generation capabilities, as exemplified by recent works such as Bagel  [ bagel ] , EMU3  [ wang2024emu3 ] , and Janus  [ wu2024janus ] . By consolidating these two abilities within a single framework, unified MLLMs have demonstrated remarkable performance and exhibit emergent properties that models possessing only one ability fail to acquire. For instance, unified MLLMs can understand interleaved context including both images and texts and subsequently produce images according to the input instruction  [ wu2025omnigen2 ,  xin2025lumina ] .

Concurrently, advances in chain-of-thought (CoT) reasoning  [ wei2022chain ,  kojima2022large ]  have shown remarkable success across various domains, including mathematical problems  [ amini2019mathqa ,  hendrycksmath2021 ,  Lu2023MathVistaEM ,  zhang2024mathverse ,  zhang2024mavis ] , visual reasoning  [ yue2023mmmu ,  jiang2025mme ,  chen2025r1v ,  zhan2025visionr1 ,  meng2025mm ] , and multi-agent systems  [ mai2025agent ,  wei2025webagent ] . While several pioneering works have extended CoT to text-to-image (T2I) generation tasks  [ guo2025can ,  jiang2025cot ,  bagel ,  zhang2025reasongen ,  pan2025focusdiff ,  gu2025improving ]  on unified MLLMs, their exploration has not fully exploited the unified architecture of MLLMs. For example, Image-Gen-CoT  [ guo2025can ]  leverages a reward model to evaluate the quality potential of an image during the early generation stage. This paradigm treats unified MLLM merely as a text-to-image generator, only employing its image generation capabilities.
Later, several methods  [ bagel ,  jiang2025cot ]  have been proposed to generate textual reasoning for the given prompt prior to image synthesis. However, for generating a dense modality like an image, only planning with text offers too vague and coarse guidance. Besides, only textual understanding ability is leveraged for CoT.
This raises a natural question:  Can we design a CoT mechanism with both textual and visual content as better planning and verifier for improved text-to-image generation?

Figure 3 :

Framework of  DraCo .

DraCo  contains three steps for generation: draft sketching, draft verification, and corrective refinement.

To answer this question, we propose  Draft-as-CoT ( DraCo ) , an interleaved CoT reasoning approach that fully leverages the unified framework for the T2I task.
Our method first generates a low-resolution draft image as visual planning, then leverages the model’s inherent understanding capability to verify the draft with any semantic misalignments, and ultimately refines and enriches the draft through selective correction and super-resolution. We show the comparison of  DraCo  with previous methods in Fig.

1  .

Our method is motivated by two key limitations of textual reasoning for T2I generation. First, text is too abstract to design every aspect of an image in detail, especially on low-level features like appearance and styles. In contrast, planning with a draft image could sufficiently display all the essential visual information during planning. Second, reasoning with a draft image provides an opportunity to preview the image to generate.
This addresses a fundamental challenge where current T2I models suffer: difficulty in producing the correct image in one pass. Specifically, due to the natural distribution of real-world data, rare attributes or object combinations are significantly underrepresented in training datasets  [ samuel2024generating ,  chen2025failureatlas ] . For example, when prompted with “a white orange”, models often struggle because they have learned to strongly associate the object “orange” with its typical color attribute rather than treating them as independent concepts. This binding leads to systematic failures in generating unusual yet valid attribute combinations. However, by previewing the draft, we do not force the model to directly generate the perfect image. Instead, we make the unified MLLM itself identify and refine its flawed planning for the final output.

To facilitate our proposed  DraCo  reasoning paradigm, the unified MLLM must be capable of identifying mistakes and precisely controlling and operating on drafts to ensure successful corrections. Since no existing dataset provides such capabilities, we carefully curate a training dataset,  DraCo -240K, targeting three atomic correction capabilities. For each capability, we design a rigorous data synthesizing pipelines that full exploits the synergy between the MLLM  [ qwen3technicalreport ] , advanced editing models  [ labs2025flux1kontextflowmatching ] , and segmentation models  [ ren2024grounded ] .
Additionally, how to conduct classifier-free guidance (CFG)  [ ho2022classifier ]  for interleaved reasoning remains an open question. We propose  DraCo -CFG, a specific type of CFG for  DraCo  to explicitly strengthen the two major conditions for final generation: visual semantics from the draft image and correc