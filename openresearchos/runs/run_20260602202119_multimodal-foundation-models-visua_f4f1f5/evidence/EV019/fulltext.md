[2508.02951] MedBLINK: Probing Basic Perception in Multimodal Language Models for Medicine

MedBLINK: Probing Basic Perception in Multimodal Language Models for Medicine

Mahtab Bigverdi*  1

Wisdom O. Ikezogwo*  1

Kevin Zhang*  1

Hyewon Jeong

2

Mingyu Lu

1

Sungjae Cho  3

Linda G. Shapiro  1

Ranjay Krishna  1

1  University of Washington,

2  Massachusetts Institute of Technology,

3  Seoul National University

Abstract

Multimodal language models (MLMs) show promise for clinical decision support and diagnostic reasoning, raising the prospect of end-to-end automated medical image interpretation. However, clinicians are highly selective in adopting AI tools; a model that makes errors on seemingly simple perception tasks such as determining image orientation or identifying whether a CT scan is contrast-enhanced—are unlikely to be adopted for clinical tasks. We introduce  MedBLINK , a benchmark designed to probe these models for such perceptual abilities.  MedBLINK  spans eight clinically meaningful tasks across multiple imaging modalities and anatomical regions, totaling 1,429 multiple-choice questions over 1,605 images. We evaluate 19 state-of-the-art MLMs, including general-purpose (GPT‑4o, Claude 3.5 Sonnet) and domain-specific (Med-Flamingo, LLaVA-Med, RadFM) models. While human annotators achieve 96.4% accuracy, the best-performing model reaches only 65%. These results show that current MLMs frequently fail at routine perceptual checks, suggesting the need to strengthen their visual grounding to support clinical adoption. Data is available on our  project page .

1  Introduction

Figure 1 :

The visual tasks that medical professionals could solve within a blink but MLMs struggle. These tasks cover a range of clinically relevant problems, including anatomical orientation, morphology qualification, visual and wave-based depth estimation, and histology analysis.

Would you trust ChatGPT if it failed to identify whether an image was upside down? For artificial intelligence (AI) systems to be adopted, they must demonstrate competence not only on complex benchmarks, but also on simple, intuitive tasks. The same expectation holds perhaps even more critically for AI in medicine, where failures on basic perceptual cues can erode clinician confidence. As multimodal language models (MLMs) increasingly enter clinical settings  [  55  ,

40  ,

41  ] , their reliability on foundational tasks becomes as important as their performance on advanced diagnostic reasoning.

Recent advances in vision-language modeling have sparked enthusiasm about fully automated systems that can interpret medical images and inform clinical decision-making  [  64  ,

38  ,

33  ] . Yet clinicians remain appropriately cautious in adopting AI tools  [  7  ,

22  ,

53  ] . Physicians rely on deeply internalized mental models for interpreting medical images, and they expect AI to match this fluency. Models that fail on what clinicians consider “obvious” tasks like detecting image orientation or identifying contrast-enhancement, risk immediate dismissal, regardless of downstream capabilities  [  5  ] . For example, knowing whether a CT scan is contrast-enhanced directly influences downstream diagnostic interpretation and subsequent clinical decisions  [  28  ] .

These basic assessments, often termed “blink tasks” [  18  ] , occur almost reflexively in expert workflows. They rely on low-effort perceptual and contextual cues, not elaborate reasoning or cross-modal fusion. If a model struggles here, it signals a failure to internalize visual priors critical for real-world use  [  42  ] .
It raises questions about whether the model genuinely “sees” the content of medical images or simply exploits superficial correlations  [  49  ] .

We introduce MedBLINK, a benchmark designed to probe exactly these capabilities. MedBLINK comprises eight
perceptually simple but clinically meaningful tasks chosen by consulting with a senior radiologist. The questions are deliberately simple, asking models to perform basic visual perception rather than complex reasoning.
Figure

1

illustrates one example from each task, including cases like visual depth estimation and image orientation detection. Failures on these tasks would reveal that models struggle to capture spatial relationships and maintain a coherent understanding of anatomical structures.

We evaluate 19 state-of-the-art MLMs, including general-purpose models such as GPT‑4o  [  27  ]  and Claude 3.5 Sonnet  [  1  ] , as well as medical-domain models such as Med-Flamingo  [  41  ] , LLaVA‑Med  [  33  ] , and RadFM  [  60  ] . While human annotators achieve 96.4% accuracy, the best-performing model reaches only 65%.
By probing what models fail to perceive, not just what they fail to predict,  MedBLINK  highlights a fundamental gap in current evaluation protocols. Addressing this gap is essential: models must first master the same low‑effort, common‑sense perceptual tasks before they can be trusted to support real-world diagnostic reasoning and clinical adoption.

2  Related Work

Multimodal Language Models in Healthcare :
Medical image analysis has evolved from early computer-aided detection efforts  [  37  ,

20  ,

58  ]  to recent advances in Multimodal Language Models (MLMs)  [  54  ,

3  ,

46  ,

31  ] . These models combine text and image understanding and are typically evaluated using visual question answering (VQA) tasks. Their adoption in healthcare has enabled progress on diagnostic and report generation tasks across various domains  [  55  ,

64  ,

41  ,

56  ,

33  ,

15  ,

27  ] .

However, limitations persist due to the lack of large-scale multimodal medical datasets and the heterogeneity of medical image formats (e.g., 2D X-rays, 3D CT/MRI, video, gigapixel histology). This has motivated domain-specific models such as VoxelPrompt  [  25  ]  and Dia-LLaMA  [  12  ]  for volumetric imaging, and Quilt-LLaVA  [  50  ] , PathChat  [  38  ] , and PathFinder  [  19  ]  for histopathology.

Multimodal Benchmarks in Medicine : Growing medical MLM development has spurred numerous benchmarks evaluating performance across modalities and tasks, primarily assessing medical knowledge  [  30  ,

34  ,

36  ,

59  ,

24  ,

26  ,

49  ,

47  ,

66  ,

60  ,

65  ,

50  ,

61  ] . SLAKE

[  34  ]  and VQA-RAD

[  30  ]  sample radiology images, from existing datasets, to create clinical diagnostic QA pairs. Path-VQA

[  24  ]  curates pathology images from textbooks with QA pairs from captions. Quilt-VQA

[  50  ]  benchmarks histopathology from pedagogical videos, extracting QA from transcriptions. OmniMedVQA

[  26  ]  develops large-scale VQA covering 12 medical modalities. GMAI-MMBench

[  65  ]  leverages 38 modalities for perceptual tasks beyond diagnosis. CARES

[  61  ]  assesses trustworthiness across 16 modalities in five dimensions: trustfulness, fairness, safety, privacy, robustness. MediConfusion

[  49  ]  probes failure modes on visually dissimilar image pairs. RadVUQA  [  43  ]  highlights critical gaps in spatial, anatomic, and quantitative reasoning. Unlike complex deductive benchmarks, BLINK

[  18  ]  shows perceptually demanding tasks remain challenging for MLMs.  MedBLINK  extends this line of work by targeting foundational perceptual skills that are easy for clinicians but consistently missed by current models. Rather than emphasizing complex reasoning, it probes the basic visual competencies essential for earning trust in clinical deployment—filling a critical gap in how model trustworthiness is currently evaluated.

3

MedBLINK  Benchmark

Figure 2 :

MedBLINK  characterization. The benchmarks contain 8 tasks, ranging from tasks like enhancement detection and depth estimation to anatomical understanding tasks like morphology quantization.

Clinical image interpretation relies on both perceptual and conceptual reasoning. Perception enables clinicians to extract key visual features before engaging in more complex dia