[2604.16060] Chain-of-Thought Degrades Visual Spatial Reasoning Capabilities of Multimodal LLMs

Chain-of-Thought Degrades Visual Spatial Reasoning Capabilities of Multimodal LLMs

Sai Srinivas Kancheti  1

1  footnotemark:

1

2

2  footnotemark:

2

Aditya Sanjiv Kanade  1

1  footnotemark:

1

IIT, Hyderabad

Microsoft Research India

cs21resch01004@iith.ac.in

kanade850@gmail.com

Vineeth N. Balasubramanian

Tanuja Ganu

Microsoft Research India

Microsoft Research India

vineeth.nb@microsoft.com

tanuja.ganu@microsoft.com

Abstract

Multimodal Reasoning Models (MRMs) leveraging Chain-of-Thought (CoT) based thinking have revolutionized mathematical and logical problem-solving. However, we show that this paradigm struggles with generalized spatial intelligence. We perform a comprehensive evaluation of seventeen models across thirteen spatial benchmarks and identify a critical gap: CoT prompting consistently degrades performance in visual spatial reasoning.
Furthermore, through a novel  No-Image++  ablation, we demonstrate that MRMs and CoT prompted MLMs suffer from severe shortcut learning, and hallucinate visual details from textual priors even when the image is absent. These findings challenge the efficacy of text-only CoT for spatial tasks and underscore the need for vision-centric reasoning paradigms.

Chain-of-Thought Degrades Visual Spatial Reasoning Capabilities of Multimodal LLMs

Sai Srinivas Kancheti  1

1

1  Equal contribution.

2

2

2  Work done while at Microsoft Research India.

Aditya Sanjiv Kanade  1

1

1  Equal contribution.

IIT, Hyderabad

Microsoft Research India

cs21resch01004@iith.ac.in

kanade850@gmail.com

Vineeth N. Balasubramanian

Tanuja Ganu

Microsoft Research India

Microsoft Research India

vineeth.nb@microsoft.com

tanuja.ganu@microsoft.com

1  Introduction

The emergence of "System 2" Multimodal Reasoning Models (MRMs) — models post-trained via SFT and RL to generate step-by-step reasoning — has driven remarkable progress in mathematical and logical domains. By leveraging Reinforcement Learning (RL)

TULU3 ;  DeepSeekAI2025DeepSeekR1IR

and long Chain-of-Thought (CoT)

cot1 ;  cot2

inference, MRMs demonstrate the ability to self-correct and reason through complex problems. Separately,  CoT prompting  is a general technique that instructs any Multimodal Language Model (MLM) to think step-by-step before answering. However, a fundamental question remains:  does this text-centric reasoning paradigm translate to spatial intelligence?  Spatial reasoning requires grounding, geometric intuition, and precise localization, which are skills that may not easily arise from verbose, text-based reasoning

cambrian1 ;  MMVP  .

In this work, we conduct a comprehensive evaluation of seventeen models, including nine state-of-the-art open-source MRMs (e.g., GThinker, Vision-R1, ViGoRL, Qwen3-VL) and eight diverse backbone MLMs. We benchmark these models across thirteen datasets covering static 2D relations, 3D geometry, and dynamic/temporal understanding. To isolate the impact of CoT reasoning, we standardize our evaluation using a uniform evaluation and scoring policy. Our findings reveal that contrary to trends in other domains, CoT prompting degrades performance in visual spatial tasks. Our contributions are as follows: (i) We show that MRMs consistently underperform their own backbone on generalized spatial benchmarks. In our experiments, 7 out of 8 reasoning models failed to surpass the backbone they were distilled from. (ii) We demonstrate in Figure

1

that CoT prompting lowers accuracy by an average of 3% across a diverse range of MLMs. (iii) Through a novel  No-Image++  ablation, we show that MRMs suffer from severe shortcut learning. When presented with a blank image and a “Cannot determine” option, reasoning models continue to hallucinate visual details and confidently select incorrect answers based solely on textual priors.

These results suggest that simply scaling text-based reasoning is insufficient for robust spatial intelligence, highlighting the need for vision-centric training paradigms.

Model

CoT

Non-CoT

GThinker

62.52

39.38 (

-23.14  %

\textbf{-23.14}\%

)

R1-Ov

46.88

47.84 (

+0.96  %

\textbf{+0.96}\%

)

ViGoRL

60.68

62.52 (

+1.84  %

\textbf{+1.84}\%

)

VL-Re.

60.99

62.18 (

+1.19  %

\textbf{+1.19}\%

)

Vision-G1

63.26

62.85 (

-0.41  %

\textbf{-0.41}\%

)

Vision-R1

58.86

59.6 (

+0.74  %

\textbf{+0.74}\%

)

TreeVGR

61.11

62.6 (

+1.49  %

\textbf{+1.49}\%

)

ThinkLite

62.61

62.74 (

+0.13  %

\textbf{+0.13}\%

)

Figure 1:  (Left) CoT vs Non-CoT performance of open-source MRMs. (Right) Bar chart showing the average accuracy of various families of MLMs over 13 benchmark datasets. For each model, the left bar shows the accuracy achieved by CoT prompting, and the right bar shows for base prompt (non-CoT). We observe CoT prompting drops performance over a wide range of backbones and model scales, including Qwen3-VL-8B-Thinking

qwen3vl  , a model with explicitly enhanced spatial perception.

Models

3DSRBench

BLINK

CV-Bench

MindCube

MMSIBench

MMVP

2D

3D

Qwen2.5-VL-7B  cot

57.11  0.39

\textbf{57.11}_{0.39}

53.44  0.30

53.44_{0.30}

75.92  0.03

75.92_{0.03}

76.09  0.31

76.09_{0.31}

30.83  0.25

30.83_{0.25}

27.47  0.21

\textbf{27.47}_{0.21}

72.44  0.68

72.44_{0.68}

Qwen2.5-VL-7B

55.38  0.06

55.38_{0.06}

56.04  0.03

\textbf{56.04}_{0.03}

77.17  ¯

0.03

\underline{77.17}_{0.03}

83.78  ¯

0.04

\underline{83.78}_{0.04}

35.11  0.18

35.11_{0.18}

26.87  0.09

26.87_{0.09}

75.78  ¯

0.32

\underline{75.78}_{0.32}

\rowcolor light-gray
GThinker-7B

56.58  0.20

56.58_{0.20}

54.76  0.17

54.76_{0.17}

77.40  0.06

\textbf{77.40}_{0.06}

82.95  0.04

82.95_{0.04}

40.16  ¯

0.32

\underline{40.16}_{0.32}

27.33  ¯

0.31

\underline{27.33}_{0.31}

73.78  0.42

73.78_{0.42}

R1-Onevision-7B

48.52  0.20

48.52_{0.20}

43.27  0.50

43.27_{0.50}

53.31  0.04

53.31_{0.04}

58.00  0.42

58.00_{0.42}

27.09  0.42

27.09_{0.42}

13.30  0.30

13.30_{0.30}

56.16  0.16

56.16_{0.16}

\rowcolor light-gray
ViGoRL-7B-Spatial

55.84  0.20

55.84_{0.20}

52.51  0.26

52.51_{0.26}

76.59  0.27

76.59_{0.27}

86.14  0.10

\textbf{86.14}_{0.10}

39.36  0.16

39.36_{0.16}

25.87  0.17

25.87_{0.17}

73.22  0.42

73.22_{0.42}

VL-Rethinker-7B

56.99  ¯

0.11

\underline{56.99}_{0.11}

54.60  ¯

0.23

\underline{54.60}_{0.23}

76.06  0.12

76.06_{0.12}

80.75  0.14

80.75_{0.14}

37.81  0.27

37.81_{0.27}

26.90  0.08

26.90_{0.08}

75.89  0.16

75.89_{0.16}

\rowcolor light-gray
Vision-G1

55.91  0.01

55.91_{0.01}

54.60  ¯

0.08

\underline{54.60}_{0.08}

76.70  0.15

76.70_{0.15}

83.75  ¯

0.14

\underline{83.75}_{0.14}

38.10  0.31

38.10_{0.31}

26.07  0.12

26.07_{0.12}

76.56  0.16

\textbf{76.56}_{0.16}

Vision-R1-7B

55.01  0.20

55.01_{0.20}

46.47  0.66

46.47_{0.66}

71.58  0.12

71.58_{0.12}

75.83  0.31

75.83_{0.31}

36.95  0.59

36.95_{0.59}

22.90  0.36

22.90_{0.36}

72.22  0.32

72.22_{0.32}

\rowcolor light-gray
TreeVGR-7B

51.53  0.03

51.53_{0.03}

53.16  0.25

53.16_{0.25}

76.24  0.09

76.24_{0.09}

75.17  0.13

75.17_{0.13}

44.25  0.66

\textbf{44.25}_{0.66}

27.17  0.33

27.17_{0.33}

71.33  0.27

71.33_{0.27}

ThinkLite-7B

57.26  0.13

57.26_{0.13}

57.13  0.08

57.13_{0.08}

76.89  0.18

76.89_{0.18}

80.44  0.16

80.44_{0.16}

30.13  0.23

30.13_{0.23}

27.97  0.56

27.97_{0.56}

73.56  0.16

73.56_{0.16}

Models

OmniSpatial

RealWorldQA

SAT

SpatialBench

VSR

V*Bench

Avg.

Qwen2.5-VL-7B  cot

40.40  0.80

40.40_{0.80}

63.05  0.27

63.05_{0.27}

59.22  0.57

59.22_{0.57}

61.75  0.70

61.75_{0.70}

81.83  0.35

81.83_{0.35}

76.27  0.25

76.27_{0.25}

59.68

Qwen2.5-VL-7B

45.23  0.11

45.23_{0.11}

69.02  ¯

0.00

\underline{69.02}_{0.00}

63.11  ¯

0.16

\underline{63.11}_{0.16}

62.87  ¯

0.00

\underline{62.87}_{0.00}

85.38  ¯

0.04

\underline{85.38}_{0.04}

79.06  0.00

79.06_{0.00}

62.68

\rowcolor light-gray
GThinker-7B

47