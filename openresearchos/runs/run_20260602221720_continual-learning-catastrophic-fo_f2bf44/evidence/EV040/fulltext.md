[2505.13308] Seek in the Dark: Reasoning via Test-Time Instance-Level Policy Gradient in Latent Space

Seek in the Dark : Reasoning via Test-Time Instance-Level Policy Gradient in Latent Space

marginparsep has been altered.
 topmargin has been altered.
 marginparpush has been altered.

The page layout violates the ICML style. 
Please do not change the page layout, or include packages like geometry,
savetrees, or fullpage, which change it for you.
We’re not able to reliably undo arbitrary changes to the style. Please remove
the offending package(s), or layout-changing commands and try again.

Seek in the Dark  : Reasoning via Test-Time Instance-Level Policy Gradient in Latent Space

Hengli Li

1  ,  2

⁣  ∗

{}^{\,1,2\,*}

, Chenxi Li

2  ,  3

⁣  ∗

{}^{\,2,3\,*}

, Tong Wu

2  , Xuekai Zhu

2,4  , Yuxuan Wang

2  , Zhaoxin Yu

5  , Eric Hanchen Jiang

6  , Song-Chun Zhu

1,2,3  , Zixia Jia

2  , Ying Nian Wu

6

{}^{\,6\,}

✉

and Zilong Zheng

2

{}^{\,2\,}

✉

1

{}^{1\,}

Institute for Artificial Intelligence, Peking University

2

{}^{2\,}

NLCo Lab, Beijing Institute for General Artificial Intelligence

3

{}^{3\,}

Department of Automation, Tsinghua University

4

{}^{4\,}

Shanghai Jiao Tong University

5

{}^{5\,}

Institute of Automation, Chinese Academy of Sciences

6

{}^{6\,}

University of California, Los Angeles

lihengli@stu.pku.edu.cn, lichenxi23@mails.tsinghua.edu.cn, ywu@stat.ucla.edu, zlzheng@bigai.ai

†

†  footnotetext:

*  Equal Contributions.  ✉

Corresponding author(s): Ying Nian Wu, Zilong Zheng.

Reasoning ability, a core component of human intelligence, continues to pose a significant challenge for Large Language Models (LLMs) in the pursuit of AGI. Although model performance has improved under the training scaling law, significant challenges remain, particularly with respect to training algorithms—such as catastrophic forgetting—and the limited availability of novel training data. As an alternative, test-time scaling enhances reasoning performance by increasing test-time computation without parameter updating. Unlike prior methods in this paradigm focused on token space, we propose leveraging latent space for more effective reasoning and better adherence to the test-time scaling law. We introduce  LatentSeek , a novel framework that enhances LLM reasoning through  Test-Time Instance-level Adaptation (TTIA) within the model’s latent space . Specifically,  LatentSeek  leverages policy gradient to iteratively update latent representations, guided by self-generated reward signals.  LatentSeek  is evaluated on a range of reasoning benchmarks, including GSM8K, MATH-500, and AIME2024, across multiple LLM architectures. Results show that  LatentSeek  consistently outperforms strong baselines, such as Chain-of-Thought prompting and fine-tuning-based methods. Furthermore, our analysis demonstrates that  LatentSeek  is highly efficient, typically converging within a few iterations for problems of average complexity, while also benefiting from additional iterations, thereby highlighting the potential of test-time scaling in the latent space. These findings position  LatentSeek  as a lightweight, scalable, and effective solution for enhancing the reasoning capabilities of LLMs.

\faGithub

Code

https://github.com/bigai-nlco/LatentSeek

\faGlobe

Project

https://bigai-nlco.github.io/LatentSeek/

Figure 1:  Comparison of  LatentSeek  with RL-based fine-tuning and Prompt Engineering. RL-based fine-tuning methods generally require iterative updates to model parameters guided by reward signals. Prompt engineering approaches depend heavily on manually designed prompts. In contrast,  LatentSeek  performs optimization within the latent space. Of note, the output of  LatentSeek  may be incoherent and semantically ungrounded (

§  \S

3.6  ).

1  Introduction

Large Language Models (LLMs) have demonstrated exceptional performance across a wide array of tasks, particularly in complex reasoning and deductive analysis  (Brown et al.,  2020 ; Chowdhery et al.,  2022 ; OpenAI,  2023 ; Zhao et al.,  2025 ) . Despite these advancements, LLMs still exhibit difficulties in reasoning, particularly in tasks that demand structured thinking and meticulous step-by-step analysis  (Wei et al.,  2022 ; Kojima et al.,  2022 ) . Common approaches to enhancing the reasoning capabilities of LLMs involve training model parameters on reasoning-centric datasets or providing reasoning-oriented feedback  (Ouyang et al.,  2022 ; Bai et al.,  2022 ; DeepSeek-AI,  2025 ) . However, training methodologies such as supervised fine-tuning, reinforcement learning, and test-time training  Sun et al. ( 2020 ); Hardt and Sun ( 2023 )  need to update model parameters, which incur substantial computational costs and present potential risks, including catastrophic forgetting of general competencies  (Luo et al.,  2025 ) . Moreover, the widely used reinforcement learning approach may reduce the model’s exploration capacity  Yue et al. ( 2025 )  and, in some cases, lead to the generation of overly verbose responses  Aggarwal and Welleck ( 2025 ); Wu et al. ( 2025 ) . In light of these challenges, we focus on an alternative paradigm,

Test-Time Instance-Level Adaptation

(  TTIA  ) , that  does not require parameter updating  and operates on a  per-instance basis  during the  testing phase .

To enhance TTIA performance in reasoning, recent advances  Deng et al. ( 2022 ); Hao et al. ( 2024 )  reveal that reasoning capabilities can be internalized within the  latent space  1

1

1

In this work, we take the convention

Kong et al.  (  2025  ) ; Hao et al.  (  2024  )

that treats the transformers’ output space ahead of the final language model (LM) head as latent space (

Figure

1

), and the vector in the space as latent representation (

Figure

1

); refer to

§  \S

2.2

for notations.

through fine-tuning. However, the training strategies adopted in these works tend to substantially modify the latent space and fail to fully leverage its semantic richness. As a result, their performance remains inferior to that of

Chain-of-Thought

(  CoT  ). Nevertheless, these studies provide evidence supporting the adequacy of semantic information encoded within latent representations, i.e., the hidden states corresponding to language tokens.

Motivated by these observations, we present the first attempt to perform seeking in the latent space by introducing  LatentSeek , a framework that significantly enhances instance-level reasoning at test time.  LatentSeek  introduces updated instance-specific latent representations that steer the pre-trained model’s reasoning process without modifying its parameters.
These latent representations act as a planning or control mechanism that guides the model toward better reasoning paths for each specific problem instance.
We optimize latent representations at test time using the policy gradient method  Williams ( 1992 )  to maximize reward (

§  \S

2.3  ).
Specifically, for each reasoning problem, we update the token-wise latent representations using guidance from the reward function, treating them as independent variables. In each iteration, the updated latent representations are decoded into tokens, which serve as inputs for computing the reward. Importantly, the reward function operates in a self-rewarding manner, relying solely on the model’s internal capabilities without incorporating any external information. The process continues until the reward exceeds a predefined threshold or the maximum number of iterations is reached.

Our innovative latent space TTIA method is simple yet surprisingly effective:
Notably,  LatentSeek  achieves an average improvement of  10.75%  over the

CoT

method on the GSM8K dataset,  3.93%  on MATH-500, and  4.73%  on AIME2024. Furthermore, when using LlaMA3.1-8B-Instruct as the backbone,  LatentSeek  surpasses prior arts including SimpleRL-Zoo  Williams ( 1992 )  ( +18.1% ) and Genius  Xu et al. ( 2025a )  ( +12.7% )