[2509.25137] The Era of Real-World Human Interaction: RL from User Conversations

1]FAIR at Meta
2]Johns Hopkins University
 \contribution [†]Joint second author

The Era of Real-World Human Interaction:

RL from User Conversations

Chuanyang Jin

Jing Xu

Bo Liu

Leitian Tao

Olga Golovneva

Tianmin Shu

Wenting Zhao

Xian Li

Jason Weston

[

[

( October 6, 2025 )

Abstract

We posit that to achieve continual model improvement and multifaceted alignment, future models must learn from natural human interaction. Current conversational models are aligned using pre-annotated, expert-generated human feedback. In this work, we introduce Reinforcement Learning from Human Interaction (RLHI), a paradigm that learns directly from in-the-wild user conversations. We develop two complementary methods: (1)  RLHI with User-Guided Rewrites , which revises unsatisfactory model outputs based on users’ natural-language follow-up responses, (2)  RLHI with User-Based Rewards , which learns via a reward model conditioned on knowledge of the user’s long-term interaction history (termed persona). Together, these methods link long-term user personas to turn-level preferences via persona-conditioned preference optimization. Trained on conversations derived from WildChat, both RLHI variants outperform strong baselines in personalization and instruction-following, and similar feedback enhances performance on reasoning benchmarks. These results suggest organic human interaction offers scalable, effective supervision for personalized alignment.

\correspondence

Chuanyang Jin at  cjin33@jhu.edu , Jason Weston at  jase@meta.com

Figure 1 :

From annotated feedback to the  era of real-world human interaction .

Left:  Traditional alignment relies on expert-curated annotations of ranked responses or labels, providing static, out-of-distribution supervision.  Right:  In-the-wild conversations reveal users’ long-term histories, dynamic demands, and diverse signals, enabling personalized, contextual, and continual learning.

1  Introduction

Today, language model post-training primarily depends on static corpora of expert-annotated data: verifiable questions, fixed demonstrations, and rankings or ratings collected outside of natural conversational contexts. While these datasets are effective for instilling general capabilities, they reflect the opinions and heuristics of annotators in unnatural scenarios rather than the  authentic, diverse long-term goals and preferences of real users ; they capture static, context-free judgments instead of  evolving, situational demands ; and they scale with labeling budgets rather than with actual usage and diversity of organic users, as is illustrated on the left side of Figure

1  .

In contrast, humans learn and improve through continual experience by interacting with their environment and other actors, receiving feedback, and adjusting behavior over time  (Tomasello et al.,  2005 ) . Likewise, a rich and organic source of supervision for language models already exists in the wild:  human interaction —the ongoing, natural exchanges between models and real users. As is shown on the right side of Figure

1  , such organic interactions reveal hidden user preferences from long-term histories and dynamic, context-dependent demands, as people reveal their priorities and concerns not through annotation formats, but by discussing what matters to them, revising or re-attempting questions, explicitly or implicitly approving or critiquing model outputs, following up, or switching goals mid-dialogue. Because they arise directly from model outputs in authentic usage contexts, such interactions provide a rich signal for learning personalized and adaptable behavior, paving the way toward personal superintelligence. While this source of supervision has historically been hard to extract, resulting in resorting to collecting static training data instead, the power of modern language models now gives us a greater ability to extract these signals.

To achieve this vision, we introduce RLHI, a paradigm that learns directly from in-the-wild conversations through two complementary methods:
(1)  RLHI with User-Guided Rewrites  (§  2.3  ), which revises unsatisfactory model outputs based on users’ natural-language follow-ups, and pairs the rewrites with the originals for preference learning; and
(2)  RLHI with User-Based Rewards  (§  2.4  ), which ranks candidate responses using a reward model conditioned on user personas derived from long-term histories to generate preference pairs. Together, these methods link long-term user personas to turn-level preferences via persona-conditioned preference optimization.

We evaluate RLHI in three settings. (i)  User-based evaluation  with our  WildChat UserEval : both RLHI variants outperform strong baselines in personalization and instruction-following, and a human study corroborates these trends. (ii)  Standard instruction-following benchmarks :  User-Based Rewards  attains a 77.9% length-controlled win rate on AlpacaEval 2.0, surpassing all baselines. (iii)  Reasoning :  User-Guided Rewrites  raises average accuracy from 26.5 to 31.8 across four benchmarks. Our ablation studies further show that RLHI benefits from user guidance and interaction diversity, that reinforcement learning outperforms supervised finetuning, and that quality filtering is essential for effectively leveraging noisy human interaction data.

2  RLHI: Reinforcement Learning from Human Interaction

2.1  The Era of Real-World Human Interaction

Artificial intelligence (AI) has progressed rapidly in recent years through large-scale pretraining and fine-tuning with human examples and preferences. Yet this trajectory is slowing: high-quality data is running out, and imitation alone cannot push systems beyond existing human knowledge. Recent proposals call for an  era of experience

(Silver and Sutton,  2025 ) , in which AI systems advance by continually learning from their own interactions with the world. Since these systems ultimately exist to assist humans, interaction with users becomes a natural and essential dimension of this shift. The  era of real-world human interaction  thus forms a core pillar of the era of experience, providing both the raw data and personalization signals necessary for adaptive, human-centered intelligence.

We define learning from human interaction as the process of improving AI models through natural, continual exchanges with real users. Such interactions may involve messages, actions, requests, or demonstrations provided in direct response to the model’s outputs. These exchanges not only reveal user goals and preferences but also create an evolving feedback loop that enables systems to refine their behavior over time. To truly benefit from human interaction, AI needs to go beyond coarse binary labels to absorb knowledge, preferences, reasoning skills, perceptual cues, cooperative strategies, and social norms, learning deeper forms of intelligence through interaction.

Compared with other training data sources, human interaction is distinguished by three key properties:

1.

Contextual grounding  — arises within the flow of ongoing tasks or conversations, directly tied to the user’s situational needs and the model’s prior outputs, while being shaped by personalized knowledge of the user’s profile, history, and preferences;

2.

Evolving distribution  — reflects goals that shift, environments that change, and preferences that adapt over time, thereby providing supervision that is temporally relevant and aligned with the real distribution of human needs and priorities; and

3.

Diverse supervision signals  — appears in both explicit high-bandwidth signals beyond scalar rewards (e.g., corrections or clarifications) and implicit cues (e.g., disengagement or frustration), and may include style and role assignments, emotional tone, or even adversarial inputs such as jailbreak attempts, which require careful handling, but also offer v