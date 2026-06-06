[2305.16291] Voyager: An Open-Ended Embodied Agent with Large Language Models

Voyager : An Open-Ended Embodied Agent
 with Large Language Models

Guanzhi Wang

1 2  ​

\faIcon

[regular]envelope

12

\faIcon

[regular]envelope

{}^{1\,2\,\text{\faIcon[regular]{envelope}}}

, Yuqi Xie 3 , Yunfan Jiang  4∗  , Ajay Mandlekar  1∗  ,

Chaowei Xiao  1 5  , Yuke Zhu  1 3  , Linxi “Jim” Fan

1  †

\faIcon

[regular]envelope

†  1

\faIcon

[regular]envelope

{}^{1\dagger\,\text{\faIcon[regular]{envelope}}}

, Anima Anandkumar  1 2†

1

NVIDIA,  2 Caltech,  3 UT Austin,  4 Stanford,  5 UW Madison

∗ Equal contribution

† Equal advising

\faIcon

[regular]envelope

\faIcon

[regular]envelope

{}^{\text{\faIcon[regular]{envelope}}}

Corresponding authors

https://voyager.minedojo.org

Abstract

We introduce  Voyager , the first LLM-powered embodied lifelong learning agent in Minecraft that continuously explores the world, acquires diverse skills, and makes novel discoveries without human intervention.  Voyager  consists of three key components: 1) an automatic curriculum that maximizes exploration, 2) an ever-growing skill library of executable code for storing and retrieving complex behaviors, and 3) a new iterative prompting mechanism that incorporates environment feedback, execution errors, and self-verification for program improvement.
 Voyager  interacts with GPT-4 via blackbox queries, which bypasses the need for model parameter fine-tuning. The skills developed by  Voyager  are temporally extended, interpretable, and compositional, which compounds the agent’s abilities rapidly and alleviates catastrophic forgetting.
Empirically,  Voyager  shows strong in-context lifelong learning capability and exhibits exceptional proficiency in playing Minecraft.
It obtains

3.3  ×

3.3\times

more unique items, travels

2.3  ×

2.3\times

longer distances, and unlocks key tech tree milestones up to

15.3  ×

15.3\times

faster than prior SOTA.  Voyager  is able to utilize the learned skill library in a new Minecraft world to solve novel tasks from scratch, while other techniques struggle to generalize.

Figure 1 :

Voyager

discovers new Minecraft items and skills continually by self-driven exploration, significantly outperforming the baselines. X-axis denotes the number of prompting iterations.

1  Introduction

Building generally capable embodied agents that continuously explore, plan, and develop new skills in open-ended worlds is a grand challenge for the AI community  [ 1 ,  2 ,  3 ,  4 ,  5 ] .
Classical approaches employ reinforcement learning (RL)  [ 6 ,  7 ]  and imitation learning  [ 8 ,  9 ,  10 ]  that operate on primitive actions, which could be challenging for systematic exploration  [ 11 ,  12 ,  13 ,  14 ,  15 ] , interpretability  [ 16 ,  17 ,  18 ] , and generalization  [ 19 ,  20 ,  21 ] .
Recent advances in large language model (LLM) based agents harness the world knowledge encapsulated in pre-trained LLMs to generate consistent action plans or executable policies  [ 16 ,  22 ,  19 ] . They are applied to embodied tasks like games and robotics  [ 23 ,  24 ,  25 ,  26 ,  27 ] , as well as NLP tasks without embodiment  [ 28 ,  29 ,  30 ] .
However, these agents are not lifelong learners that can progressively acquire, update, accumulate, and transfer knowledge over extended time spans  [ 31 ,  32 ] .

Let us consider Minecraft as an example. Unlike most other games studied in AI  [ 33 ,  34 ,  10 ] , Minecraft does not impose a predefined end goal or a fixed storyline but rather provides a unique playground with endless possibilities  [ 23 ] . Minecraft requires players to explore vast, procedurally generated 3D terrains and unlock a tech tree using gathered resources. Human players typically start by learning the basics, such as mining wood and cooking food, before advancing to more complex tasks like combating monsters and crafting diamond tools.
We argue that an effective lifelong learning agent should have similar capabilities as human players: (1)  propose suitable tasks  based on its current skill level and world state, e.g., learn to harvest sand and cactus before iron if it finds itself in a desert rather than a forest; (2)  refine skills  based on environmental feedback and  commit mastered skills to memory  for future reuse in similar situations (e.g. fighting zombies is similar to fighting spiders); (3)  continually explore the world  and seek out new tasks in a self-driven manner.

Figure 2 :

Voyager

consists of three key components: an automatic curriculum for open-ended exploration, a skill library for increasingly complex behaviors, and an iterative prompting mechanism that uses code as action space.

Towards these goals, we introduce  Voyager , the first  LLM-powered embodied lifelong learning agent  to drive exploration, master a wide range of skills, and make new discoveries continually without human intervention in Minecraft.
 Voyager  is made possible through three key modules (Fig.

2  ): 1) an  automatic curriculum  that maximizes exploration; 2) a  skill library  for storing and retrieving complex behaviors; and 3) a new  iterative prompting mechanism  that generates executable code for embodied control.
We opt to use code as the action space instead of low-level motor commands because programs can naturally represent temporally extended and compositional actions  [ 16 ,  22 ] , which are essential for many long-horizon tasks in Minecraft.
 Voyager  interacts with a blackbox LLM (GPT-4  [ 35 ] ) through prompting and in-context learning  [ 36 ,  37 ,  38 ] . Our approach bypasses the need for model parameter access and explicit gradient-based training or finetuning.

More specifically,  Voyager  attempts to solve progressively harder tasks proposed by the  automatic curriculum , which takes into account the exploration progress and the agent’s state. The curriculum is generated by GPT-4 based on the overarching goal of “discovering as many diverse things as possible”. This approach can be perceived as an in-context form of  novelty search

[ 39 ,  40 ] .
 Voyager  incrementally builds a  skill library  by storing the action programs that help solve a task successfully. Each program is indexed by the embedding of its description, which can be retrieved in similar situations in the future. Complex skills can be synthesized by  composing  simpler programs, which compounds  Voyager ’s capabilities rapidly over time and alleviates catastrophic forgetting in other continual learning methods  [ 31 ,  32 ] .

However, LLMs struggle to produce the correct action code consistently in one shot  [ 41 ] . To address this challenge, we propose an  iterative prompting mechanism  that: (1) executes the generated program to obtain observations from the Minecraft simulation (such as inventory listing and nearby creatures) and error trace from the code interpreter (if any); (2) incorporates the feedback into GPT-4’s prompt for another round of code refinement; and (3) repeats the process until a self-verification module confirms the task completion, at which point we commit the program to the skill library (e.g.,  craftStoneShovel()  and  combatZombieWithSword() ) and query the automatic curriculum for the next milestone (Fig.

2  ).

Empirically,  Voyager  demonstrates strong  in-context lifelong learning  capabilities. It can construct an ever-growing skill library of action programs that are reusable, interpretable, and generalizable to novel tasks.
We evaluate  Voyager  systematically against other LLM-based agent techniques (e.g., ReAct  [ 29 ] , Reflexion  [ 30 ] , AutoGPT  [ 28 ] ) in MineDojo  [ 23 ] , an open-source Minecraft AI framework.
 Voyager  outperforms prior SOTA by obtaining

3.3  ×

3.3\times

more unique items, unlocking key tech tree milestones up to

15.3  ×

15.3\times

faster, and traversing

2.3  ×

2.3\times

longer distances. We further demonstrate that  Voyager  is able to utilize the learned skill library in a