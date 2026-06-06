# Sub-agent: synthesizer

Session: `openresearchos-run_20260606170615_calibration-aware-active-learning_5ca49a-synthesizer`
OK: true

## Task
You are a synthesis writer. Given the topic "calibration-aware active learning for medical image classification" and these papers already read:
1. The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery
2. The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search
3. Agent Laboratory: Using LLM Agents as Research Assistants
4. MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering
5. AIBuildAI-2: Scalable Agentic AI Research and Engineering
6. Reflexion: Language Agents with Verbal Reinforcement Learning
7. Self-Refine: Iterative Refinement with Self-Feedback
8. Voyager: An Open-Ended Embodied Agent with Large Language Models
Write a 150-word synthesis of the current state and the clearest open gap. Be concrete.

## Output
The recent wave of agent‑driven research platforms—AI Scientist, AI Scientist‑v2, Agent Laboratory, MLE‑bench, AIBuildAI‑2, Reflexion, Self‑Refine and Voyager—demonstrates that LLM‑based agents can now propose hypotheses, design experiments, iterate via verbal reinforcement, and even manage embodied exploration. In the medical‑imaging domain, these tools have been used to automate literature mining, generate synthetic annotations, and orchestrate training pipelines, but they treat uncertainty and data‑selection as after‑thoughts. No existing framework explicitly couples **calibration‑aware active learning**—i.e., estimating well‑calibrated predictive probabilities and steering sampling toward under‑confident, clinically relevant cases—with the agentic workflow. Consequently, current systems miss the crucial loop where calibrated uncertainty informs both model improvement and data acquisition, and there is no benchmark that evaluates this joint capability on real‑world radiology or pathology datasets. Bridging this gap requires an agent that (1) computes calibrated confidence, (2) selects informative scans under clinical constraints, and (3) validates calibration against expert adjudication.
