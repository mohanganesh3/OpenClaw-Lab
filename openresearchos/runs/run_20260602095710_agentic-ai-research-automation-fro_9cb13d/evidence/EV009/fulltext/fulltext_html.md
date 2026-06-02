[2505.00174] Real-World Gaps in AI Governance Research AI safety and reliability in everyday deployments

Real-World Gaps in AI Governance Research

AI safety and reliability in everyday deployments

Ilan Strauss

AI Disclosures Project, Social Science Research Council

Institute for Innovation and Public Purpose, University College London

Isobel Moure

AI Disclosures Project, Social Science Research Council

Tim O’Reilly

AI Disclosures Project, Social Science Research Council

O’Reilly Media

Sruly Rosenblat

We gratefully acknowledge funding support from The Alfred P. Sloan Foundation, the Omidyar Network, and the Patrick J. McGovern Foundation.
 Contact:  istrauss@ssrc.org .
 Code and data:  https://github.com/AI-Disclosures-Project/The-State-of-AI-Governance-Research

AI Disclosures Project, Social Science Research Council

Abstract

Drawing on 1,178 safety and reliability papers from 9,439 generative AI papers (January 2020 – March 2025), we compare research outputs of leading  AI companies  (Anthropic, Google DeepMind, Meta, Microsoft, and OpenAI) and  AI universities  (CMU, MIT, NYU, Stanford, UC Berkeley, and University of Washington). We find that corporate AI research increasingly concentrates on pre-deployment areas—model alignment and testing &amp; evaluation—while attention to deployment-stage issues such as model bias has waned. Significant research gaps exist in high-risk deployment domains, including healthcare, finance, misinformation, persuasive and addictive features, hallucinations, and copyright. Without improved observability into deployed AI, growing corporate concentration could deepen knowledge deficits. We recommend expanding external researcher access to deployment data and systematic observability of in-market AI behaviors.

Keywords : AI research; alignment; interpretability; commercialization risks; cloud providers; model developers.

Contents

1  Introduction

2  Motivation, Data, and Methods

2.1  Pre- versus post-deployment research

2.2  Why commercial incentives may drive research gaps

2.3  Data access challenges for independent research

2.4  Data collection and sample construction

3  Findings

3.1  Corporate vs. academic generative AI research

3.2  Post-deployment research gaps

4  Policy Discussion

5  Conclusion

6  Appendix

6.1  Additional Analysis

6.2  Research Dataset Construction

6.3  Classification Process: Categories

6.4  Selective Behavioral Impact Papers

1  Introduction

As generative AI becomes integrated into every facet of our work and social lives, there is an  urgent need to understand the performance and impact of AI products in such commercial “post-deployment” contexts

[ 16 ,  81 ] . Yet corporate research, now increasingly dominant, focuses on AI risks in  pre-deployment  laboratory settings through model alignment and testing (Figure

4  ).  1

1

1  AI alignment covers ‘post-training’ interventions, fine-tuning &amp; reinforcement learning from human and AI feedback.

User, system, and society-level impacts remain neglected.  2

2

2  AI companies’ do revise their models based based on red-teaming and user experience feedback  [ 41 ] .

Unless AI governance research follows AI systems into the real world, areas currently considered highest risk by AI companies themselves will remain underexplored . These include model persuasiveness, emergent behaviors from reinforcement learning exploitation (‘reward hacking’), and misinformation  [ 64 ,  80 ,  40 ] . De-prioritization of  research  into such areas both impedes developing industry-wide best  practices  for deployed AI systems and confines essential AI safeguards to siloed corporate efforts, limiting knowledge diffusion and public accountability.

Growing corporate concentration in AI research risks exacerbating these deficiencies . The commercial ‘AI race’ prioritizes an engaging user experience over broader societal impacts  [ 35 ] . Evidence of this shift includes corporate research teams becoming tightly integrated with product teams  [ 85 ] , research findings increasingly kept internal  [ 34 ]  (Figure

3  ), and alignment research overlooking dangerous side-effects, such as sycophancy and degraded answer quality  [ 3 ,  69 ,  24 ,  92 ] .

Method

We analyze AI governance research using a dataset of 1,178 safety and reliability papers from 9,439 generative AI papers  written by five dominant  AI companies  (Anthropic, Google DeepMind, Meta, Microsoft, and OpenAI), and six prominent  AI research universities  (Carnegie Mellon University (CMU), MIT, New York University (NYU), Stanford, UC Berkeley, and University of Washington) between January 2020 and March 2025. We call these two groups ‘Corporate AI’ and ‘Academic AI’, respectively. Our dataset combines generative AI research papers from Anthropic and OpenAI’s websites  [ 22 ]  with OpenAlex’s database. We define AI governance research as technical and applied safety and reliability research pre- and post-deployment. In conjunction with OpenAI’s o3-mini, we determine if papers are “safety &amp; reliability” research, and then classify them into one of eight sub-categories. We also conduct separate ‘regex’ key word searches in paper abstracts and titles for high-risk deployment domains (medical, finance, commercial, &amp; copyright) and capabilities (misinformation, disclosures, behavioral, &amp; accuracy).

Core Findings

1.

AI governance research is highly concentrated within a handful of uniquely resourced and integrated AI tech companies, with a disproportionately influential research impact . Anthropic, OpenAI, and Google DeepMind each have far more citations for their AI safety &amp; reliability work than any of the major U.S. academic institutions we track. Google DeepMind has more citations for its general generative AI research than the top four AI academic institutions combined.

2.

As leading AI companies race to commercialize powerful AI systems, their research priorities are increasingly shaped by business incentives rather than by comprehensive risk assessments and mitigations . Most of the corporate governance research we review focuses on model performance divorced from its applications. Ethics &amp; bias research – needed to understand systematic, unjustified differences in LLM behavior or outputs – now only receives attention from academic researchers.

3.

Corporate AI labs severely neglect deployment-stage behavioral and business risks . Only 4% of Corporate AI papers (6% Academic AI) tackle high-stakes areas like persuasion, misinformation, medical &amp; financial contexts, disclosures, or core business liabilities (IP violations, coding errors, hallucinations) – despite emerging lawsuits showing these risks to already be material.

Policy Considerations

To guard against commercialization-driven risks, third-party researchers (and auditors) need data on AI systems operating in real-world environments . Commercial incentives drive innovation but also foster corporate risk-taking, potentially lowering safeguards when they conflict with profit-maximizing business models  [ 35 ,  26 ] .  Post-deployment monitoring research is therefore publicly vital but currently limited to piecemeal AI incident databases

[ 49 ,  86 ,  53 ] ,  old or overly aggregated user-LLM chat data

[ 76 ,  94 ] , and public testing of models. Real-world visibility into the effects of AI systems is negligible.

Structured access is needed into deployed AI systems’ telemetry data and artifacts to systematically analyze real-world risks and harms. 
Monitoring and evaluation of LLMs in real-world environments is now essential to quality assurance (QA), as in ‘LLMOps’  [ 9 ] . But the data used for this is the preserve of corporate  practice , resulting in society losing essential insight into AI’s ongoing risks and harms. Disclosure of AI system  telemetry data  (logs, traces, &amp; business metrics) and LLM model  data artifacts  (e.g., training/fine-tuning datasets) may expose co