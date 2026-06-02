[2508.17767] ISACL: Internal State Analyzer for Copyrighted Training Data Leakage

ISACL: Internal State Analyzer for Copyrighted Training Data Leakage

Guangwei Zhang 1  ,
 Qisheng Su 2  ,
 Jiateng Liu 3  ,
 Cheng Qian 3  ,

Yanzhou Pan 4  ,
 Yanjie Fu 5  ,
 Denghui Zhang 6

1 City University of Hong Kong,
 2 Microsoft,

3 University of Illinois Urbana-Champaign,
 4 Google LLC,

5 Arizona State University,
 6 Stevens Institute of Technology

dzhang42@stevens.edu

Corresponding author.

Abstract

Large Language Models (LLMs) have revolutionized Natural Language Processing (NLP) but pose risks of inadvertently exposing copyrighted or proprietary data, especially when such data is used for training but not intended for distribution. Traditional methods address these leaks only after content is generated, which can lead to the exposure of sensitive information. This study introduces a proactive approach: examining LLMs’ internal states before text generation to detect potential leaks. By using a curated dataset of copyrighted materials, we trained a neural network classifier to identify risks, allowing for early intervention by stopping the generation process or altering outputs to prevent disclosure. Integrated with a Retrieval-Augmented Generation (RAG) system, this framework ensures adherence to copyright and licensing requirements while enhancing data privacy and ethical standards. Our results show that analyzing internal states effectively mitigates the risk of copyrighted data leakage, offering a scalable solution that fits smoothly into AI workflows, ensuring compliance with copyright regulations while maintaining high-quality text generation. The implementation is available on GitHub.  1

1  1  \url https://github.com/changhu73/Internal_states_leakage

◄

Feeling lucky?

Conversion report

Report an issue

View&nbsp;original on&nbsp;arXiv  ►

Copyright

Privacy Policy

Generated  on Fri Sep  5 16:08:56 2025 by

L a T e

XML