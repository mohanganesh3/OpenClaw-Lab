[2601.07790] Benchmarking Small Language Models and Small Reasoning Language Models on System Log Severity Classification

Benchmarking Small Language Models and Small Reasoning Language Models on System Log Severity Classification

Yahya Masri

George Mason University

Emily Ma

George Mason University

Zifu Wang

Harvard University

Joseph Rogers

George Mason University

Chaowei Yang

George Mason University

Email:  cyang3@gmu.edu

Abstract

System logs are crucial for monitoring and diagnosing modern computing infrastructure, but their scale and complexity require reliable and efficient automated interpretation. Since severity levels are predefined metadata in system log messages, having a model merely classify them offers limited standalone practical value, revealing little about its underlying ability to interpret system logs. We argue that severity classification is more informative when treated as a benchmark for probing runtime log comprehension rather than as an end task. Using real-world journalctl data from Linux production servers, we evaluate nine small language models (SLMs) and small reasoning language models (SRLMs) under zero-shot, few-shot, and retrieval-augmented generation (RAG) prompting. The results reveal strong stratification. Qwen3-4B achieves the highest accuracy at 95.64% with RAG, while Gemma3-1B improves from 20.25% under few-shot prompting to 85.28% with RAG. Notably, the tiny Qwen3-0.6B reaches 88.12% accuracy despite weak performance without retrieval. In contrast, several SRLMs, including Qwen3-1.7B and DeepSeek-R1-Distill-Qwen-1.5B, degrade substantially when paired with RAG. Efficiency measurements further separate models: most Gemma and Llama variants complete inference in under 1.2 seconds per log, whereas Phi-4-Mini-Reasoning exceeds 228 seconds per log while achieving

&lt;  &lt;

10% accuracy. These findings suggest that (1) architectural design, (2) training objectives, and (3) the ability to integrate retrieved context under strict output constraints jointly determine performance. By emphasizing small, deployable models, this benchmark aligns with real-time requirements of digital twin (DT) systems and shows that severity classification serves as a lens for evaluating model competence and real-time deployability, with implications for root cause analysis (RCA) and broader DT integration.

1  Introduction

System logs are vital components of modern computing infrastructure, capturing operational events, warnings, and performance information across distributed systems  [ 26 ,  21 ,  69 ,  3 ] . They play a critical role in diagnosing faults, monitoring system health, and supporting automated responses in large-scale environments such as data centers and digital twins (DTs)  [ 76 ,  17 ,  92 ] . However, as computing systems generate massive volumes of logs with complex, context-dependent language, manual interpretation has become infeasible  [ 84 ,  38 ,  43 ,  30 ,  59 ] . The continuous flow of log entries produced by servers far exceeds human review capacity, leading to delayed fault detection and increased mean time to resolve  [ 7 ,  72 ] .

Recent advances in combining large language models (LLMs) and retrieval-augmented generation (RAG) have demonstrated that external context can significantly improve model reliability on tasks that require domain awareness and factual grounding. Rather than relying solely on parameters learned during training, RAG allows models to incorporate relevant evidence at inference time by querying a knowledge database. This approach has shown strong benefits in knowledge-intensive applications such as question answering, structured information extraction, and technical reasoning, where precision and traceability are essential  [ 81 ,  60 ,  70 ] . In system operation settings, retrieval enables models to surface historical signals, recurring patterns, and contextual system metadata, creating the foundation for downstream diagnostic reasoning. As DT architectures increasingly integrate intelligence and autonomy, retrieval becomes a key mechanism for linking runtime observations to prior system states, enabling more interpretable and context-aware decision pipelines.

Events and messages originating from the kernel, applications, and users of a system are recorded in system logs. Thus, these logs form an extensive record of processes executed within a network  [ 86 ] . This provides system administrators with a crucial resource for monitoring performance, detecting security threats, and conducting root cause analysis (RCA)  [ 13 ,  44 ,  86 ] .

The Syslog protocol was created as a framework to allow machines to transmit these event notifications  [ 25 ] . Each Syslog message contains a PRI component that consists of a “

&lt;  &lt;

”, a number, and a “

&gt;  &gt;

”. The number is known as the severity value, which is a combination of the log’s Facility and Severity values. The Facility value is an integer ranging from 0 to 23, which describes the context of the log. The Severity value is an integer from 0 to 7 that quantifies the risk level of each log. Multiplying the Facility value by 8 and adding the Severity yields the Priority, which administrators use to identify and resolve system errors  [ 57 ] . This lack of strict standardization limits the use of severity labels as a canonical ground truth. However, it also makes them a realistic and challenging probe for evaluating whether language models (LMs) can align log content with operational intent under ambiguity  [ 24 ] .

Within DT-oriented monitoring pipelines, such log interpretation must be both accurate and latency-efficient, motivating the evaluation of compact, deployable LMs under strict output and runtime constraints.

The study focuses on evaluating small language models (SLMs) and small reasoning language models (SRLMs) using log severity classification as a controlled probe of their ability to ground real-world system log semantics under constrained outputs, based on logs collected from journalctl within the computing infrastructure.

2  Related work

2.1  Log Classification with Manual Methods

In the past, developers created sets of rules for processes such as anomaly detection, leading to manually implemented systems. However, with the rapid development of computing infrastructure in both complexity and scale, these methods have become time-consuming and error-prone  [ 61 ] . Modern computing systems generate logs rapidly; for example,  Le and Zhang [ 46 ]  estimate a rate of 30-50 gigabytes (about 120—200 million lines) per hour for Alibaba’s email production cloud computing system. The massive volume of system logs produced is impossible to manually traverse, complicating efforts to uncover patterns and nuanced faults that contribute to system failures  [ 95 ] .

He  et al.  [ 32 ]  observe another notable limitation: developers may suffer from insufficient technical expertise, such as an understanding of runtime behaviors, the functions of different log levels, or best logging practices. Often, large-scale systems will have hundreds of contributors, with each individual having deep knowledge of one sub-component of the overall system. Due to this specialization, developers may have gaps in their understanding of overall system behaviors and relationships  [ 33 ] . Both conditions lead to struggles with assigning the proper log level. As a result, developers often must retrace their work and revise the levels assigned to previous system logs, further decreasing the efficiency  [ 49 ] .

2.2  Log Classification with Traditional ML Methods

Due to the challenges of manual log classification, autonomous analysis methods have been explored through the creation of traditional machine learning (ML) methods  [ 18 ,  35 ,  91 ] . Commonly used ML algorithms include random forest (RF) and support vector machine (SVM)  [ 4 ] .

Recently,  Qi  et al.  [ 66 ]  utilized RF as a baseline, observing strong perfor