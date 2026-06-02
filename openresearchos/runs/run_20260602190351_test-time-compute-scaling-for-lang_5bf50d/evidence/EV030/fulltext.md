[2503.23395] Scaling Auditory Cognition via Test-Time Compute in Audio Language Models

Scaling Auditory Cognition via Test-Time Compute in Audio Language Models

Ting Dang  ♠  , Yan Gao  ⋄  , and Hong Jia  ♠

♠  School of Computing and Information Systems,
The University of Melbourne, Australia

⋄  Department of Computer Science and Technology,
University of Cambridge, UK

{ting.dang,hong.jia}@unimelb.edu.au ,  yg381@cam.ac.uk

Abstract

Large language models (LLMs) have shown exceptional versatility in natural language processing, prompting recent efforts to extend their multimodal capabilities to speech processing through the development of audio large language models (Audio LLMs). While Audio LLMs excel in tasks such as speech recognition and synthesis, it remains unclear how they perform when faced with the auditory cognitive challenges posed by real-world environments, such as audio comprehension and listening recall, particularly in the presence of background noise or overlapping speech. Unlike text-based LLMs, which have access to vast amounts of text data for pre-training, retraining Audio LLMs with diverse auditory cognitive scenes is difficult due to the limited datasets that simulate real-world auditory cognitive scenarios and the challenge of acquiring auditory cognitive labels for training. While test-time compute (TTC) methods have been shown to enhance the capabilities of text-based LLMs during inference, a key challenge lies in designing these TTC methods to improve the auditory capabilities of Audio LLMs. This study aims to address these two research gaps by: i) exploring the auditory cognitive capabilities of Audio LLMs, and ii) enhancing their capabilities using TTC approaches. We have investigated five different Audio LLMs for auditory cognition using a  self-collected  database and have proposed five TTC approaches to enhance auditory cognitive capabilities during inference. Our findings reveal that Audio LLMs performance decreases in more challenging auditory cognitive tasks. The proposed TTC approaches significantly enhance cognitive auditory capabilities, advancing the development of more adaptable and resilient Audio LLMs for practical applications such as assistive listening devices, voice-based AI assistants, and communication technologies.

1  Introduction

Large language models (LLMs) have achieved remarkable success as versatile natural language processing systems, demonstrating strong performance across a wide range of tasks such as machine translation  (Xu et al.,  2024 ) , question answering  (Zhuang et al.,  2023 ; Robinson et al.,  2022 ) , and summarization  (Laban et al.,  2023 ; Zhang et al.,  2024b ) . Building on this progress, recent research has explored extending LLMs to speech processing by integrating acoustic information into their architectures. This has led to the development of audio large language models (Audio LLMs), which can simultaneously process both spoken and written language. Through cross-modal alignment techniques, these models are increasingly capable of handling auditory tasks that require a unified understanding of both linguistic and paralinguistic information, such as emotion recognition  (Bellver et al.,  2024 ) , speech quality evaluation  (Wang et al.,  2025 ) , and music understanding  (Verma,  2025 ) .

Existing research on audio LLMs primarily focuses on their ability to process and align spoken and textual content, with efforts underway to expand their capabilities to include not only speech but also acoustic scenes and music  (Gong et al.,  2023 ; Li et al.,  2024 ) .
However, most existing studies were conducted on curated datasets or corpora (e.g., clean speech), which presents a significant limitation when applying the proposed LLMs to real-world auditory environments characterized by background noise, reverberation, and overlapping speech. Unlike text-based benchmarks, often built on large-scale annotated corpora, auditory cognition in complex settings relies on human adaptive mechanisms that are inherently difficult to quantify or label. The scarcity of datasets that faithfully replicate diverse real-world auditory scenarios has led to a limited number of studies addressing these challenging yet realistic conditions. Moreover, even fundamental questions regarding how audio LLMs exhibit cognitive adaptation across varying real-world listening environments remain largely unexplored.
Human listeners exhibit remarkable adaptability in complex situations, effortlessly extracting meaningful information despite distortions like the ”cocktail party problem”  (Haykin &amp; Chen,  2005 ) , supported by enhanced cognitive capabilities and working memory.
A natural question arises: do audio large LLMs possess capabilities comparable to those of humans? More specifically,  can audio LLMs develop human-like cognitive strategies to maintain comprehension in adverse auditory environments?

Enhancing these models to improve cognitive auditory capabilities under challenging conditions remains an important open question. In particular, a key challenge lies in effectively adapting pre-trained audio LLMs, originally trained on curated datasets, to real-world auditory environments with minimal additional effort.
Recent studies have showed a potential solution, test-time compute (TTC), wherein dynamic inference strategies enable models to ”think longer” on more difficult problems, thereby significantly improving inference performance without requiring additional fine-tuning  (Snell et al.,  2024 ; Bi et al.,  2024 ) .
Despite its effectiveness, applying TTC to real-world auditory environments remains largely unexplored. This presents a fundamental challenge:  How can the auditory cognitive robustness of audio LLMs be further enhanced during inference time? 
Addressing these two gaps is crucial for advancing the next generation of AI-driven auditory systems, enabling them to function reliably in real-world applications such as assistive listening devices, voice-based AI assistants, and communication technologies.

This paper aims to bridge this gap by investigating the inherent cognitive capabilities of audio LLMs in auditory processing across varying contexts, comparing their performance to human perception using a  self-collected  dataset, and exploring TTC methods to enhance auditory cognitive tasks.
To address the first question, we explored various audio LLMs in processing different auditory cognitive tasks, ranging from simpler tasks such as audio event recognition to more complex auditory cognitive tasks, including memorizing, recalling, and computing digits spoken in a sequence with overlapped speech. Additionally, to address the second gap, we designed five different TTC strategies to enhance auditory cognitive capabilities. The results on five common audio LLMs revealed that all models performed below human perception, while GPT-4 demonstrated surprisingly strong performance in recognizing overlapped speech.
The performance of the five proposed TTC approaches varies; however, all demonstrated improvements, with gains ranging from 9% to 150%.
The optimal strategy was found to be highly dependent on both the model structure and the complexity of the task. The contributions of this paper are summarized as follows:

•

This is the  first-study  to investigate the auditory cognitive capabilities of audio LLMs in real-world environments, demonstrating the inferior performance of current audio LLMs on certain auditory tasks.

•

This work innovatively proposed and evaluated five TTC strategies aimed at enhancing the performance of audio LLMs on auditory cognitive tasks. It represents the  first-attempt  to enhance audio LLM capabilities during inference time.

•

The TTC experiments, conducted using five different audio LLMs across various tasks, demonstrated significant improvements, highlighting their potential to enhance auditory cognitive processing.

2  Related Work

2.1  Auditory Pro