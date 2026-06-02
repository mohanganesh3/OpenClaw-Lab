[2412.11385] Why Does ChatGPT “Delve” So Much? Exploring the Sources of Lexical Overrepresentation in Large Language Models

Why Does ChatGPT “Delve” So Much? Exploring the Sources of Lexical Overrepresentation in Large Language Models

Tom S. Juzek

Zina B. Ward
 Florida State University

tjuzek@fsu.edu, zward@fsu.edu

Conceptually, both authors contributed equally to this work. Tom wrote the code to the paper, which can be accessed at  github.com/tjuzek/delve .

Abstract

Scientific English is currently undergoing rapid change, with words like “delve,” “intricate,” and “underscore” appearing far more frequently than just a few years ago. It is widely assumed that scientists’ use of large language models (LLMs) is responsible for such trends. We develop a formal, transferable method to characterize these linguistic changes. Application of our method yields 21 focal words whose increased occurrence in scientific abstracts is likely the result of LLM usage. We then pose “the puzzle of lexical overrepresentation”:  why  are such words overused by LLMs? We fail to find evidence that lexical overrepresentation is caused by model architecture, algorithm choices, or training data. To assess whether reinforcement learning from human feedback (RLHF) contributes to the overuse of focal words, we undertake comparative model testing and conduct an exploratory online study. While the model testing is consistent with RLHF playing a role, our experimental results suggest that participants may be reacting differently to “delve” than to other focal words. With LLMs quickly becoming a driver of global language change, investigating these potential sources of lexical overrepresentation is important. We note that while insights into the workings of LLMs are within reach, a lack of transparency surrounding model development remains an obstacle to such research.

Why Does ChatGPT “Delve” So Much? Exploring the Sources of Lexical Overrepresentation in Large Language Models

Tom S. Juzek  and Zina B. Ward  †

†  thanks:  Conceptually, both authors contributed equally to this work. Tom wrote the code to the paper, which can be accessed at  github.com/tjuzek/delve .

Florida State University

tjuzek@fsu.edu, zward@fsu.edu

1  Introduction

Like all human language, Scientific English has changed substantially over time  Degaetano-Ortlieb and Teich ( 2018 ); Degaetano-Ortlieb et al. ( 2018 ); Bizzoni et al. ( 2020 ); Menzel ( 2022 ) . New discoveries have fueled (and perhaps been fueled by) the introduction of new lexical items into scientific discourse  Degaetano-Ortlieb and Teich ( 2018 ) .

Figure 1:  We formalize a procedure for identifying words whose increasing prevalence is likely the result of LLM usage. Although our focus is Scientific English, the method can be applied across domains and languages.

Changes in dominant methodological and explanatory frameworks – such as the rise of mechanical philosophy, or the mathematization of scientific fields – have been accompanied by changes in word usage and syntactic structures as well  Degaetano-Ortlieb and Teich ( 2018 ); Krielke ( 2024 ) . Such changes continue through the present  Banks ( 2017 ); Leong ( 2020 ) .

Over the last two years, however, Scientific English has witnessed increasing usage of certain lexical items at a seemingly unprecedented pace. Discussions on social media (e.g.,  Koppenburg  2024 ; Nguyen  2024 ; Shapira  2024  ) and in academic discourse  Gray ( 2024 ); Kobak et al. ( 2024 ); Liang et al. ( 2024b ); Liu and Bu ( 2024 ); Matsui ( 2024 )  have pointed out that words such as “delve,” “intricate,” and “nuanced” have appeared far more frequently in scientific abstracts from 2023 and 2024 compared to earlier years. Unlike many previous changes in Scientific English, these trends do not seem to be explained by changes in the content of science or in wider language use. Instead, it is widely assumed that the sharp increase is due to the use of large language models (LLMs) like ChatGPT for scientific writing. Evidence supporting this hunch has recently emerged (e.g.,  Cheng et al.  2024 ; Liang et al.  2024a  ).

The goals of the present research were twofold. First, we aimed to provide a systematic characterization of this linguistic phenomenon. Some existing work has relied on informal methods to identify words observed to occur more frequently in AI-generated writing (e.g.,  Matsui  2024  ). We developed a method for extracting lexical items of interest, described in Section

2  , which is rigorous, reproducible, and transferable to other data and models. We identified 21 “focal words”: lexical items that have recently spiked in Scientific English and are overused by ChatGPT-3.5 in scientific writing tasks, as illustrated in Figure

1  .

Prior research has focused on quantifying such focal words’ increasing prevalence and estimating how much recent scientific writing has been produced with LLM assistance (e.g.,  Kobak et al.  2024 ; Liang et al.  2024b  ). By contrast, our second goal was to explore the factors that might contribute to the phenomenon of lexical overrepresentation:  Why  does ChatGPT use “delve” (and other focal words) so frequently when generating scientific text? We identified a set of possible factors, characterized in Section

3  , and began to assess them. We did not find evidence that model architecture or algorithmic decisions play a major role in the overrepresentation of focal words (Section

5  ), nor that lexical overrepresentation stems from training or fine-tuning data (Section

4  ).

LLM training often involves reinforcement learning based on information about quality outputs from human evaluators. We found mixed evidence that reinforcement learning from human feedback (RLHF) contributes to the overrepresentation of our focal words in LLM-generated text. Positive evidence comes from model testing on Meta’s Llama LLM (Section

5  ). An exploratory experiment described in Section

6

is inconclusive, although our findings indicate that participants became wary of the word “delve” in the first sentence of an abstract (e.g., ’This article delves into …’). Since the experiment’s inconclusiveness stems partly from methodological issues, we believe a follow-up study is warranted. Many important questions about the future of LLM-driven language change remain (Section

7  ).

2  Corpus Analysis: Identification of Overrepresented Lexical Items

To probe recent changes in Scientific English, we used PubMed’s publicly available repository of scientific abstracts, which focuses on biomedical literature  National Library of Medicine ( 2023 )  (downloaded through the PubMed API using a Python script  Python Software Foundation ( 2024 ) ; Snapshot: May 4, 2024; all code on our GitHub). Our analysis includes more than 5.2 billion tokens (inflected forms) from 26.7 million abstracts. To track changes in word usage over time, we measured occurrences per million (opm) of a given token in each year. Figure

2

illustrates the usage trajectories of some baseline items over time. We focus on the period from 1975 to May 2024 as data prior to 1975 are less extensive.

Figure 2:  Selected lexical entries: change over time.

The goal of our corpus analysis was to identify words whose recent overuse in scientific writing is likely the result of LLM deployment. Our approach involved three steps. First, we determined which words were more prevalent in abstracts from 2024 compared to 2020 (since LLMs were not widespread pre-2021). We calculated the percentage increase in opm for each token in the database between 2020 and 2024. Unsurprisingly, there was a straightforward explanation for why some words spiked in usage during that time. For example, “omicron” and “metaverse” were two of the words that showed the largest percentage increase (for “omicron”, see Figure

2  ). We only considered increases deemed significant by chi-square tests, of which there were about 7300.

We were interested in