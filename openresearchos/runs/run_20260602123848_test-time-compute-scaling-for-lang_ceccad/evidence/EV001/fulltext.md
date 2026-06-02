[2408.02361] Dialogue Ontology Relation Extraction via Constrained Chain-of-Thought Decoding

Dialogue Ontology Relation Extraction via Constrained Chain-of-Thought Decoding

Renato Vukovic,
David Arps,
Carel van Niekerk,
Benjamin Matthias Ruppik,

Hsien-Chin Lin,

Michael Heck,

Milica Gašić

Heinrich Heine University Düsseldorf

{renato.vukovic, david.arps, niekerk, ruppik, linh, heckmi, gasic}@hhu.de

Abstract

State-of-the-art task-oriented dialogue systems typically rely on task-specific ontologies for fulfilling user queries.
The majority of task-oriented dialogue data, such as customer service recordings, comes without ontology and annotation.
Such ontologies are normally built manually, limiting the application of specialised systems.
Dialogue ontology construction is an approach for automating that process and typically consists of two steps: term extraction and relation extraction.
In this work, we focus on relation extraction in a transfer learning set-up.
To improve the generalisation, we propose an extension to the decoding mechanism of large language models.
We adapt Chain-of-Thought (CoT) decoding, recently developed for reasoning problems, to generative relation extraction.
Here, we generate multiple branches in the decoding space and select the relations based on a confidence threshold.
By constraining the decoding to ontology terms and relations, we aim to decrease the risk of hallucination.
We conduct extensive experimentation on two widely used datasets and find improvements in performance on target ontology for source fine-tuned and one-shot prompted large language models.  1

1  1 The code is available under  https://gitlab.cs.uni-duesseldorf.de/general/dsml/dialogue-ontology-relation-extraction-via-constrained-chain-of-thought-decoding

Dialogue Ontology Relation Extraction via Constrained Chain-of-Thought Decoding

Renato Vukovic,
David Arps,
Carel van Niekerk,
Benjamin Matthias Ruppik,

Hsien-Chin Lin,

Michael Heck,

Milica Gašić

Heinrich Heine University Düsseldorf

{renato.vukovic, david.arps, niekerk, ruppik, linh, heckmi, gasic}@hhu.de

1  Introduction

Figure 1:

Example of constrained CoT-decoding for dialogue ontology extraction for a dialogue from MultiWOZ 2.1  Eric et al. ( 2020 ) .
Domains are highlighted in green, slots in yellow and values in red.
Branch 0 predicts an incorrect relation ( hotel  misclassified as slot) with lower confidence.
Branch 1 has the highest confidence in the relation prediction, which is why it is selected as the final response.
Also, it contains a form of reasoning that stresses the type of terms that are part of the relations to be predicted, i.e., slots and values.
Branch 2 visualises constrained decoding, where the prediction of terms and relations is not possible if they are not present in the input.

State-of-the-art task-oriented dialogue (TOD) systems still rely on a fixed ontology to model their scope  (Nguyen et al.,  2023 ; Hudeček and Dusek,  2023 ) .
A  TOD ontology  comprises three levels of hierarchy: domains, slots and values.
 Domains  are general topics of interest,  slots  are types of information about entities in a domain, and  values  are concrete instantiations of slots.
Ontology thus forms a hierarchy: it is a directed graph where slots belong to domains and values in turn belong to slots.
Note that slots can be shared across domains, and so can values.
An ontology is typically a prerequisite for generating API calls that access the underlying databases for entity retrieval.
Further, the ontology defines the dialogue state, which is tracked by the system to determine the next actions given the evolving discourse.

The dependency on an ontology poses a significant challenge in transferring existing TOD systems to new domains and use cases.
Although ontology-agnostic approaches do exist, their transfer capabilities are limited and their performance remains sub-par on novel data  Heck et al. ( 2022 ) .

Large quantities of domain-specific TOD data, e.g. customer service recordings, are frequently available, but tend to come without annotation, rendering direct use for system development difficult  Brusco and Gravano ( 2023 ) .
Manual labelling is error-prone, does not scale well and quickly becomes prohibitively expensive  Eric et al. ( 2020 ); Rosenbaum et al. ( 2022 ); Gung et al. ( 2023 ) .
Despite topical or domain mismatch, existing annotated datasets may provide information about TOD that can be leveraged to harness new data.
For this reason, we are interested in utilising existing labelled TOD datasets to automatically generate a full ontology for new, yet-unlabelled, data.

Automatic dialogue  ontology construction  typically consists of two steps, dialogue term extraction  (Vukovic et al.,  2022 )  and hierarchy establishment.
Although hierarchy establishment is often done via clustering  Hudeček et al. ( 2021 ); Yu et al. ( 2022 )  we approach it via  relation extraction  (RE), which is more similar to common information extraction pipelines  Genest et al. ( 2022 ); Xu et al. ( 2023 ) .
We call this task  dialogue ontology relation extraction  (DORE).
A hierarchy is established by inferring in which level extracted terms lie, and by connecting terms across levels.

Although large language models (LLMs) have demonstrated considerable task transfer abilities  Brown et al. ( 2020 ); Ouyang et al. ( 2022 ) , they still lack behind specialised systems in TOD modelling when appropriate training data is available  Heck et al. ( 2023 ); Hudeček and Dusek ( 2023 ) .

In this work, we assume that some labelled out-of-domain source dialogue data is available to facilitate transfer learning.
We examine two strategies of providing source data to an instruction-tuned LLM; 1) as one-shot examples in the prompt, and 2) as data for an additional round of supervised fine-tuning.
We establish a challenging transfer setup by conducting experiments on two well-established medium to large scale multi-domain task-oriented dialogue benchmark datasets: MultiWOZ 2.1  Budzianowski et al. ( 2018 ); Eric et al. ( 2020 )  and the Schema-Guided Dialogue  (SGD; Rastogi et al.,  2020 )  dataset.
Since our focus is solely on DORE, we assume that the results of the first step of ontology construction, namely term extraction, are provided.

We propose to improve the decoding mechanism of an LLM in order to better leverage task-specific knowledge.
Concretely, we constrain the generation to terms and relation types given in the model input to force the model to consider terms from the target data and output the desired format.
We further adapt  chain-of-thought (CoT) decoding

Wang and Zhou ( 2024 ) , which was recently proposed for logical reasoning, for DORE.
Traditionally, CoT methods prompt or train the model to generate reasoning paths before giving the final answer  Wei et al. ( 2022 ); Kim et al. ( 2023 ) .
CoT-decoding on the other hand exploits the observation that the presence of CoT-paths is correlated with higher confidence in the predicted answer in logical reasoning.
We extend CoT-decoding to DORE by selecting the final answer based on the confidence of predicted relations in multiple generated model answer branches.
Our final proposal,  constrained CoT-decoding for dialogue ontology extraction , is the combination of our CoT-decoding approach to RE with constrained decoding, see Fig.

1  .
Empirically, this new decoding mechanism significantly outperforms both source one-shot and source fine-tuned baselines on the target data.
Our contributions are as follows:

•

We propose to induce an ontological hierarchy by accumulating ontology relation predictions from the dialogues in a TOD dataset.

•

To the best of our knowledge, we are the first to apply  CoT-decoding  to dialogue ontology relation extraction.

•

We develop an extension, called  constrained CoT-decoding , for multi-relation extraction from task-oriented dialogues.

•

Constrained CoT-decoding significantly improv