[2508.03555] PyLate: Flexible Training and Retrieval for Late Interaction Models

PyLate: Flexible Training and Retrieval for Late Interaction Models

Antoine Chaffin

antoine.chaffin@lighton.ai

0000-0003-3605-4097

LightOn  Nancy  France

and

Raphaël Sourty

raphael.sourty@lighton.ai

0009-0000-2158-4832

LightOn  Paris  France

Abstract.

Neural ranking has become a cornerstone of modern information retrieval. While single vector search remains the dominant paradigm, it suffers from the shortcoming of compressing all the information into a single vector. This compression leads to notable performance degradation in out-of-domain, long-context, and reasoning-intensive retrieval tasks. Multi-vector approaches pioneered by ColBERT aim to address these limitations by preserving individual token embeddings and computing similarity via the MaxSim operator. This architecture has demonstrated superior empirical advantages, including enhanced out-of-domain generalization, long-context handling, and performance in complex retrieval scenarios. Despite these compelling empirical results and clear theoretical advantages, the practical adoption and public availability of late interaction models remain low compared to their single-vector counterparts, primarily due to a lack of accessible and modular tools for training and experimenting with such models. To bridge this gap, we introduce PyLate, a streamlined library built on top of Sentence Transformers to support multi-vector architectures natively, inheriting its efficient training, advanced logging, and automated model card generation while requiring minimal code changes to code templates users are already familiar with. By offering multi-vector-specific features such as efficient indexes, PyLate aims to accelerate research and real-world application of late interaction models, thereby unlocking their full potential in modern IR systems. Finally, PyLate has already enabled the development of state-of-the-art models, including GTE-ModernColBERT and Reason-ModernColBERT, demonstrating its practical utility for both research and production environments.

†

†  copyright:  none

†

†  conference:  Make sure to enter the correct
conference title from your rights confirmation email; ;

†

†  isbn:  978-1-4503-XXXX-X/2018/06

1.  Introduction

Information retrieval (IR) has evolved significantly with the integration of pre-trained transformers such as BERT  (Devlin et al.,  2019 )  to perform semantical search, overcoming the vocabulary mismatch of lexical approaches.
Dense (single vector) search  (Karpukhin et al.,  2020 )  employs these transformers-based models to create contextualized representations ( embeddings ) of the input sequence tokens and then apply a pooling operation such as max, mean or CLS (first token) to aggregates these token-level embeddings into a single fixed-dimensional vector representing the entire sequence.
The single vector paradigm facilitates pre-computation and indexing of document embeddings, enabling rapid retrieval by identifying nearest neighbors to a query embedding within a shared vector space using a similarity such as cosine.

Numerous models have been trained following this simple approach with various training objectives, datasets and sizes of models  (Zhang et al.,  2024 ; Li et al.,  2023b ; Chen et al.,  2024 ; Sturua et al.,  2024 ; Nussbaum et al.,  2024 ) . However, despite strong performance on various benchmarks, models using single vector approach suffer from an inherent limitation: compressing rich semantic information into a single dense representation is lossy. Indeed, compressing hundreds of vectors into a single one requires to select what information to keep and what to throw. The model learns this selective behavior during training and is thus very dependent on the data used, explaining why dense models struggle when used out-of-domain. Besides, the limitation of this compression is also getting stronger as the quantity of information is getting bigger, and thus dense models struggle at handling long context  (Zhu et al.,  2024 ; Ognawala and Cureton-Griffiths,  2025 ) . This is particularly damaging considering modern pipelines are exposed to such long context elements and recent encoders support them  (Warner et al.,  2024 ; Zhang et al.,  2024 ) .

To address these limitations,  ColBERT

(Khattab and Zaharia,  2020 )  introduced a  multi-vector approach  based on  late interaction . ColBERT deviates from single-vector models by removing the pooling operation, thereby preserving individual contextualized embeddings for every query and document token. The similarity between a query and a document is then computed through the  MaxSim operator : the summation of each query token maximum similarity with any of the document tokens. Formally, given a query

Q  Q

with

|  Q  |

|Q|

token embeddings

q  i

q_{i}

and a document

D  D

with

|  D  |

|D|

token embeddings

d  j

d_{j}

, the MaxSim score

S  ​

(  Q  ,  D  )

S(Q,D)

is calculated as:

S  ​

(  Q  ,  D  )

=

∑

i  =  1

|  Q  |

max

j  =  1

|  D  |

⁡

(

q  i

⋅

d  j

)

S(Q,D)=\sum_{i=1}^{|Q|}\max_{j=1}^{|D|}(q_{i}\cdot d_{j})

This multi-vector approach enable stronger out-of-domain generalization  (Khattab and Zaharia,  2020 ; Santhanam et al.,  2022b ; Clavié,  2024 ) , tremendously better handling of long context  (Warner et al.,  2024 ; Chaffin,  2025a ; Bergum,  2024 )  and outperform single-vector models more than 45 times its size  (Chaffin,  2025b )  on reasoning-intensive retrieval tasks

(Su et al.,  2025 ) .

Despite these strong empirical results and theoretical advantage, very few late interaction models  1

1  1 This process is referred to as late interaction by opposition to cross-encoders that merge the query and the document to encode them jointly, thus being early interaction.

have been released compared to the large number of dense models  2

2  2 As an illustration of this disparity, over 15,000 dense models, many of which are developed and released using the Sentence Transformers library, are publicly available on platforms like Hugging Face at the time of writing ( huggingface.co/models?library=sentence-transformers ).

.
This disparity is in part attributable to the lack of accessible tools to train and experiment with these models. While foundational, the  original ColBERT-ai codebase  functions primarily as a research repository that can be difficult to use and update for newcomers. For example, simply integrating a new model requires non-trivial modifications of a few files. This makes experimenting outside of the existing scope of ColBERT work very difficult.

A key factor contributing to the widespread popularity of dense models is the  Sentence Transformers (ST)  library  (Reimers and Gurevych,  2019 ) , which defines a standard and offers a framework to train, use, and experiment with diverse models easily. Recognizing this, the present work introduces  PyLate , a library designed to extend Sentence Transformers to support multi-vector models.

2.  Extending Sentence Transformers

Essentially, late interaction models are dense models without the pooling operation and using the MaxSim operator rather than traditional similarity metrics. Thanks to the modular implementation of Sentence Transformers, both the pooling operation and the similarity computation are decoupled from the main modeling of the transformer model. We thus created PyLate, a library that extends Sentence Transformers to multi-vector models. This design choice offers several key advantages:

Loading any model. 
Any model available on Hugging Face can be used as the base transformer model by specifying the model name. It includes experimental models from community not yet merged in transformers by setting the parameter

u  ​  s  ​  e  ​  _  ​  r  ​  e  ​  m  ​  o  ​  t  ​  e  ​  _  ​  c  ​  o  ​  d  ​  e

=

T  ​  r  ​  u  ​  e

use\_remote\_code=True

.

Efficient and transparent training.