[2503.22051] [Preprint] Non-Monotonic Attention-based Read/Write Policy Learning for Simultaneous Translation

[Preprint] Non-Monotonic Attention-based Read/Write Policy Learning for Simultaneous Translation

Zeeshan Ahmed, Frank Seide, Zhe Liu, Rastislav Rabatin, Jachym Kolar,

Niko Moritz ,  Ruiming Xie ,  Simone Merello  and Christian Fuegen

Meta AI

work done while at Meta Inc.work done while at Meta Inc.

Abstract

Simultaneous or streaming machine translation generates translation while reading the input stream. These systems face a quality/latency trade-off, aiming to achieve high translation quality similar to non-streaming models with minimal latency. We propose an approach that efficiently manages this trade-off. By enhancing a pretrained non-streaming model, which was trained with a seq2seq mechanism and represents the upper bound in quality, we convert it into a streaming model by utilizing the alignment between source and target tokens. This alignment is used to learn a read/write decision boundary for reliable translation generation with minimal input. During training, the model learns the decision boundary through a read/write policy module, employing supervised learning on the alignment points (pseudo labels). The read/write policy module, a small binary classification unit, can control the quality/latency trade-off during inference. Experimental results show that our model outperforms several strong baselines and narrows the gap with the non-streaming baseline model.

[Preprint] Non-Monotonic Attention-based Read/Write Policy Learning for Simultaneous Translation

Zeeshan Ahmed, Frank Seide, Zhe Liu, Rastislav Rabatin  †

†  thanks:  work done while at Meta Inc.

, Jachym Kolar  †

†  thanks:  work done while at Meta Inc.

,

Niko Moritz ,  Ruiming Xie ,  Simone Merello

and  Christian Fuegen

Meta AI

1  Introduction

Simultaneous or streaming machine translation initiates the translation process while still receiving the input. This contrasts with conventional translation systems, which waits to process the entire input before starting the translation. Simultaneous translation is particularly beneficial for translating live speech in real-time, as well as in two-party conversations where users want to hear their partner promptly to have a more natural conversation even if they speak different languages.

Standard non-streaming machine translation systems rely on an encoder-decoder architecture, which typically requires processing the entire input before starting to translate. The decoder uses an attention module that sees all encoder outputs. For simultaneous translation, both the encoder and attention module must be adaptive, allowing the encoder to operate incrementally and the cross-attention in the decoder to handle variable-length encoder output sequences. In this setup, the streaming model is fed one token at a time, and at each decoding step, it decides whether it wants to wait for another token or write a token to the output. The simultaneous translation systems face a trade-off between quality and latency. Prioritizing higher quality increases latency (with non-streaming systems still representing the upper bound in quality), while reducing latency can impact translation quality. Therefore, simultaneous systems must carefully balance this trade-off for practical applications.

With the recent approaches  Ma et al. ( 2023 ,  2019b ); Arivazhagan et al. ( 2019 ); Chiu and Raffel ( 2018 ); Raffel et al. ( 2017 ) , streaming machine translation is achieved with an alternative attention mechanism called Monotonic Attention (MA). The MA doesn’t have access to the full sentence thus can only attend to the previous encoder states. The MA has an attention component together with a policy component. The attention component job is still to attend to the relevant pieces of the available input while the policy learner on the other hand learns a policy to decide whether to read more input or generate the translation (write). This read/write policy decision makes the model capable of generating a translation as soon as the model has seen enough information at the input.

In addition to the MA approach, there are approaches which use a fixed size policy. For example, the wait-

k  k

approach which waits for the

k  k

input tokens to arrive before generating the translation. It then alternates between read/write operations based on the read/write probability  Dalvi et al. ( 2018 ); Ma et al. ( 2019a ) . Even a non-streaming model has been used for streaming translation where the system uses some heuristics to estimate the threshold based on MT confidence to drive the read/write policy decision  Cho and Esipova ( 2016 ) .

We present a novel approach to simultaneous translation called ALIgnment BAsed STReaming Machine Translation (AliBaStr-MT) which leverages a high quality pretrained non-streaming model as a starting point and uses the  non-monotonic attention as monotonic attention  when equipped with a learnable read/write module. The read/write module is a binary classifier which is trained based on the alignment obtained from the pretrained model.

These are the advantages of our approach:

•

Training efficiency : the model doesn’t need to be retrained or trained with expensive operations as in MMA model. Only a light weight read/write classifier is trained with alignment information from the base non-streaming model (which is also used to mask the future encoder states during training).

•

Inference efficiency : computation of read/write probability for MMA model during inference is expensive. In our case, we only need to run a small classifier on the top of the model which is quite fast.

•

Easy fallback to non-streaming model : The base non-streaming model either remains unchanged or trained together with a read/write module. It offers the flexibility to still use the base model as the non-streaming model when high quality is required.

•

Flexibility between latency and accuracy trade-off : The calibration threshold of the read/write binary classifier becomes the inference time hyper-parameter instead of training time parameter as in monotonic attention i.e. changing the parameter value requires retraining the model. In our approach, it allows us to tune the model for the quality/latency trade-off during inference time.

2  Background

The simultaneous translation systems can be broadly classified into fixed policy and adaptive policy systems. In the fixed policy system, the model uses the predefined heuristic and rules to generate the translation. These heuristics remain constant throughout the translation. The fixed policy systems such as wait-

k  k

or prefix-to-prefix translation systems by  Dalvi et al. ( 2018 ); Ma et al. ( 2019a )  are simple and yet effective methods for streaming translation. However, the value for

k  k

in wait-

k  k

is not a learnable parameter of the model and set beforehand making it difficult to adjust based on the input and so far generated translation. The

k  k

is also language dependent for different translation directions making it difficult to build a multilingual streaming translation model with wait-

k  k

.  Guo et al. ( 2024 )  on the other hand tries to fix the quality gap in simultaneous translation models by introducing a look-ahead feature and the curriculum learning during training.

The adaptive policy methods on the other hand dynamically adjust the read/write decision based on the input and previously generated translation. The adaptive method includes the ones where non-streaming MT model (trained on full sentences) is kept fixed/frozen and a heuristics or learnable module is added to make read/write decisions e.g. the confidence threshold (What-if-*) method of  Cho and Esipova ( 2016 )  and reinforcement learning approaches by  Grissom II et al. ( 2014 ); Gu et al. ( 2017 )  use pretrained non-streaming model which is not modified during the training.  Press and Smi