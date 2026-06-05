[2109.07812] Transductive Learning for Unsupervised Text Style Transfer

Transductive Learning for Unsupervised Text Style Transfer

Fei Xiao  1,4  ,
Liang Pang  1∗  ,
Yanyan Lan  3∗  ,
Yan Wang  5  ,
Huawei Shen  1,4  ,
Xueqi Cheng  2,4

1  Data Intelligence System Research Center
 and

2  CAS Key Lab of Network Data Science and Technology,

Institute of Computing Technology, Chinese Academy of Sciences

3 Institute for AI Industry Research, Tsinghua University

4 University of Chinese Academy of Sciences

5 Tencent AI Lab

{xiaofei19s, pangliang, shenhuawei, cxq}@ict.ac.cn

lanyanyan@tsinghua.edu.cn ,

brandenwang@tencent.com

Abstract

Unsupervised style transfer models are mainly based on an inductive learning approach, which represents the style as embeddings, decoder parameters, or discriminator parameters and directly applies these general rules to the test cases. However, the lacking of parallel corpus hinders the ability of these inductive learning methods on this task. As a result, it is likely to cause severe inconsistent style expressions, like  the salad is rude .
To tackle this problem, we propose a novel transductive learning approach in this paper, based on a retrieval-based context-aware style representation. Specifically, an attentional encoder-decoder with a retriever framework is utilized. It involves top-

K

𝐾

K

relevant sentences in the target style in the transfer process. In this way, we can learn a context-aware style embedding to alleviate the above inconsistency problem. In this paper, both sparse (BM25) and dense retrieval functions (MIPS) are used, and two objective functions are designed to facilitate joint learning. Experimental results show that our method outperforms several strong baselines. The proposed transductive learning approach is general and effective to the task of unsupervised style transfer, and we will apply it to the other two typical methods in the future.

†

†  footnotetext:  *Corresponding Author

1  Introduction

Text style transfer is an essential topic of natural language generation, which is widely used in many tasks such as sentiment transfer  (Hu et al.,  2017 ; Shen et al.,  2017 ) , dialogue generation  (Zhou et al.,  2018 ; Niu and Bansal,  2018 ; Su et al.,  2021 ) , and text formalization  (Jain et al.,  2019 ) . The target is to change the style of the text while retaining style-independent content. As it is usually hard to obtain large parallel corpora with the same content and different styles, unsupervised text style transfer becomes a hot yet challenging research topic in recent years.

Figure 1:

Illustration of the inconsistency problem and the idea of our transductive learning approach.

Most existing methods in this area try to find the general style transfer rules with an inductive learning paradigm, where style is represented as a specific form, e.g., embeddings, decoder parameters, or classifier parameters.
For example, embedding based methods  (Shen et al.,  2017 ; Lample et al.,  2019 ; John et al.,  2019 ; Dai et al.,  2019 ; Yi et al.,  2020 ; He et al.,  2020 )  utilize a highly generalized style embedding to replace the original sentence style and direct the generation process. Decoder based methods  (Prabhumoye et al.,  2018 ; Fu et al.,  2018 ; Luo et al.,  2019 ; Gong et al.,  2019 ; Krishna et al.,  2020 )  use multiple decoders for generation, where each encoder corresponds to an independent style. Classifier based methods  (Wang et al.,  2019 ; Liu et al.,  2020 ; Mai et al.,  2020 )  employ the gradient of a pre-trained style classifier to edit the latent representation of the target text.

It has been well accepted that inductive learning methods have the ability to work well when there are numerous supervised labels. However, in the case of unsupervised style transfer, we are just given corpora with different styles without knowing the parallel relation, i.e. supervision label for this task. As a result, the inductive learning methods fail to produce an accurate style transfer rule, leading to generating some severe inconsistent texts, such as ‘ the salad is rude ’, as shown in Figure

1  . The underlying reason for this phenomenon is that a perfect style is usually highly dependent on the context, e.g. ‘ terrible ’ for ‘ the salad ’ and ’ rude ’ for ’ a person ’. Without a large scale of parallel data, it is difficult to learn a general style transfer rule working for various contexts.

Inspired by the idea of transductive learning  (Vapnik,  1998 )  and some successful historical examples, such as Transductive SVM  (Joachims et al.,  1999 ) , we propose to introduce transductive learning to the area of unsupervised text style transfer. Specifically, transduction learning reasoning from specific cases to specific cases, which avoids learning a general rule to represent the style. For example, once we get a reference sentence ‘ the sandwiches are terrible ’ with negative emotion, a transductive learning method may connect the two sentences by the two kinds of food, e.g. ‘ salad ’ and ‘ sandwiches ’, then use ‘ terrible ’ to express the negative emotion for the food ’ salad ’.

From the above discussion, we can see that there are two challenges in applying transductive learning to unsupervised text style transfer: 1) how to find specific samples that are beneficial for the style transfer of the current text; 2) how to use the style expressions in these samples to complete the style transfer process. To tackle these two challenges, we propose a novel  T ran S ductive  S tyle  T ransfer (TSST) model. In TSST, a retriever is employed to obtain the required similar samples, which tackles the first challenge. An attention-based encoder-decoder framework is then utilized to combine the specific samples to tackle the second challenge. Specifically, TSST first encodes the original text to a contextual representation and a style-independent embedding. Then either sparse (BM25) or dense retrieval functions (MIPS) are used to find the top-

K

𝐾

K

samples in the target style corpus, which are encoded by the same encoder. After that, a recurrent decoder is utilized to generate transfer text word by word based on the representation of those retrieved samples, contextual representation, and the representations in the last step. To jointly learn the dense retriever, encoder, and decoder, two kinds of objective functions are used in this paper, i.e. retrieval loss and bag-of-words loss.

In summary, our contributions are as follows:

•

Facing the inconsistency problem in unsupervised style transfer, we propose a novel transductive learning approach, which avoids learning a general rule but relies on specific samples to complete the style transfer process.

•

We design a  T ran S ductive  S tyle  T ransfer (TSST) model, which employs a retriever to involve highly related samples to guide the learning of the target style.

•

Experiments on two benchmark datasets show that TSST alleviates the inconsistency problem and achieves competitive results against traditional baselines. Our code is available at  https://github.com/xiaofei05/TSST .

2  Related Work

Previous unsupervised text style transfer methods can be divided into three categories according to the way they control the text style, e.g. embedding based method, decoder based method, and classifier based method.

The embedding based methods assign a separated embedding for each style to control the style of generated text. Early work tries to disentangle the content and style in the text. They first implicitly eliminate the original style information from the text representation using adversarial training

(Shen et al.,  2017 ; Fu et al.,  2018 ; John et al.,  2019 )  or explicitly delete style-related words

(Li et al.,  2018 ; Sudhakar et al.,  2019 ; Wu et al.,  2019b ; Malmi et al.,  2020 ) . Then, decode or rewrite the style-independent content with the target style embedding. As a complete disentanglement i