[2412.17741] Reasoning to Attend: Try to Understand How &lt;SEG&gt; Token Works

Reasoning to Attend: Try to Understand How  &lt;SEG&gt;  Token Works

Rui Qian  1

Xin Yin  2

Dejing Dou  1

1  School of Computer Science and Technology, Fudan University

2  The State Key Laboratory of Blockchain and Data Security, Zhejiang University

qianruii@126.com, xyin@zju.edu.cn, dejingdou@gmail.com

Corresponding Author (  dejingdou@gmail.com ).

Abstract

Current Large Multimodal Models (LMMs) empowered visual grounding typically rely on  &lt;SEG&gt;  token as a text prompt to jointly optimize the vision-language model (e.g., LLaVA) and the downstream task-specified model ( e.g ., SAM). However, we observe that little research has looked into how it works.
In this work, we first visualize the similarity maps, which are obtained by computing the semantic similarity between the  &lt;SEG&gt;  token and the image token embeddings derived from the last hidden layer in both the LLaVA encoder and SAM decoder. Intriguingly, we have found that a striking consistency holds in terms of activation responses in the similarity map,
which reveals that what  &lt;SEG&gt;  token contributes to is the semantic
similarity within image-text pairs.
Specifically,  &lt;SEG&gt;  token, a placeholder expanded in text vocabulary, extensively queries among individual tokenized image patches to match the semantics of an object from text to the paired image while the Large Language Models (LLMs) are being fine-tuned. Upon the above findings, we present READ, which facilitates LMMs’ resilient  REA soning capability of where to atten D  under the guidance of highly activated points borrowed from similarity maps. Remarkably, READ features an intuitive design, Similarity as Points module (SasP), which can be seamlessly applied to  &lt;SEG&gt; -like paradigms
in a plug-and-play fashion.
Also, extensive experiments have been conducted on the ReasonSeg and RefCOCO(+/g) datasets. To validate whether READ suffers from
catastrophic forgetting of previous skills after fine-tuning, we further assess its generation ability on an augmented FP-RefCOCO(+/g) dataset. All codes and models are publicly available at  https://github.com/rui-qian/READ .

Figure 1 :

Qualitative analysis of
 &lt;SEG&gt;  token on ReasonSeg  train  split. Points derived from

(  c  )

(c)

serve as prompt with original SAM in

(  a  )

(a)

. Text “antler” with image token from CLIP is in

(  b  )

(b)

.
The similarity between  &lt;SEG&gt;  token and image token embeddings stemming from the last hidden layer are obtained by Eq.(  5  ), w.r.t. LLaVA encoder in

(  c  )

(c)

and SAM decoder in

(  d  )

(d)

. The consistency in

(  b  )

(b)

,

(  c  )

(c)

,

(  d  )

(d)

indicates that  &lt;SEG&gt;  token in
LMMs learns semantics similar to those of direct
mention in texts. Refer to Appendix

B

for more illustrations.

1  Introduction

Reasoning segmentation has been newly proposed yet largely unexplored at present  [  15  ] . As an extension of classical Referring Expression Segmentation (RES)  [  10  ] , it aims to output nuanced masks for implicitly referred objects given descriptive language expressions. As shown in Fig.

1  , when asked, “What part of the deer’s body in the picture is used for defense and attracting mates?”, reasoning segmentation infers “antler” without an explicit mention, in contrast to traditional RES, which relies on direct referring. Solving such intricate visual tasks is non-trivial, requiring models to comprehend user intentions based on given queries while also possessing pertinent world knowledge  [  41  ] .

Recent works  [  15  ,

41  ,

42  ,

31  ,

34  ] 
have advanced reason segmentation tasks by
leveraging  &lt;SEG&gt;  token as a text prompt to seamlessly align LMMs empowered visual encoder ( e.g ., LLaVA  [  20  ] ) and the downstream task-specified decoder ( e.g ., SAM  [  14  ] ) in vision space for fine-grained output formats,  i.e ., segmentation masks. Specifically, SESAME  [  15  ] 
teaches LMMs to respond to false premises
by introducing negative samples into the pipeline. GSVA  [  42  ]  bridges the gap where the multiple-target and empty-target cases are neglected. GLaMM  [  31  ] 
and PixelLM  [  34  ]  enhances the model both in textual and visual domains, with versatile capability at various levels of granularity.

However, we observe that little research has looked into how  &lt;SEG&gt;  token works when mapping language vocabulary embedding into corresponding visual space. The  &lt;SEG&gt;  token, an extended placeholder in the text vocabulary, lacks inherent semantics on its own. However, when it was inserted into conversation templates and jointly trained with LMMs, ultimately being able to ground objects within an image. Recent works  [  15  ,

41  ,

42  ]  all employ the SAM  [  14  ]  model as a mask decoder. Initially, segmenting the red region in Fig.

1

could be achieved by prompting SAM with the text ”antler,” but now  &lt;SEG&gt;  token embedding fulfills the same purpose in place. This leads us to ask:  What is the connection of embeddings between  &lt;SEG&gt;  token and text prompt “antler” in terms of semantics?

Bearing this in mind, we begin by visualizing similarity maps, which are generated by computing dot product similarity between  &lt;SEG&gt;  token and image token embeddings extracted from the last hidden layer in both LLaVA  [  20  ]  and SAM  [  14  ]  models. Notably,
we observe a striking consistency in activation responses across this similarity
maps, suggesting that  &lt;SEG&gt;  token is pivotal in bridging semantic connections between textual prompts and their visual correspondences. Specifically,  &lt;SEG&gt;  token, an expansion of text vocabulary, initiates a thorough query across each tokenized image patch, aligning the textual semantics of an object with its visual counterparts in the paired image. Inspired by the above findings, an intuitive idea is to see if we can imply to the model where to “attend” by leveraging similarity maps.

To this end, we present READ, which
facilitates LMMs’ resilient  REA soning capabilities of where to atten D ,
informed by highly activated points stemming from similarity
maps. In particular, our READ consists of three core modules: (1) a LLaVA encoder, (2) a Similarity as Points module (SasP), and (3) a SAM decoder. Specifically, our LLaVA  [  20  ]  enhanced encoder consumes image-text pairs as input to generate text output, from which
the last hidden layer embedding for  &lt;SEG&gt;  token is then gathered. To guide the model where to attend, our SasP module computes similarity maps by performing a dot product between the  &lt;SEG&gt;  token embedding and their associated image patches, whereupon regions with high similarity scores are then converted into point coordinates for fine-grained mask predictions through the SAM decoder, along with textual prompts,  i.e . the  &lt;SEG&gt;  embedding. To address the challenge posed by discrete, non-differentiable points during back-propagation, we apply a Gaussian-like weighted average interpolation to render them continuously differentiable. This modification facilitates gradients across similarity maps back to the LMMs, empowering the model to “reason to attend” in the forwards, and “attend to reason” in the backward and vice versa. Particularly, READ’s intuitive design, SasP, can be effortlessly integrated into off-the-shelf  &lt;SEG&gt; -like paradigms with minimal overheads in a plug-and-play manner. In summary, our contributions are threefold:

•

We have looked into how  &lt;SEG&gt;  token works when mapping
language vocabulary embedding into corresponding visual space. Such investigation reveals that what  &lt;SEG&gt;  token mainly contributes to is the semantic correspondences from image-text pairs based on our findings.

•

We present our model — READ,
which empowers LMMs to “reason to attend” and “attend to reason” vice versa. Importantly,