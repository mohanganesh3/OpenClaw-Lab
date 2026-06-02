[2502.10881] CiteCheck: Towards Accurate Citation Faithfulness Detection

CiteCheck : Towards Accurate Citation Faithfulness Detection

Ziyao Xu 1 , Shaohang Wei 1 , Zhuoheng Han 1 , Jing Jin 1 , Zhe Yang 1

Xiaoguang Li  2  , Haochen Tan  2  , Zhijiang Guo  2  , Houfeng Wang  1

1

National Key Laboratory for Multimedia Information Processing,

School of Computer Science, Peking University

2 Huawei Noah’s Ark Lab

{xzyxzy, 11jj617, yz_young, wanghf}@pku.edu.cn

{shaohang, 2100017789}@stu.pku.edu.cn  lixiaoguang11@huawei.com

haochetan2-c@my.cityu.edu.hk  zhijiangguo@hkust-gz.edu.cn

Abstract

Citation faithfulness detection is critical for enhancing retrieval-augmented generation (RAG) systems, yet large-scale Chinese datasets for this task are scarce. Existing methods face prohibitive costs due to the need for manually annotated negative samples. To address this, we introduce the first large-scale Chinese dataset  CiteCheck  for citation faithfulness detection, constructed via a cost-effective approach using two-stage manual annotation. This method balances positive and negative samples while significantly reducing annotation expenses.  CiteCheck  comprises training and test splits. Experiments demonstrate that: (1) the test samples are highly challenging, with even state-of-the-art LLMs failing to achieve high accuracy; and (2) training data augmented with LLM-generated negative samples enables smaller models to attain strong performance using parameter-efficient fine-tuning.  CiteCheck  provides a robust foundation for advancing citation faithfulness detection in Chinese RAG systems. The dataset is publicly available at  https://github.com/xzy-xzy/CiteCheck  to facilitate research.

CiteCheck : Towards Accurate Citation Faithfulness Detection

Ziyao Xu  1  , Shaohang Wei  1  , Zhuoheng Han  1  , Jing Jin  1  , Zhe Yang  1

Xiaoguang Li  2  , Haochen Tan  2  , Zhijiang Guo  2  , Houfeng Wang  1

1 National Key Laboratory for Multimedia Information Processing,

School of Computer Science, Peking University

2 Huawei Noah’s Ark Lab

{xzyxzy, 11jj617, yz_young, wanghf}@pku.edu.cn

{shaohang, 2100017789}@stu.pku.edu.cn  lixiaoguang11@huawei.com

haochetan2-c@my.cityu.edu.hk  zhijiangguo@hkust-gz.edu.cn

1  Introduction

Large Language Models (LLMs) are prone to generating factual errors through hallucinations when answering real-world questions. Retrieval-augmented generation (RAG) systems  Lewis et al. ( 2020 ); Guu et al. ( 2020 ); Borgeaud et al. ( 2022 )  address this limitation by leveraging external information retrieval to ground LLM responses in verifiable sources. Recent advancements extend RAG systems to generate text with inline citations  Gao et al. ( 2023b ) , enabling users to validate the reliability of generated content by cross-referencing cited documents. However, studies reveal a critical weakness in these systems: citation faithfulness. A substantial portion of generated text may lack proper support from the cited references  Liu et al. ( 2023 ); Hu et al. ( 2024b ) , undermining the trustworthiness and verification capability of RAG outputs. This challenge necessitates accurate citation faithfulness detection—determining whether cited passages genuinely support their associated claims—as a fundamental requirement for improving RAG reliability.

Developing robust citation faithfulness detection methods requires large-scale, high-quality datasets. While English benchmarks have emerged  Yue et al. ( 2023 ) , Chinese datasets remain notably absent. Constructing such resources presents unique challenges: realistic negative samples (unsupported citations) must originate from strong RAG systems to ensure practical usage, yet these systems rarely produce such errors. For instance, a RAG system with a 10% error rate would require annotating approximately 70,000 samples to collect 7,000 negative examples—a prohibitively expensive endeavor. This tension between dataset quality and construction cost demands innovative solutions for efficient data curation without compromising sample integrity.

To bridge this gap, we introduce  CiteCheck , the first large-scale Chinese dataset for citation faithfulness detection. Our approach combines 11,307 knowledge-intensive questions with a novel two-stage annotation framework that reduces labeling costs while preserving data quality.  CiteCheck  comprises two distinct components designed to address both detection difficulty and training efficacy.

The development and test sets each contain 500 positive (supported) and 500 negative (unsupported) samples totaling 2,000 unmodified RAG outputs. Experimental analysis demonstrates these original samples pose significant challenges, with state-of-the-art LLMs achieving limited detection accuracy. The training set includes 9,796 samples (4,898 positive/negative pairs) where negative instances are generated through LLM-based document modification rather than relying solely on rare RAG errors. Despite this augmentation, parameter-efficient fine-tuning on 7B-8B parameter models yields strong detection performance, confirming the preserved quality of modified negative samples.

Our contributions are threefold:  CiteCheck  establishes the first comprehensive benchmark for Chinese citation faithfulness detection; (2) We propose an efficient data augmentation strategy that reduces annotation costs by 86% compared to conventional approaches; (3) Extensive experiments validate the dataset’s quality and utility, showing that models trained on our augmented data effectively generalize to challenging real-world samples. This work advances reliable RAG development by providing essential resources and methodologies for building verifiable, citation-grounded LLM applications in Chinese.

2  Dataset Construction

Question

特斯拉在中国的纯电动汽车销量占比是多少？

What is Tesla’s share of all-electric car sales in China?

乘坐飞机的时候托运一个行李箱，再带一个20寸的箱子，带上飞机的箱子会被称重吗？

When I check a suitcase on an airplane and bring a 20” case, will the case be weighed when I bring it on the plane?

为什么当你把一幅画切成几块时，每一块都只有一部分，但是当你把一张全息图切成几块时，每一块都有完整的图像？

Why is it that when you cut a painting into pieces, each piece has only a part of it, but when you cut a hologram into pieces, each piece has the full image?

如果闪电劈中混凝土大楼之类的，就算在房间里处于隔缘状态，也会受伤害么？

If lightning strikes a concrete building or something like that, can you get hurt even if you’re in a room that’s insulated?

Answer

( Statement )

特斯拉在2023年上半年纯电动

汽车市场的占有率为21.7%。

Tesla’s share of the all-electric car market in the first half of 2023 is 21.7 percent.

是的，乘坐飞机时…。
 每件托

运行李的重量不能超过50公斤

，而每位旅客随身携带物品的

重量以5公斤为限。 
此外，每位旅客…。

The weight of each piece of checked baggage cannot exceed 50 kilograms, while the weight of each passenger’s carry-on is limited to 5 kilograms.

这是因为…。
 全息的原理在于

将动量空间通过全息技术记录

在实空间里，因此， 如果全息

图缺失一部分，那部分实空间

的信息就丢失了，像就不完整

了。 
即使…。

If a portion of the hologram is missing, that portion of real space is lost and the likeness is incomplete.

闪电击中混凝土大楼…
 在澳

大利亚新南威尔士州纽卡斯尔

的一个案例中，闪电击中了一

座大楼，虽然 大楼的结构保持

良好 ，但是巨大的爆裂声和震

动可能会对内部的人造成伤害

。 因此，…。

The building is structurally sound.

Cited

Documents

[1]【 2023上半年 全球纯电动汽车销量出炉…】…据该报道， 特斯拉在纯电动汽车市场期间占据21.7%的份额。 …

First half of 2023 / Tesla held a 21.7% share of the all-electric car market during the period.

[1] 办理 托运行李 对行李物品规定如下：… 每件行李物品重量不能超过50公斤。 …

Check-in baggage / The weight of each baggage item can not exceed 50 kilograms.

[2]  随身携带物品的重量，每位旅客以5公斤为限。 …

The weight of carry-on items is limited to 5 kg per passenger.

[1] … 如果普通照片缺失一部分，那部分实空间的信息就丢失了，像就不完整了。 全息照片如果缺失一部分，同样会造成信息的缺失，但是…

If a portion of an ordinary photograph is missing, that portion of real space is lost and the likeness is incomplete.

[1] …澳大利亚新南威尔士州纽卡斯尔…可清楚看到闪电击中大楼的场面，同时可听到巨大的爆裂声。据悉，闪电所击中的大楼为一处健身房。…

Label

positive

positive

negative

negative

Note

supported by a single document

supported by multiple documents

contradictory information

unmentioned information

Table 1:

Sample examples of the dataset. For the answer a