[2110.15232] Guided Evolution for Neural Architecture Search

Guided Evolution for Neural Architecture Search

Vasco Lopes
 NOVA Lincs, Universidade da Beira Interior

vasco.lopes@ubi.pt

&amp;Miguel Santos

NOVA Lincs, Universidade da Beira Interior

miguel.santos@ubi.pt

&amp;Bruno Degardin

Universidade da Beira Interior

bruno.degardin@ubi.pt

&amp;Luís A. Alexandre

NOVA LINCS, Universidade da Beira Interior

luis.alexandre@ubi.pt

Abstract

Neural Architecture Search (NAS) methods have been successfully applied to image tasks with excellent results. However, NAS methods are often complex and tend to converge to local minima as soon as generated architectures seem to yield good results. In this paper, we propose G-EA, a novel approach for guided evolutionary NAS. The rationale behind G-EA, is to explore the search space by generating and evaluating several architectures in each generation at initialization stage using a zero-proxy estimator, where only the highest-scoring network is trained and kept for the next generation. This evaluation at initialization stage allows continuous extraction of knowledge from the search space without increasing computation, thus allowing the search to be efficiently guided. Moreover, G-EA forces exploitation of the most performant networks by descendant generation while at the same time forcing exploration by parent mutation and by favouring younger architectures to the detriment of older ones. Experimental results demonstrate the effectiveness of the proposed method, showing that G-EA achieves state-of-the-art results in NAS-Bench-201 search space in CIFAR-10, CIFAR-100 and ImageNet16-120, with mean accuracies of 93.98%, 72.12% and 45.94% respectively.

1  Introduction

Convolutional Neural Networks (CNNs) have been extensively applied with success to a panoply of tasks, from image classification  ( Deng et al. ,

2014  ;  Goodfellow et al. ,

2016  ) , to semantic segmentation  ( Garcia-Garcia et al. ,

2018  ) , text analysis  ( Conneau et al. ,

2017  ) , amongst many others  ( Khan et al. ,

2020  ) . Their inherent capability of feature extraction allows CNNs to be easily applied and transferred to different problems. Over the years, several brilliantly and carefully designed architectures have incrementally out-performed the state-of-the-art by proposing novel components and mechanisms, such as skip and residual connections, faster and less size intensive operations and attention mechanisms  ( Krizhevsky et al. ,

2012  ;  Szegedy et al. ,

2015  ;  He et al. ,

2016  ;  Huang et al. ,

2017  ;  Chollet ,

2017  ;  Tan and Le ,

2019  ;  Dosovitskiy et al. ,

2021  ) . However, designing tailor-made highly performant CNNs for a given task is extremely difficult, as the required design choices intrinsic to the architectures, layer combination and training requires extensive architecture engineering. Thus, there is a growing interest in Neural Architecture Search (NAS) to automate architecture engineering and design.

NAS has successfully been successfully applied in the task of designing different types of neural network’s architectures  ( Wistuba et al. ,

2019  ) , specially for image and text problems  ( Elsken et al. ,

2019b  ;  Wistuba et al. ,

2019  ) . These methods are commonly composed of three components, being the first the search space, which specifies the possible operations to be sampled and their connections, ultimately defining the type of architectures that the search method can generate. The second component is the search method, which represents the approach used to explore the search space and generate architectures. The most common approaches are reinforcement learning, evolutionary strategies and gradient-based methods, which commonly work by updating a controller to sample more efficient architectures based on the performance of the generated models. Finally, the performance estimation strategy defines how the generated architectures are evaluated. Thus, the goal of a NAS method is to, based on the search method, efficiently search a large set of possible networks to find an optimal architecture for a given problem. Despite the excellent results obtain by prominent NAS methods, the computational cost of most approaches is high, which in some cases can be in the order of months of GPU computation  ( Zoph and Le ,

2017  ;  Liu et al. ,

2018  ;  Zoph et al. ,

2018  ) . To mitigate this, interesting approaches focus on a cell-based design, where NAS methods design small cells that are replicated through an outer-skeleton, thus alleviating the complexity of the search space  ( Shu et al. ,

2020  ;  Zoph and Le ,

2017  ;  Zoph et al. ,

2018  ;  Carlucci et al. ,

2019  ) . More, several performance estimation strategies have been proposed to reduce the time constraint of NAS methods, by mainly conducting low-fidelity estimates, learning curve extrapolations, statistical approaches  ( White et al. ,

2021  ;  Elsken et al. ,

2019b  )  or by proposing one-shot methods, where the weights of the generated models are inherited  ( Liu et al. ,

2019  ;  Pham et al. ,

2018  ;  Xu et al. ,

2020  ) . Searching through extensive search spaces is highly complex, even when there is some prior knowledge about the space. It has been shown that some of the most prominent NAS methods fail to generalise to new datasets due to converging extremely fast to local minima  ( Dong and Yang ,

2020  ;  Yang et al. ,

2020  ) . The most reliable approach to obtain information about the search space while searching is to fully train generated architectures and optimise the search based on the most performant ones. However, this is costly, and results are highly dependant on the training schemes and initialisation setups. Therefore, zero-proxy estimators present an attractive solution, where statistics are drawn from the generated architectures to score them at initialisation stage, thus requiring no training  ( Lopes et al. ,

2021  ;  Mellor et al. ,

2021  ) . These methods are time efficient and capable of performing good correlations between the score and respective accuracies when the architectures are trained.

This paper proposes G-EA, an evolutionary NAS method that leverages zero-proxy estimation to guide the search. By using an evolutionary strategy, where operations can be mutated and younger architectures are prefered, G-EA forces an exploitation of the most performant networks, and an exploration of the search space by conducting mutations. More, we solve the problem of conducting full evaluation of the generated networks to obtain knowledge about the search space, by generating several architectures in each generation, where all are evaluated at initialisation stage using a zero-proxy estimator and only the highest scoring network is trained and kept for the next generation. By doing so, G-EA is capable of continuously extracting knowledge about the search space without compromising the search, resulting in state-of-the-art results in NAS-Bench-201 search space in CIFAR-10, CIFAR-100 and ImageNet16-120.

Our contributions can be summarized as:

•

We propose a novel guided NAS method based on evolutionary strategies and zero-proxy estimation to generate image classifier architectures - Convolutional Neural Networks.

•

We empirically show that guided mechanisms can be used to improve the generated models performance without compromising time efficiency. Also, we detail the algorithm, emphasizing the accessible transferability of the guiding mechanism.

•

We achieve state-of-the-art results in the NAS-Bench-201 search space, in all datasets: CIFAR-10, CIFAR-100 and ImageNet16-120.

2  Related Work

NAS was initially proposed as a Reinforcement Learning (RL) problem, where a controller is trained based on the generated architecture’s performances to sample more efficient ones  ( Zoph and Le ,

2017  ) . Follow-up approaches focused on improving the overall performance, and the computati