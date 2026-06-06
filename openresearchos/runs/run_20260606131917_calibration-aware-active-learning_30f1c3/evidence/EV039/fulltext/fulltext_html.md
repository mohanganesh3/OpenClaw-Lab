[2505.23448] Network Inversion for Uncertainty-Aware Out-of-Distribution Detection

Network Inversion for Uncertainty-Aware Out-of-Distribution Detection

Pirzada Suhail
 IIT Bombay

psuhail@iitb.ac.in

Rehna Afroz Shaik
 IIT Bombay

rehna@iitb.ac.in

Amit Sethi
 IIT Bombay

asethi@iitb.ac.in

Abstract

Out-of-distribution (OOD) detection and uncertainty estimation (UE) are critical components for building safe machine learning systems, especially in real-world scenarios where unexpected inputs are inevitable. In this work, we propose a novel framework that combines network inversion with classifier training to simultaneously address both OOD detection and uncertainty estimation. For a standard n-class classification task, we extend the classifier to an (n+1)-class model by introducing a ”garbage” class, initially populated with random gaussian noise to represent outlier inputs. After each training epoch, we use network inversion to reconstruct input images corresponding to all output classes that initially appear as noisy and incoherent and are therefore excluded to the garbage class for retraining the classifier. This cycle of training, inversion, and exclusion continues iteratively till the inverted samples begin to resemble the in-distribution data more closely, suggesting that the classifier has learned to carve out meaningful decision boundaries while sanitising the class manifolds by pushing OOD content into the garbage class. During inference, this training scheme enables the model to effectively detect and reject OOD samples by classifying them into the garbage class. Furthermore, the confidence scores associated with each prediction can be used to estimate uncertainty for both in-distribution and OOD inputs. Our approach is scalable, interpretable, and does not require access to external OOD datasets or post-hoc calibration techniques while providing a unified solution to the dual challenges of OOD detection and uncertainty estimation.

1  Introduction

The increasing deployment of machine learning models in high-stakes, real-world applications—such as autonomous driving, medical diagnosis, and financial decision-making—has underscored the importance of model reliability and robustness. A key limitation of modern neural networks is their tendency to produce overconfident predictions  [  9  ]  even on inputs that lie far outside the training distribution. This makes it crucial to develop models capable of both out-of-distribution (OOD) detection—the ability to identify inputs that fall outside the training distribution—and uncertainty estimation (UE)—the ability to quantify confidence in predictions to ensure safe decision-making under distributional shift.

Both capabilities are vital for trustworthiness in deployment scenarios where the data encountered during inference may deviate from the training distribution in subtle or unexpected ways. Although these two problems are inherently linked, most existing approaches treat them separately, often relying on post-hoc calibration techniques or auxiliary OOD datasets, which may not always be available.

Recent work in  [  1  ]  proposed Autoinverse, a framework for neural network inversion that prioritizes solutions near reliable training samples, using embedded regularization and predictive uncertainty minimization to improve robustness. Later  [  6  ]  introduced a semantically coherent OOD detection (SCOOD) approach by combining uncertainty-aware optimal transport with dynamic cost modeling and inter-cluster enhancements. While  [  3  ]  developed a Gaussian process-based model that operates solely on in-distribution data, defining predictive uncertainty scores without requiring OOD examples during training. Similarly,  [  2  ]  presents PostNet, which employs normalizing flows to model posterior distributions over predicted probabilities, allowing reliable uncertainty estimation and effective OOD discrimination—even without OOD supervision.

In this work, we propose a novel framework that leverages network inversion [  7  ,

8  ] , not only to detect OOD inputs but also to estimate prediction uncertainty, unifying the two objectives in a single training procedure. By extending a standard (n+1)-class model with an auxiliary garbage class, and iteratively refining the model using inverted reconstructions, we encourage the network to carve out clean decision boundaries while isolating ambiguous or anomalous regions. Unlike prior approaches, our method requires no external OOD datasets or post-hoc calibration, offering a simple and interpretable solution to ensure robustness in classification under distributional shift.

2  Methodology

Our unified training approach integrates out-of-distribution (OOD) detection and uncertainty estimation (UE) into a single framework using network inversion and an auxilary garbage class. For an n-class classification task, we extend the classifier to an (n+1)-class model by introducing an additional ”garbage” class designed to absorb anomalous inputs. This garbage class is initially populated with random Gaussian noise, representing unstructured inputs not observed in the training distribution.

Between successive training epochs, we perform network inversion as in  [  8  ]  to reconstruct samples from the input space of the classifier for all output classes. Given the vastness of the input space, during early training stages, these reconstructions tend to be visually incoherent and do not resemble real data, reflecting the model’s incomplete or uncertain understanding of the class manifolds. These reconstructions are assigned to the garbage class and added to the training set for the subsequent epochs. In subsequent epochs the classifier is trained using a weighted cross-entropy loss to account for the class imbalance introduced by addition of garbage samples.

By iteratively repeating this cycle of training, inversion, and exclusion, the model gradually learns to refine the decision boundaries while pushing anomalous content into the garbage class. As the training progresses, inverted samples in Fig

1

begin to look like training data, indicating that the classifier has effectively carved out the in-distribution manifold while isolating outliers into the garbage class.

During inference, this training procedure equips the classifier to identify and reject out-of-distribution (OOD) inputs by assigning them to the garbage class. Additionally, the softmax confidence scores corresponding to class predictions can be used to assess the model’s uncertainty. Low softmax confidence on in-distribution predictions indicates ambiguous or uncertain inputs, while high confidence in the garbage class suggests a strong belief that the input is OOD. We quantify uncertainty using the softmax confidence values across all

n  +  1

n+1

output classes by capturing how sharply peaked or spread out the model’s predictive distribution is. The uncertainty estimate for a prediction

𝐩  \mathbf{p}

is given by:

UE  ​

(  𝐩  )

=

1  −

∑

i  =  1

n  +  1

(

p  i

−

1

n  +  1

)

2

∑

i  =  1

n  +  1

(

δ

i  ,  k

−

1

n  +  1

)

2

\text{UE}(\mathbf{p})=1-\frac{\sum_{i=1}^{n+1}\left(p_{i}-\frac{1}{n+1}\right)^{2}}{\sum_{i=1}^{n+1}\left(\delta_{i,k}-\frac{1}{n+1}\right)^{2}}

(1)

where

k  =

arg  ⁡

max  i

⁡

p  i

k=\arg\max_{i}p_{i}

and

δ

i  ,  k

\delta_{i,k}

is the Kronecker delta. The resulting score ranges from 0 (maximum certainty) to 1 (maximum uncertainty), providing an interpretable measure of prediction confidence.
Specifically, it computes the squared distance between the predicted softmax vector

𝐩  \mathbf{p}

and the uniform distribution, normalized by the maximum possible distance under a one-hot prediction.

3  Results

We evaluate the effectiveness of our approach to uncertainty-aware out-of-distribution detection across four benchmark image classification datasets: MNIST  [  4  ] , FashionMNIST  [  10  ] , SVHN, and CIFAR-10  [  5  ] . To as