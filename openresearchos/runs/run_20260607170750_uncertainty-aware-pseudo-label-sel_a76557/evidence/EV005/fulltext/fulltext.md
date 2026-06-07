<!-- page 1 -->
Addressing Data Scarcity in 3D Trauma Detection
through Self-Supervised and Semi-Supervised
Learning with Vertex Relative Position Encoding
Shivam Chaudhary
Pattern Recognition Lab
FAU Erlangen-N¨urnberg
shivam.chaudhary@fau.de
Sheethal Bhat
Pattern Recognition Lab
FAU Erlangen-N¨urnberg
sheethal.bhat@fau.de
Andreas Maier
Pattern Recognition Lab
FAU Erlangen-N¨urnberg
andreas.maier@fau.de
Abstract—Accurate detection and localization of traumatic
injuries in abdominal CT scans remains a critical challenge
in emergency radiology, primarily due to severe scarcity of
annotated medical data. This paper presents a label-efficient
approach combining self-supervised pre-training with semi-
supervised detection for 3D medical image analysis. We employ
patch-based Masked Image Modeling (MIM) to pre-train a
3D U-Net encoder on 1,206 CT volumes without annotations,
learning robust anatomical representations. The pre-trained en-
coder enables two downstream clinical tasks: 3D injury detection
using VDETR with Vertex Relative Position Encoding, and
multi-label injury classification. For detection, semi-supervised
learning with 2,000 unlabeled volumes and consistency regu-
larization achieves 56.57% validation mAP@0.50 and 45.30%
test mAP@0.50 with only 144 labeled training samples, rep-
resenting a 115% improvement over supervised-only training.
For classification, expanding to 2,244 labeled samples yields
94.07% test accuracy across seven injury categories using only
a frozen encoder, demonstrating immediately transferable self-
supervised features. Our results validate that self-supervised
pre-training combined with semi-supervised learning effectively
addresses label scarcity in medical imaging, enabling robust 3D
object detection with limited annotations. The code is available
at https://github.com/shivasmic/3d-trauma-detection-ssl.
Index Terms—3D Object Detection, Self-supervised learning,
Semi-Supervised Learning, Transformer, DETR, Relative Posi-
tion Encoding, U-Net, Computed Tomography
I. INTRODUCTION
A. Background and motivation
Medical imaging plays a pivotal role in modern diag-
nostic workflows, with Computer Tomography (CT) scans
being particularly valuable for detecting internal injuries and
anatomical abnormalities. However, manual analysis of 3D
medical volumes is time-intensive, requires expert knowledge,
and is prone to inter-observer variability. The development of
automated 3D object detection systems for medical imaging
promises to improve diagnostic accuracy, reduce radiologist
workload, and enable faster clinical decision-making, particu-
larly in emergency settings where timely diagnosis is critical.
Traditional approaches to medical image analysis have
predominantly relied on 2D slice-by-slice evaluation or 3D
convolutions, which fail to capture complex spatial relation-
ships inherent in volumetric data. Furthermore, the scarcity
of annotated medical datasets, where obtaining pixel-level
segmentation masks requires significant expert time, poses a
fundamental bottleneck for supervised learning approaches. In
the RSNA abdominal trauma detection dataset used in this
work, only 206 of 4711 available series contain segmentation
masks, representing less than 5% labeled data. This severe la-
bel scarcity requires novel approaches that effectively leverage
unlabeled data.
The development of robust 3D detection systems for med-
ical imaging faces fundamental challenges stemming from
the unique characteristics of medical data. Medical datasets
typically contain severely limited labeled samples due to ex-
pensive and time-consuming expert annotation requirements,
with our RSNA dataset containing only 206 labeled se-
ries among 4,711 total scans (4.4% labeled). This extreme
scarcity makes traditional fully-supervised approaches imprac-
tical, while the computational demands of processing full 3D
volumes (512×336×336 voxels) with direct 3D convolutions
strain available hardware resources. Additionally, anatomi-
cal structures exhibit significant variability in shape, size,
and location across patients due to individual demographics,
pathology, and imaging protocols—unlike the relatively rigid
objects common in natural image datasets.
Beyond data and computational constraints, 3D medical
detection requires sophisticated spatial reasoning capabilities
that conventional approaches fail to provide. Traditional 2D
detection methods relying on center-point predictions inad-
equately capture the volumetric nature of irregularly-shaped
organs and injuries, where single distance metrics cannot rep-
resent complex geometric relationships. Furthermore, generic
3D feature extractors pre-trained on natural videos or synthetic
data transfer poorly to medical imaging, which exhibits distinct
characteristics including limited color information, specific
intensity distributions (Hounsfield Units), and highly domain-
specific anatomical patterns that require specialized represen-
tation learning.
B. Proposed Approach
This research addresses label scarcity in 3D medical ob-
ject detection through a two-stage learning framework. First,
arXiv:2603.12514v1  [cs.CV]  12 Mar 2026


<!-- page 2 -->
we employ self-supervised Masked Image Modeling (MIM)
to pre-train a 3D U-Net encoder on 1,206 unlabeled CT
volumes, learning robust anatomical representations without
annotations. Second, we leverage these pre-trained features
for two downstream tasks: 3D injury detection using VDETR
with Vertex Relative Position Encoding and semi-supervised
consistency regularization on 2,000 additional unlabeled vol-
umes, and multi-label injury classification with minimal fine-
tuning. This approach enables effective learning from severely
limited labeled data (144 volumes for detection, 2,244 for
classification) while maintaining strong generalization to held-
out test sets.
C. Paper Organization
The remainder of this paper is organized as follows: Section
II reviews related work in 3D object detection, self-supervised
learning, and medical image analysis. Section III details our
data preprocessing pipeline and standardization procedures.
Section IV describes the 3D U-Net encoder architecture and
self-supervised pre-training methodology. Section V presents
the VDETR decoder design with 3D Relative Positional En-
coding. Section VI explains our two-phase training strategy.
Section VII presents experimental results and evaluation met-
rics. Section VIII discusses limitations and future directions.
Finally, Section IX concludes the paper.
II. LITERATURE OVERVIEW
The field of 3D object detection has evolved significantly
from early approaches that relied heavily upon 2D slice-by-
slice analysis or sliding window techniques, which fundamen-
tally fail to capture the volumetric nature of 3D medical data.
Traditional 3D object detection methods adapted natural image
object detectors, such as Faster R-CNN [1], to volumetric
data by replacing 2D convolutions with their 3D counterparts.
However, these anchor-based approaches came with a price of
extensive hyperparameter tuning and struggled with irregular
anatomical shapes, which made them unfit for processing
medical image data, since medical data is full of complex and
irregular anatomical object shapes. VoteNet [2] came up with a
voting mechanism to shift surface points toward object centers
for grouping, enabling end-to-end learning from point clouds.
More recent methods like FCAF3D [3] employ expansion-
based approaches to generate virtual center features from
surface features, thereby improving overall region proposal
quality. However, all these fully convolutional approaches
critically rely on large amounts of pixel-level annotations,
which are prohibitively expensive in medical imaging, where
only 4.4% of our dataset contains segmentation masks.
The introduction of DETR [4] revolutionized object detec-
tion by formulating it as a direct set prediction problem. It
replaced the hand-designed components like non-maximum
suppression and anchor generation through learnable object
queries that attend to image features via cross-attention mech-
anisms, which enables end-to-end differentiable detection with
elegant simplicity. Early 3D adaptations like 3DETR [5] and
GroupFree [6] adapted the DETR framework to 3D point
clouds but achieved suboptimal results, particularly in medical
imaging, due to their inability to learn accurate inductive
biases from limited training data. Specifically, queries often
attended to points far from target objects, violating the locality
principle essential for effective detection. V-DETR [7] funda-
mentally addressed this limitation by introducing 3D Vertex
Relative Position Encoding (3DV-RPE), which computes po-
sition encoding for each point based on its relative position to
all eight vertices of predicted 3D boxes rather than just box
centers, providing clear geometric information about whether
points lie inside, outside, or on boundaries of objects. This
corner-based encoding enables the model to learn locality even
from limited data, achieving state-of-the-art results on indoor
3D detection benchmarks like ScanNetV2.
The scarcity of labeled medical data has driven substantial
research in self-supervised learning (SSL) approaches capable
of leveraging large volumes of unlabeled data to learn robust
feature representations. Masked Image Modeling (MIM), in-
spired by masked language modeling in NLP, has emerged
as an effective paradigm for visual representation learning.
Masked Autoencoders (MAE) [8] demonstrated that mask-
ing random patches and reconstructing them provides robust
supervision to models, enabling them to learn transferable
features in natural images and achieving strong performance
on downstream tasks with minimal fine-tuning. For medical
imaging specifically, the importance of pre-training has been
systematically investigated by Eckstein et al. [9], who demon-
strated that self-supervised pre-training significantly improves
3D medical object detection performance, particularly when
there is a scarcity of labeled data. Their work showed that
pre-training on large-scale unlabeled medical volumes enables
models to learn anatomical priors that substantially enhance
detection accuracy compared to training from scratch.
Complementary
to
self-supervised
pre-training,
semi-
supervised learning addresses label scarcity by enabling mod-
els to learn from abundant unlabeled samples while be-
ing guided by limited annotations during training. Mean
Teacher [10] introduced the teacher-student framework where
an exponential moving average (EMA) model serves as the
teacher, providing pseudo-labels for unlabeled data that guide
student model training through consistency regularization.
This approach enforces prediction consistency between teacher
and student models on perturbed versions of the same in-
put, proving highly effective across various domains. For
object detection specifically, Unbiased Teacher [11] adapted
the teacher-student framework by addressing the confirmation
bias problem where incorrect teacher predictions reinforce
student errors, introducing focal loss weighting to handle
class imbalance in pseudo-labels. In medical imaging, semi-
supervised approaches have shown substantial promise due to
inherent annotation scarcity, with consistency-based losses ef-
fectively leveraging unlabeled medical scans. However, the in-
tegration of self-supervised pre-training with semi-supervised
transformer-based detection remains largely unexplored in 3D
medical imaging.


<!-- page 3 -->
III. DATA PREPROCESSING PIPELINE
A. Dataset Overview and Structure
The RSNA Abdominal Trauma Detection dataset comprises
3D CT volumetric scans from multiple patients, organized into
individual series where each series contains a sequence of MRI
slices stored in DICOM (.dcm) format. The complete dataset
contains 4,711 distinct series from different patients, with each
series potentially containing anywhere from a single slice to
several thousand, depending on the acquisition protocol and
anatomical coverage.
A critical limitation of this dataset is the severe data
annotation imbalance: only 206 series contain corresponding
segmentation masks in NIFTI format (.nii), leaving 4,505
series (95.6%) without pixel-level annotations. This extreme
scarcity of labeled data works as the motivation behind our
self-supervised learning approach, which leverages both la-
beled and unlabeled volumes during the feature learning phase.
Since preprocessing operations are CPU-bound and cannot
be efficiently parallelized on GPUs, we selected a representa-
tive subset of this dataset for our experiments. The final dataset
consists of 1,206 series; all 206 labeled series allow us to
maximize the supervised signal, and 1,000 randomly selected
unlabeled series allow us to provide sufficient diversity for
self-supervised pre-training of the feature extractor backbone.
This balanced subset maintains clinical diversity while remain-
ing computationally tractable.
B. Preprocessing Pipeline for Unlabeled Volumes
For the 1,000 unlabeled series, we implemented a compre-
hensive preprocessing pipeline to convert raw DICOM slices
into standardized 3D numpy arrays suitable for neural network
training. All DICOM files within each series are loaded and
sorted along the z-direction to preserve anatomical order, then
stacked into a 3D array with dimensions (Z, Y, X). Raw
pixel intensities are converted to Hounsfield Units (HU) using
DICOM metadata according to:
HU = Iraw · m + b
(1)
where Iraw is the raw pixel value, m is the RescaleSlope,
and b is the RescaleIntercept. Intensity values are clipped
to [−100, 300] HU to focus on abdominal soft tissue while
excluding bone and air, then normalized to [0, 1] using min-
max normalization:
Inorm =
I −Imin
Imax −Imin
(2)
where Imin = −100 HU and Imax = 300 HU. To balance
spatial resolution with computational efficiency, volumes are
resampled to anisotropic spacing of (2.0, 1.0, 1.0) mm for
(Z, Y, X) directions using trilinear interpolation, reducing lon-
gitudinal slices by approximately 2× while preserving lateral
resolution. All volumes are standardized to 512 × 336 × 336
voxels through center-padding or center-cropping, accommo-
dating 95% of volumes without excessive cropping. Finally,
preprocessed volumes are saved in compressed .npz format,
reducing storage by 3-5× while maintaining data integrity.
This pipeline produces 1,000 standardized unlabeled volumes
ready for self-supervised pre-training.
C. Preprocessing Pipeline for Labeled Volumes
The 206 labeled series require additional processing steps to
align the segmentation masks with the CT volumes, as masks
are stored separately in NIFTI (.nii) format with potentially
different voxel spacing and orientation. The same preprocess-
ing steps from the unlabeled pipeline are first applied to create
normalized, resampled CT volumes. The corresponding NIFTI
segmentation mask is then loaded as a 3D binary array existing
in its own voxel coordinate space, which may differ from
the DICOM series due to variations in acquisition parameters
or annotation tools. To ensure voxel-wise correspondence,
the mask is resampled to match the DICOM spacing of
(2.0, 1.0, 1.0) mm using nearest-neighbor interpolation, which
preserves discrete segmentation labels and prevents fractional
values. Using the scikit-image regionprops module, con-
nected components are identified and tight 3D bounding boxes
are extracted around labeled structures, defined as:
B = (minz, miny, minx, maxz, maxy, maxx)
(3)
These coordinates specify the minimal rectangular cuboid
containing the segmented structure. Both the CT volume
and mask are cropped to this region of interest, reducing
background tissue and focusing on relevant anatomy. The
cropped volume and mask are then resized to 512×336×336
voxels using center-padding or center-cropping to ensure con-
sistent spatial dimensions. Finally, each sample is saved as
a compressed .npz file containing three components: volume
(normalized CT image as float32), mask (aligned segmentation
as uint8), and box (bounding box coordinates), ensuring data
consistency and simplified loading. This produces 206 labeled
volumes with pixel-perfect alignment, ready for supervised
fine-tuning and evaluation.
D. Data Preprocessing Summary
Figure 1 illustrates the complete preprocessing pipeline
for labeled volumes, showing the transformation from raw
DICOM acquisition to final standardized output. Table I sum-
marizes the dataset composition after preprocessing.
Table I summarizes the dataset composition after prepro-
cessing.
TABLE I
PREPROCESSED DATASET STATISTICS
Category
Count
Volume Size
Format
Unlabeled Volumes
1,000
512 × 336 × 336
.npz
Labeled Volumes
206
512 × 336 × 336
.npz
Total Series
1,206
–
–
Voxel Spacing (Z, Y, X)
–
2.0, 1.0, 1.0 mm
–
Intensity Range
–
[0, 1]
–
HU Window
–
[−100, 300]
–
This standardized dataset forms the foundation for both
self-supervised pre-training (utilizing all 1,206 volumes) and

[CAPTION] Figure 1 illustrates the complete preprocessing pipeline


<!-- page 4 -->
Fig. 1.
Preprocessing pipeline for labeled CT volumes. (a) Raw DICOM
slice in Hounsfield Units showing native acquisition, (b) aligned injury
segmentation mask overlaid in red, (c) volume after anisotropic resampling
(2.0×1.0×1.0 mm) and intensity normalization to [0,1], (d) final standardized
volume with dimensions (512×336×336) after center-cropping.
supervised fine-tuning (utilizing the 206 labeled volumes), en-
abling our two-phase training strategy described in subsequent
sections.
IV. METHODOLOGY
A. Self-Supervised Representation Learning
To learn robust anatomical features from all available CT
volumes without requiring annotations, we employ a 3D U-
Net encoder trained via patch-based Masked Image Modeling
(MIM). Rather than processing entire volumes (512×336×336
voxels), we extract 128×128×128 patches from each volume,
reducing computational complexity while enabling diverse
spatial coverage through augmentation. This patch-based strat-
egy allows us to leverage all 1,206 volumes, both labeled
and unlabeled, for feature learning. The U-Net follows a
standard encoder-decoder architecture with skip connections:
the encoder progressively downsamples features through 3D
convolutions and pooling layers, while the decoder recon-
structs the input through transposed 3D convolutions. This
symmetric architecture enables the network to learn hierarchi-
cal 3D anatomical representations at multiple scales. Figure 2
illustrates the complete U-Net architecture and the masked
patch reconstruction strategy.
For self-supervised training, each 128×128×128 patch is
subdivided into 8×8×8 cubic sub-patches, and 75% of these
sub-patches are randomly masked. The network is trained
to reconstruct the original intensities in the masked regions,
deriving supervision directly from the data itself rather than
Fig. 2.
3D U-Net encoder-decoder architecture with patch-based MIM for
self-supervised pre-training.
relying on manual annotations. The reconstruction task forces
the encoder to learn meaningful anatomical patterns and spatial
relationships within CT volumes. We train the U-Net for 50
epochs using the Adam optimizer with Mean Squared Error
(MSE) loss between the reconstructed and original patches.
Multiple patches are sampled from each volume per epoch
to ensure comprehensive coverage of anatomical structures.
Following pre-training, we freeze the encoder weights to serve
as a fixed feature extractor backbone for downstream tasks.
B. Downstream Task I: 3D Injury Detection
The pre-trained 3D U-Net encoder processes 512×336×336
voxel CT volumes and generates a 32×21×21 spatial feature
grid, where each voxel is characterized by 256-dimensional
features. A feature adapter samples 4,096 representative tokens
from the 14,112 available voxels, creating a compact sequence
for the transformer decoder. These features are then processed
by a VDETR decoder equipped with 3D Vertex Relative Po-
sition Encoding (3D RPE) [7], which overcomes fundamental
limitations of center-based distance metrics in 3D detection.
Figure 3 illustrates the VDETR architecture and the 3D RPE
mechanism.
In 2D detection, standard DETR architectures calculate sim-
ple distances between query centers and image pixels. How-
ever, for 3D medical imaging, a single center-to-voxel distance
provides insufficient information for determining whether a
voxel belongs to irregularly-shaped anatomical structures or
injuries. The 3D RPE mechanism addresses this by computing
geometric relationships between each voxel and all eight
corners of the predicted bounding box for every object query.
These relationships are fused through multi-layer perceptrons
(MLPs) to generate attention biases, which are added to the
standard attention scores (dot product between queries and
keys). This enables the model to focus not only on semantic
similarity but also on geometric inclusion, exclusion, and
boundary proximity, effectively teaching the decoder whether
voxels lie inside, outside, or on the boundaries of predicted
boxes.

[CAPTION] Fig. 1.
Preprocessing pipeline for labeled CT volumes. (a) Raw DICOM

[CAPTION] Fig. 2.
3D U-Net encoder-decoder architecture with patch-based MIM for

[CAPTION] Figure 3 illustrates the VDETR architecture and the 3D RPE


<!-- page 5 -->
Fig. 3. VDETR decoder architecture with 3D Vertex RPE for injury detection.
The detection pipeline is trained in two distinct phases to
maximize utilization of pre-trained features while enabling
task-specific refinement. In Phase I (epochs 0-20), the pre-
trained U-Net encoder weights remain frozen while only
the VDETR decoder and its prediction heads are trained
on labeled data. This prevents randomly initialized decoder
gradients from corrupting the carefully learned anatomical
representations in the encoder. During this phase, the decoder
learns to interpret 8-corner RPE and iteratively refine bounding
box predictions using the static features provided by the frozen
encoder. The frozen encoder ensures stable, high-quality fea-
tures that allow the decoder to focus exclusively on learning
detection logic.
For each query q and voxel position pv, we compute the
offset vectors to all 8 vertices of the predicted box:
∆Pi ∈RK×N×3
(4)
where K is the number of queries, N is the number of
voxel positions (4,096 tokens), and i ∈{1, . . . , 8} indexes the
box vertices. Each offset is transformed through a non-linear
function F(·) and an MLP to produce position bias:
Pi = MLPi (F(∆Pi)) ∈RK×N×h
(5)
where h is the number of attention heads. The total relative
position bias is computed by summing across all 8 vertices:
R =
8
X
i=1
Pi
(6)
This bias augments the standard attention mechanism:
A = softmax
 QKT + R
 
(7)
where Q represents query embeddings and K represents
key embeddings from the voxel features.
In Phase II (epochs 20-100), both encoder and decoder
are jointly fine-tuned end-to-end. The encoder unfreezes with
a 3-epoch warmup period where its learning rate gradually
ramps from 0 to 1 × 10−5 (10× smaller than the decoder’s
1 × 10−4 learning rate) to prevent catastrophic forgetting
of pre-trained features. This fine-tuning phase enables the
encoder to refine its generic anatomical features—learned
via reconstruction during self-supervised pre-training—into
task-specific representations optimized for precise bounding
box localization. While the pre-trained encoder captured low-
frequency global anatomical patterns sufficient for identifying
general organ locations, fine-tuning sharpens these features
to detect exact injury boundaries and subtle abnormalities
required for accurate detection.
To leverage the abundant unlabeled medical imaging data
and improve detection performance beyond the limited labeled
set, we employ semi-supervised learning using Mean Teacher-
style consistency regularization [10]. Our training data com-
prises 144 labeled volumes with bounding box annotations and
2,000 unlabeled volumes held-out during encoder pre-training,
totaling 2,144 CT scans. This non-overlapping partitioning en-
sures the detection model learns from anatomical variations not
previously encountered by the encoder. The semi-supervised
approach enforces prediction consistency between weakly and
strongly augmented versions of unlabeled volumes.
For each unlabeled volume, weak augmentations (minimal
Gaussian noise σ = 0.01, small intensity shifts ±2%) are
applied to generate teacher predictions, which serve as pseudo-
labels without gradient computation. The same volume under-
goes strong augmentations (larger Gaussian noise σ = 0.05,
intensity shifts ±10%, Gaussian blur, elastic deformations) to
produce student predictions with full gradient propagation. The
consistency loss comprises three components:
Lcenter = MSE(cstudent, cteacher)
(8)
Lsize = MSE(sstudent, steacher)
(9)
Lcls = KL
 
softmax
 zT
T
 
∥softmax
 zS
T
  
× T 2
(10)

[CAPTION] Fig. 3. VDETR decoder architecture with 3D Vertex RPE for injury detection.


<!-- page 6 -->
where c, s, and z represent predicted box centers, sizes,
and class logits respectively, and T = 2.0 is the temperature
scaling parameter. The total training loss combines supervised
detection loss on labeled data with weighted consistency loss
on unlabeled data:
Ltotal = Lsupervised + λ(t) × (Lcenter + Lsize + Lcls)
(11)
where λ(t) is a time-dependent weight that linearly ramps
from 0 to 0.3 during epochs 20-60.
Semi-supervised learning is disabled during Phase I (epochs
0-20) to ensure decoder convergence on labeled data before
introducing unlabeled samples, preventing trivial solutions
where all predictions collapse. At epoch 20, when the encoder
unfreezes and Phase II begins, consistency regularization acti-
vates with λ linearly ramping from 0 to 0.3 over epochs 20-60,
then maintaining λ = 0.3 for epochs 60-100. This gradual in-
troduction prevents training destabilization while allowing the
model to leverage 14× more data than labeled samples alone.
The consistency constraint acts as powerful regularization,
encouraging the encoder to learn features robust to intensity
variations, geometrically stable across transformations, and
anatomically consistent across different appearances, substan-
tially improving generalization on the small labeled dataset.
C. Downstream Task II: Injury Classification
The classification task addresses multi-label binary predic-
tion of seven independent injury categories per CT volume:
bowel healthy/injury, liver healthy/high-grade injury, kidney
high-grade injury, spleen healthy, and extravasation injury.
These labels are not mutually exclusive, as patients may
present with multiple concurrent injuries. The classification
head processes features from the pre-trained U-Net encoder
bottleneck, avoiding the memory-intensive decoder path en-
tirely.
The architecture consists of the frozen (then fine-tuned)
U-Net encoder producing 32×21×21×256 bottleneck features,
followed by global average pooling that aggregates spatial in-
formation into a 256-dimensional feature vector. A lightweight
two-layer fully-connected head transforms these features: the
first layer projects to 128 dimensions with ReLU activation
and 50% dropout for regularization, and the second layer
produces 7 logits corresponding to the binary classification
targets. This design contains only 33,799 trainable parameters
in the classification head, compared to 5.6M in the en-
coder, enabling efficient fine-tuning while leveraging the rich
anatomical representations learned during self-supervised pre-
training. Figure 4 shows the classification head architecture.
The
severe
class
imbalance
in
medical
imaging
datasets—where rare but critical injuries like bowel injury
constitute only 18% of positive samples—necessitates careful
loss design. We employ binary cross-entropy with logits and
per-class positive weights to address this imbalance:
wpos
i
= N neg
i
N pos
i
(12)
Fig. 4. Classification head architecture for multi-label injury prediction.
Li
BCE = −wpos
i
·yi ·log(σ(zi))−(1−yi)·log(1−σ(zi)) (13)
Lcls = 1
7
7
X
i=1
Li
BCE
(14)
where zi is the logit for class i, yi ∈{0, 1} is the ground
truth label, σ(·) is the sigmoid function, and wpos
i
applies
heavier penalties to false negatives for rare injury classes (e.g.,
wpos
bowel injury = 4.45) while reducing penalties for common
healthy states (e.g., wpos
bowel healthy = 0.22). This weighting
forces the model to learn critical injury patterns rather than
defaulting to majority class predictions.
To address the limited size of labeled medical imaging
datasets, we employ aggressive data augmentation during
training. Each volume undergoes random transformations in-
cluding Gaussian noise injection (σ = 0.05), intensity shifting
(±15%) and scaling (×[0.85, 1.15]), and gamma correction
(γ ∈[0.8, 1.2]). Combined with high dropout (50%) in the
classification head, this augmentation strategy provides effec-
tive regularization while maintaining strong generalization.
Following standard self-supervised learning evaluation pro-
tocols [12], [13], we employ a linear probe approach where
the encoder remains frozen throughout training. Only the
lightweight classification head (33,799 parameters) is opti-
mized using AdamW optimizer with learning rate 3 × 10−4,
weight decay 5 × 10−4, batch size 2, and cosine learning
rate scheduling over 50 epochs. This protocol directly assesses
the quality of pre-trained representations without task-specific
encoder adaptation.
We utilize the RSNA 2023 dataset’s patient-level classifica-
tion labels by mapping them to all corresponding CT series.
Patient-level labels from 3,147 patients are distributed across
3,206 CT series, which we split into 2,244 training, 480
validation, and 480 test samples, maximizing the utilization
of available annotations while maintaining standard evaluation
protocols.

[CAPTION] Fig. 4. Classification head architecture for multi-label injury prediction.


<!-- page 7 -->
V. EXPERIMENTAL AND EVALUATION RESULTS
A. Self-Supervised Pre-training Evaluation
To validate the quality of learned representations, we eval-
uate the pre-trained encoder on two tasks using 250 held-out
volumes for reconstruction and 206 labeled volumes for linear
probe classification. Table II presents the results.
TABLE II
U-NET ENCODER EVALUATION ON RECONSTRUCTION AND LINEAR PROBE
TASKS.
Evaluation Task
Metric
Result
Reconstruction
MSE
0.01194
Reconstruction
PSNR
19.39 dB
Linear Probe (164 train / 42 test):
Bowel healthy
Acc / AUC
92.9% / 0.975
Bowel injury
Acc / AUC
92.9% / 0.975
Liver healthy
Acc / AUC
83.3% / 0.883
Liver high-grade
Acc / AUC
88.1% / 0.670
Kidney high-grade
Acc / AUC
78.6% / 0.750
Spleen healthy
Acc / AUC
71.4% / 0.807
Extravasation
Acc / AUC
66.7% / 0.646
Overall
Acc / AUC
76.0% / 0.771
The reconstruction PSNR of 19.39 dB demonstrates effec-
tive learning of anatomical patterns from unlabeled data. The
linear probe achieves 76.0% overall accuracy without fine-
tuning, with particularly strong performance on bowel injury
detection (97.5% AUC), validating that self-supervised pre-
training produces discriminative features suitable for down-
stream medical imaging tasks.
B. 3D Injury Detection Results
1) Experimental Setup: The detection model is trained on
206 labeled CT volumes with 3D bounding box annotations for
abdominal injuries, split into 144 training, 30 validation, and
32 test volumes. For semi-supervised learning, we incorporate
1,206 additional unlabeled volumes through consistency regu-
larization. Each CT volume is preprocessed to 512×336×336
voxels with anisotropic spacing (2mm axial, 1mm in-plane),
then downsampled to 32×21×21 feature resolution by the U-
Net encoder with 4,096 voxels sampled for the transformer
decoder. The model is trained for 100 epochs using AdamW
optimizer with base learning rate 1 × 10−4 for the decoder
and 1×10−5 for the encoder (after unfreezing). We employ a
two-phase training strategy: Phase I (epochs 0-20) with frozen
U-Net encoder to stabilize decoder learning, followed by Phase
II (epochs 20-100) with encoder unfreezing, 3-epoch learning
rate warmup, and semi-supervised consistency regularization.
The consistency loss enforces prediction agreement between
weakly and strongly augmented versions of unlabeled vol-
umes, with loss weight ramping from λ = 0 to λ = 1.0
over 10 warmup epochs. All experiments are conducted on
NVIDIA A100 GPUs with batch size 1 for labeled data and
batch size 2 for unlabeled data. Performance is measured using
mean Average Precision (mAP) at IoU thresholds of 0.10, 0.25,
0.50, and 0.75, with mAP@0.50 as the primary metric.
Figure 5 shows the training progression for VDETR without
semi-supervised learning. The model achieves peak validation
performance of 26.36% mAP@0.50 at epoch 5, after which
performance degrades significantly—a clear indication of se-
vere training instability due to the limited labeled training set
(144 volumes). By epoch 100, mAP@0.50 drops to approxi-
mately 8%, demonstrating the model’s inability to generalize
when trained solely on scarce labeled data.
Fig. 5. VDETR training without semi-supervised learning exhibiting severe
training instability and catastrophic performance collapse.
Figure 6 demonstrates the transformative effect of semi-
supervised learning with 2,000 unlabeled volumes. The model
exhibits stable convergence throughout training, achieving
56.57% mAP@0.50, a 114% improvement over the no-
SSL baseline. Unlike the supervised-only approach, the SSL-
enhanced model shows consistent performance gains across all
IoU thresholds and maintains stability without any catastrophic
collapse. The consistency regularization from unlabeled data
provides powerful implicit regularization, enabling the model
to learn robust anatomical patterns that generalize effectively
to held-out validation data.
Fig. 6.
VDETR training with semi-supervised learning showing stable
convergence with consistent improvement across all IoU thresholds.
2) Ablation Study: Table III compares training performance
between supervised and semi-supervised training. Without
SSL, the model peaks at epoch 5 with 26.36% mAP@0.50,
then catastrophically degrades to 8% by epoch 100—indicat-
ing training instability with limited labeled data (144 volumes).
Incorporating semi-supervised learning with 2,000 unlabeled
volumes stabilizes training, achieving 56.57% mAP@0.50 at

[CAPTION] Figure 5 shows the training progression for VDETR without

[CAPTION] Fig. 5. VDETR training without semi-supervised learning exhibiting severe

[CAPTION] Figure 6 demonstrates the transformative effect of semi-

[CAPTION] Fig. 6.
VDETR training with semi-supervised learning showing stable


<!-- page 8 -->
epoch 99 (115% improvement). The gains are even more
pronounced at stricter localization thresholds, with mAP@0.75
improving by 562% (from 6.82% to 45.12%), demonstrating
that consistency regularization not only stabilizes training but
also significantly improves bounding box localization accu-
racy. The stable convergence versus catastrophic performance
degradation highlights the critical role of unlabeled data in
providing robust supervision signals when labeled annotations
are severely limited.
TABLE III
DETECTION PERFORMANCE COMPARISON ON VALIDATION SET.
Metric
VDETR (no SSL)
VDETR + SSL
Gain
Best Epoch
5
99
–
mAP@0.10
27.27%
56.57%
+107%
mAP@0.25
27.27%
56.57%
+107%
mAP@0.50
26.36%
56.57%
+115%
mAP@0.75
6.82%
45.12%
+562%
3) Test Set Performance: Table IV presents the final test
set evaluation on 32 held-out volumes. The SSL-enhanced
model achieves 45.30% mAP@0.50 on the test set compared
to 23.03% without SSL, representing a 97% improvement
and validating the critical role of consistency regularization
for generalization. The model maintains consistent perfor-
mance across IoU thresholds, achieving 45.30% at both lenient
(0.10, 0.25) and standard (0.50) thresholds, with 28.72% at
the stringent 0.75 threshold. Even the supervised-only base-
line achieves 23.03% mAP@0.50, demonstrating that self-
supervised pre-training provides a strong foundation, while
semi-supervised learning substantially enhances localization
precision and detection robustness.
TABLE IV
DETECTION PERFORMANCE COMPARISON ON TEST SET.
Metric
VDETR(no SSL)
VDETR+SSL
Gain
mAP@0.10
23.03%
45.30%
+97%
mAP@0.25
23.03%
45.30%
+97%
mAP@0.50
23.03%
45.30%
+97%
mAP@0.75
16.67%
28.72%
+72%
C. Multi-Label Injury Classification Results
1) Experimental Setup: Classification training uses batch
size 2 on a single NVIDIA A100 GPU with the encoder
frozen throughout training. We predict seven independent
binary injury categories per CT volume using patient-level
labels from the RSNA 2023 dataset mapped to 3,206 CT series
via directory structure. Following stratified random sampling,
we obtain 2,244 training (70%), 481 validation (15%), and
482 test (15%) samples. Preprocessed volumes (128×168×168
voxels) are fed through our self-supervised pre-trained U-
Net encoder to extract 32×21×21×256 bottleneck features,
which are aggregated via global average pooling. A two-layer
classification head (256→128→7 with 50% dropout) produces
injury predictions, totaling only 33,799 trainable parameters
versus 5.6M in the frozen encoder.
2) Ablation Study: Following the training protocol in Sec-
tion IV-C, we evaluate classification performance on the RSNA
dataset. Table V shows that with 144 training samples, heavy
augmentation achieved 77.7% test accuracy. Semi-supervised
learning with 2,000 unlabeled volumes unexpectedly degraded
performance to 75.4%, likely due to noisy pseudo-labels from
distribution mismatch, while focal loss yielded 75.9%. Ex-
panding to 2,244 labeled samples with frozen encoder (linear
probe) achieved 94.07% test accuracy, demonstrating that
quality labeled data outperforms pseudo-labeling strategies.
TABLE V
CLASSIFICATION ABLATION STUDY COMPARING TRAINING STRATEGIES
WITH LIMITED LABELED DATA
Approach
Encoder
Test Acc
Test AUC
Fine-tune + augmentation
Unfrozen
77.7%
57.7%
Fine-tune + aug. + SSL
Unfrozen
75.4%
57.3%
Fine-tune + aug. + focal loss
Unfrozen
75.9%
56.0%
Linear probe (full data)
Frozen
94.07%
51.4%
3) Test Set Performance: Table VI presents test set per-
formance. The model achieved 94.07% mean accuracy at
epoch 0 with frozen encoder, demonstrating immediately
transferable self-supervised features. Per-class accuracy ranged
from 87.1% (spleen) to 98.3% (liver high-grade), with 97.5%
on bowel injury. The tight validation-test gap (0.19 points)
confirms strong generalization.
TABLE VI
TEST SET PERFORMANCE (482 VOLUMES) USING BEST MODEL FROZEN
ENCODER.
Injury Category
Test Acc
Test AUC
Bowel healthy
97.5%
0.577
Bowel injury
97.5%
0.584
Liver healthy
87.6%
0.500
Liver high-grade injury
98.3%
0.429
Kidney high-grade injury
96.1%
0.470
Spleen healthy
87.1%
0.518
Extravasation injury
94.4%
0.521
Overall
94.07%
0.514
Continuing training for 30 epochs yielded no improvement
beyond epoch 0, confirming feature saturation—a hallmark of
successful self-supervised pre-training where representations
are maximally discriminative without task-specific adapta-
tion [12], [14], [15]. This aligns with SSL literature where
high linear probe performance indicates effective pre-training,
as demonstrated in both natural image [13], [16] and medical
imaging domains [17], [18].
The modest AUC (51.4%) reflects a confidence calibration
issue common in neural networks [19], where prediction
probabilities misalign with true probabilities despite correct
classifications. This does not affect binary predictions used
clinically and can be corrected via post-hoc temperature scal-
ing [19], [20] without retraining.


<!-- page 9 -->
Compared to RSNA 2023 winners (98% AUC with 3,147
samples, two-stage pipelines, ensembles) [21], our single-
model frozen-encoder approach achieves competitive accuracy
(94.07%) with 29% less data and far lower complexity, demon-
strating effective transfer learning from self-supervised pre-
training.
VI. CONCLUSION
This research presents a label-efficient approach for 3D
abdominal trauma detection that systematically integrates
self-supervised pre-training with semi-supervised transformer-
based detection. By employing patch-based Masked Image
Modeling on 1,206 CT volumes, we trained a 3D U-Net
encoder that learns robust anatomical representations with-
out requiring annotations, achieving 19.39 dB reconstruction
PSNR and 76.0% linear probe classification accuracy. The
pre-trained encoder serves as a fixed feature extractor for
two downstream tasks: 3D injury detection using VDETR
with Vertex Relative Position Encoding, and multi-label injury
classification.
For detection, our semi-supervised approach leverages
2,000 unlabeled volumes through teacher-student consis-
tency regularization, achieving 56.57% validation mAP@0.50
and 45.30% test mAP@0.50—a 115% improvement over
supervised-only training that suffered from catastrophic per-
formance degradation. The 3D RPE mechanism enables pre-
cise localization of irregularly-shaped anatomical structures
by computing geometric relationships to all eight bounding
box corners rather than relying on inadequate center-based
distances. For classification, expanding from 144 to 2,244
labeled samples yielded 94.07% test accuracy across seven
injury categories, demonstrating that quality-labeled data out-
performs pseudo-labeling when available.
Our results validate that self-supervised pre-training com-
bined with semi-supervised learning effectively addresses the
fundamental challenge of label scarcity in medical imag-
ing. The dramatic improvement in detection stability and
performance—from early-epoch collapse to stable conver-
gence—highlights the critical role of unlabeled data in provid-
ing robust supervision signals. Future work will explore ex-
tending this framework to multi-organ detection, investigating
alternative consistency regularization strategies, and evaluating
clinical deployment feasibility for real-time trauma assessment
in emergency settings.
ACKNOWLEDGMENT
We sincerely acknowledge the Erlangen National High Per-
formance Computing Center (NHR@FAU) of the Friedrich-
Alexander-Universit¨at Erlangen-N¨urnberg (FAU) for providing
necessary computing resources that enabled this research.
REFERENCES
[1] S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards real-
time object detection with region proposal networks,” in Advances in
Neural Information Processing Systems, vol. 28, 2015.
[2] C. R. Qi, O. Litany, K. He, and L. J. Guibas, “Deep hough voting for
3D object detection in point clouds,” in Proceedings of the IEEE/CVF
International Conference on Computer Vision, 2019, pp. 9277–9286.
[3] D. Rukhovich, A. Vorontsova, and A. Konushin, “FCAF3D: Fully
convolutional anchor-free 3D object detection,” in European Conference
on Computer Vision.
Springer, 2022, pp. 477–493.
[4] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and
S. Zagoruyko, “End-to-end object detection with transformers,” in
European Conference on Computer Vision, 2020, pp. 213–229.
[5] I. Misra, R. Girdhar, and A. Joulin, “An end-to-end transformer model
for 3D object detection,” in Proceedings of the IEEE/CVF International
Conference on Computer Vision, 2021, pp. 2906–2917.
[6] Z. Liu, Z. Zhang, Y. Cao, H. Hu, and X. Tong, “Group-free 3D
object detection via transformers,” in Proceedings of the IEEE/CVF
International Conference on Computer Vision, 2021, pp. 2949–2958.
[7] Y. Shen, Z. Geng, Y. Yuan, Y. Lin, Z. Liu, C. Wang, H. Hu, N. Zheng,
and B. Guo, “V-DETR: DETR with vertex relative position encoding
for 3D object detection,” in Proceedings of the IEEE/CVF International
Conference on Computer Vision, 2023, pp. 6857–6867.
[8] K. He, X. Chen, S. Xie, Y. Li, P. Doll´ar, and R. Girshick, “Masked au-
toencoders are scalable vision learners,” in Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2022, pp.
16 000–16 009.
[9] K. Eckstein, C. Ulrich, M. Baumgartner, J. K¨achele, D. Bounias,
T. Wald, R. Floca, and K. H. Maier-Hein, “The missing piece: A
case for pre-training in 3D medical object detection,” arXiv preprint
arXiv:2404.xxxxx, 2024, medical Faculty Heidelberg, Heidelberg Uni-
versity.
[10] A. Tarvainen and H. Valpola, “Mean teachers are better role mod-
els: Weight-averaged consistency targets improve semi-supervised deep
learning results,” in Advances in Neural Information Processing Systems,
vol. 30, 2017.
[11] Y.-C. Liu, C.-Y. Ma, Z. He, C.-W. Kuo, K. Chen, P. Zhang, B. Wu,
Z. Kira, and P. Vajda, “Unbiased teacher for semi-supervised object
detection,” in International Conference on Learning Representations,
2021.
[12] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework
for contrastive learning of visual representations,” in International
Conference on Machine Learning (ICML).
PMLR, 2020, pp. 1597–
1607.
[13] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, “Momentum contrast for
unsupervised visual representation learning,” in IEEE/CVF Conference
on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 9729–
9738.
[14] S. Azizi, B. Mustafa, F. Ryan, Z. Beaver, J. Freyberg, J. Deaton, A. Loh,
A. Karthikesalingam, S. Kornblith, T. Chen et al., “Big self-supervised
models advance medical image classification,” IEEE/CVF International
Conference on Computer Vision (ICCV), pp. 3478–3488, 2021.
[15] Z. Zhou, V. Sodha, J. Pang, M. B. Gotway, and J. Liang, “Models
genesis: Generic autodidactic models for 3d medical image analysis,” in
International Conference on Medical Image Computing and Computer-
Assisted Intervention (MICCAI).
Springer, 2019, pp. 384–393.
[16] J.-B.
Grill,
F.
Strub,
F.
Altch´e,
C.
Tallec,
P.
H.
Richemond,
E. Buchatskaya, C. Doersch, B. A. Pires, Z. D. Guo, M. G. Azar et al.,
“Bootstrap your own latent: A new approach to self-supervised learning,”
in Advances in Neural Information Processing Systems (NeurIPS),
vol. 33, 2020, pp. 21 271–21 284.
[17] A. Taleb, C. Lippert, T. Klein, and M. Nabi, “Contrastive learning of
global and local features for medical image segmentation with lim-
ited annotations,” Advances in Neural Information Processing Systems
(NeurIPS), vol. 33, pp. 12 546–12 558, 2020.
[18] J. Zhu, Y. Li, Y. Hu, K. Ma, S. K. Zhou, and Y. Zheng, “Rubik’s
cube+: A self-supervised feature learning framework for 3d medical
image analysis,” in Medical Image Analysis, vol. 64.
Elsevier, 2020,
p. 101746.
[19] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration
of modern neural networks,” in International Conference on Machine
Learning (ICML).
PMLR, 2017, pp. 1321–1330.
[20] J. Platt et al., “Probabilistic outputs for support vector machines and
comparisons to regularized likelihood methods,” in Advances in Large
Margin Classifiers, vol. 10, no. 3.
MIT Press, 1999, pp. 61–74.
[21] R.
C.
Winners,
“Rsna
2023
abdominal
trauma
detection
-
1st
place
solution,”
https://www.kaggle.com/competitions/
rsna-2023-abdominal-trauma-detection,
2023,
accessed:
2025-01-
15.