[2604.27985] Exploring Sparse Matrix Multiplication Kernels on the Cerebras CS-3

Exploring Sparse Matrix Multiplication Kernels on the Cerebras CS-3

Milan Shah

Affiliation:

North Carolina State University 
,  USA

email:  mkshah5@ncsu.edu

,

Sheng Di

Affiliation:

Argonne National Laboratory 
,  USA

email:  sdi1@anl.gov

and

Michela Becchi

Affiliation:

North Carolina State University 
,  USA

email:  mbecchi@ncsu.edu

Abstract.

In recent years, novel AI accelerators have emerged as promising alternatives to GPU for AI model training and inference tasks. One such accelerator, the Cerebras CS-3, achieves strong performance on large model training as well as scientific applications like molecular dynamics simulations. While dense compute workloads have been thoroughly explored for the CS-3, its potential for sparse workloads has not been fully examined. Applications requiring sparse linear algebra kernels, such as GNNs, linear solvers, and recommendation systems, could achieve good performance on a dataflow accelerator like the CS-3.

In this work, we explore two key sparse linear algebra kernels, sparse-dense matrix multiplication (SpMM) and sampled dense-dense matrix multiplication (SDDMM), on the Cerebras CS-3. We propose low-level CS-3 kernel designs for these operations and optimize our designs to improve I/O performance, memory footprint, and scalability to large matrices. Our evaluation examines memory footprint and SpMM/SDDMM speedup relative to CPU. The evaluation suggests that the CS-3 can outperform CPU by 100

×  \times

for SpMM with 90% sparse matrices with performance improving as sparse matrix dimensionality increases. SDDMM on CS-3 can outperform CPU 20

×  \times

for 90% sparse matrices. We additionally find that as sparsity increases to beyond 99%, the CS-3 suffers from performance degradation that makes it slower than CPU for SpMM.

1.  Introduction

Novel AI accelerators have emerged in recent years as a tool to accelerate machine learning and AI workloads. These accelerators attempt to address the limitations of GPU architectures in the context of DNN training. While GPUs can enable massive SIMD parallelism across many cores, communication between on-chip and off-chip memory from kernel to kernel can bottleneck DNN training. Newer AI accelerators, such as the Cerebras CS-3  ( cerebras_product ) , SambaNova RDU  ( sn30_product ) , Graphcore IPU  ( ipu_product ) , and Intel Habana Gaudi  ( intel_gaudi ) , opt to greatly increase their chip’s on-chip memory capacity in an effort to reduce frequent memory exchanges with off-chip memory that include data structures like activations and gradients in DNN training. Additionally, accelerators vary their architecture from GPU: the Graphcore IPU is composed of 1472 parallel compute units that can execute their own instruction stream (Multiple-Instruction-Multiple-Data architecture), while the CS-3 and RDU are dataflow architectures, where computation is mapped to physical compute units and results from one compute unit are directly fed to the next unit via high-speed on-chip interconnects  ( cerebras_product ;  sn30_product ) . These accelerators can outperform the GPU for certain tasks and can achieve significantly higher training throughput for DNNs  ( cs2_matmul ;  seismic_cs2 ;  cs2_blog ;  cs2_benchmarking ) .

In this work, we explore the ability of the CS-3 to perform sparse-dense matrix multiplication ( SpMM ) and sampled dense-dense matrix multiplication ( SDDMM ) and seek optimizations that improve the time and memory efficiency of these kernels on the CS-3.
SpMM and SDDMM are popular sparse linear algebra kernels used in a wide variety of applications, such as pruned deep neural networks, graph neural networks, recommendation systems, scientific simulations, linear solvers, sparse attention in transformers, electronic design automation, and graph analytics  ( sparsifieddnnmodels ;  spmmlinearsolvers ;  sddmmusecases ;  gcnpaper ;  gat_paper ) .
We first identify the key challenges to using the existing PyTorch programming interface for sparse linear algebra kernels on the CS-3 from both programmer effort and system constraint perspectives. Using these challenges to guide design, we propose efficient SpMM and SDDMM kernels written in CSL, the CS-3 low-level programming interface.
Our SpMM and SDDMM kernels compute outputs using a compressed representation of the sparse matrix. Our key contributions are as follows:

(1)

We propose SpMM and SDDMM CSL kernels for efficient CS-3 execution. Our kernels operate on sparse matrices represented in sparse storage formats similar to CSR and SELLPACK.

(2)

SpMM is optimized to support multiple I/O channel data streaming from the host to the CS-3 as well as supporting larger matrices than a PyTorch implementation.

(3)

We evaluate our implementations on a variety of synthetically generated sparse matrices to showcase CS-3 performance comparison with CPU, discussing both the performance potential of CS-3 as well as bottlenecks of this hardware.

(4)

Our evaluation suggests that for large and denser matrices, SpMM performance on CS-3 can be 100

×  \times

faster than CPU for 90% sparse matrices. SDDMM performance is typically an order of magnitude faster than CPU, scaling in performance at the same rate as CPU for increasing density. As matrices become hyper-sparse, the CS-3 suffers from high host-device communication overhead relative to computation, making the CS-3 slower than GPU and approaching baseline CPU performance.

2.  Background

2.1.  SpMM and SDDMM

SpMM , or sparse-dense matrix multiplication, is a fundamental sparse linear algebra routine common in GNNs  ( gcnpaper ;  gat_paper ) , pruned deep neural networks  ( sparsifieddnnmodels ) , and linear solvers  ( spmmlinearsolvers ) . Formally, SpMM performs the following computation:

(1)

Y  =

A  ​  H

,

A  ∈

ℝ

K  ×  N

,

H  ∈

ℝ

N  ×  D

Y=AH,A\in\mathbb{R}^{K\times N},H\in\mathbb{R}^{N\times D}

In many applications (such as GNNs),

K  =  N

K=N

, thus

A  A

is square.

A  A

is the sparse matrix,

H  H

is the dense matrix, and

Y  Y

is the dense output matrix.

N  N

can vary significantly from application to application, but often is on the scale of millions to billions for real-world graphs in GNNs.

SDDMM , or sampled dense-dense matrix multiplication, is a kernel used in graph attention networks  ( gat_paper ) , recommendation systems, and scientific computing  ( sddmmusecases ) . Formally, SDDMM performs the following computation:

(2)

Y  =

A  ⊙

(

B  ​  C

)

,

A  ∈

ℝ

N  ×  N

,

B  ∈

ℝ

N  ×  D

,

C  ∈

ℝ

D  ×  N

Y=A\odot(BC),A\in\mathbb{R}^{N\times N},B\in\mathbb{R}^{N\times D},C\in\mathbb{R}^{D\times N}

As is the case with SpMM,

N  N

can be extremely large depending on the application, ranging from millions to billions. SDDMM essentially only computes output elements of

Y  Y

where there exists a nonzero in

A  A

.

2.2.  SpMM and SDDMM applications

To contextualize the importance of using sparse kernels like SpMM and SDDMM as opposed to a dense-dense matrix multiplication, we focus on one application of these kernels: graph neural networks.

Table 1 .

Popular graph benchmarks for GNNs. “Dense (GB)” is the dense adjacency matrix size in GB and “CSR (GB)” is the adjacency matrix size in GB represented with Compressed Sparse Row format.

Graph

Nodes

Edges

Dense (GB)

CSR (GB)

cora

2.71E+03

1.09E+04

2.73E-02

5.05E-05

pubmed

1.97E+04

1.08E+05

1.45E+00

4.77E-04

arxiv

1.69E+05

1.17E+06

1.07E+02

4.98E-03

products

2.45E+06

6.19E+07

2.23E+04

2.40E-01

For GNN, storing the sparse matrix is SpMM or SDDMM in a dense format can quickly exhaust many training devices’ memory, including GPU and the CS-3. For this reason, graphs are often stored in a sparse format, like compressed sparse row (CSR) format. Table

1

reports four common GNN benchmark graphs (“cora”  ( cora_dataset ) , “pubmed” ( pubmed ) , “arxiv” ( arxiv_m