<!-- page 1 -->
A VIRTUALLY CONNECTED PROBABILISTIC COMPUTER AS A
SOLVER FOR HIGHER-ORDER, DENSELY CONNECTED, OR
RECONFIGURABLE COMBINATORIAL OPTIMISATION PROBLEMS
A PREPRINT
Amy J. Searle1,*
Harry Youel1,2,3
Fredrik Hasselgren1,4
Annika Möslein1
Ramy Aboushelbaya1
Marko von der Leyen1,**
1Quantum Dice Limited, 264 Banbury Road,
Oxford, OX2 7DY, United Kingdom
2London Centre for Nanotechnology, University College London,
London, WC1H 0AH, United Kingdom
3Department of Physics and Astronomy, University College London,
London, WC1E 6BT, United Kingdom
4Mathematical Institute, University of Oxford,
Oxford, OX2 6GG, United Kingdom
*amyjadesearle@gmail.com, **marko.leyen@quantum-dice.com
ABSTRACT
Recently, there has been growing interest in unconventional computing as an approach for solv-
ing NP-hard problems, by developing dedicated hardware to find solutions more efficiently than
conventional CPUs. In many of these approaches, however, certain problem geometries must be
transformed into forms that are more amenable to the available hardware topology through techniques
such as embedding, sparsification, and quadratisation, leading to a deterioration in solution quality.
A probabilistic computing architecture based on high speed photonic quantum random number
generators was recently proposed which utilises virtual hardware connections (Aboushelbaya et al.,
2025), circumventing the necessity for such procedures. Here, we discuss the applicability of virtually
connected hardware for running heuristic solving methods to solve a selection of problems, which due
to their geometry, would suffer from topological hardware restrictions. We also employ greedy graph
colouring algorithms for hardware parallelisation, allowing favourable scaling for desirable solution
qualities. To emphasise the difficulty in solving these problems on physically connected hardware,
we demonstrate the increase in problem size that would occur with quadratisation or sparsification.
Using simulations to emulate hardware, we predict that a photonic probabilistic computer would
outperform the time to solution recently reported for digital annealing units, on the ground state
approximation of Erd˝os–Rényi graph spin-glasses, by orders of magnitude.
Keywords Probabilistic computing, photonics, combinatorial optimisation, set cover, travelling salesperson problem,
spin-glass, Max-Cut, Ising model, QUBO
1
Introduction
The Ising model, and the equivalent class of quadratic unconstrained binary optimisation (QUBO) problems, act as
important test-beds for solving many combinatorial optimisation problems; appropriate choices for the quadratic and
linear terms enable the solution to a specified optimisation problem to be encoded in the ground state of a model [1].
Typically, Monte-Carlo sampling is employed with a global temperature variable to maximise the probability of
reaching the ground state, with simulated annealing (SA) and parallel tempering (PT) being common examples of such
algorithms.
arXiv:2605.06037v1  [cs.AR]  7 May 2026


<!-- page 2 -->
A VCPC as a solver for higher-order, densely connected, or reconfigurable COPs
A PREPRINT
Besides the simplicity of using a single energy function formulation to capture a number of diverse optimisation
problems, the Ising formulation is attractive because of the possibility of using dedicated hardware to implement such
algorithms. The purpose of this is to achieve a large reduction in the time to solution (TTS) for a given problem [2,3],
often in cases where exact answers are not required, and there is a preference for approximate solutions in reduced
runtimes. This form of computing, coined ‘Nature-inspired’ and first conceptualised in 1982 by Feynman [4], has
motivated the development of hardware including quantum annealers [5], digital ASIC annealers [6], oscillator
networks [7], and probabilistic Ising machines. The latter has been implemented in various forms, including using
digital CMOS technology [8] and magnetic tunnel junctions (MTJs) [9], which employ interconnected binary stochastic
neurons to drive computation [10].
Formulating the Ising model, however, involves specifying a quadratic energy function, consequently only allowing for
dyadic or pairwise interactions. This practically restricts us from implementing problems which require higher-order
energy functions without the introduction of auxiliary variables which significantly increases the search space of
the problem, making energy landscapes less favourable for heuristic algorithms to explore [11]. Moreover, practical
hardware implementations often impose additional constraints on the interaction density, particularly in quantum
annealers, but also in CMOS or MTJ-based probabilistic computing, since realising dense, all-to-all connectivity at
scale presents a large engineering challenge. One avenue has proposed reconfigurable architectures, but their approach
was only valid in the limit of sparse interactions [12]. Quantum annealers aim to circumvent the issue by embedding the
problem of interest into the architecture of the hardware, but since such embeddings are expensive to compute, usually
a universal but inefficient embedding is used [13]. Other platforms offer an advantage in this regard, such as digital
annealers that support all-to-all connectivity [6], or recently developed hardware in probabilistic computing, which
leverages photonic quantum random number generators as an entropy source for probabilistic bits and implements their
connections virtually [14]. These virtual connections are of particular interest, as we envision that this could simplify the
hardware realisation of three types of problem: dense quadratic problems, reconfigurable problems (where the geometry
changes dynamically), and higher-order problems. Reconfigurable problems present a challenge because multiple
embeddings must occur sequentially, increasing the complexity of the algorithm. Following this observation, we study
three problems, each of which constitutes a difficult geometry for distinct reasons. First, the set cover problem presents
higher-order interactions; it belongs to a general class of cover problems with applications ranging from crew scheduling
to search algorithms [15,16]. Second, the travelling salesperson problem (TSP) requires a reconfigurable geometry
when solved with k-means clustering (KMC) preprocessing. TSP is a widely studied combinatorial optimisation
task with applications in logistics, route planning [17,18], genomics, and astronomy [19,20]. Finally, spin-glasses
on Erd˝os–Rényi graphs necessitate dense interactions; this problem originates in condensed matter physics but has
implications for tasks ranging from network flow to job scheduling [21].
We begin this study by demonstrating the performance of a simulated virtually connected probabilistic computer
(VCPC) across these three problem classes, evaluating the best attainable solution qualities and the iteration count
required to reach specific target qualities by employing both SA and linear PT algorithms. Furthermore, we quantify the
impact of hardware constraints by demonstrating how sparsification and quadratisation lead to a significant increase in
problem size. For the higher-order set cover problem, we show that this leads to a clear deterioration in solution quality
for a fixed number of iterations. Regarding the spin-glass problem, we solve the NP-complete weighted Max-Cut
problem on Erd˝os–Rényi graphs with edge weights randomly chosen from {−1, 1}, making the results directly relevant
to a broad class of computational tasks, that can be expressed through NP-complete reductions [22]. We compare
our estimated VCPC performance to findings reported in Ref. [6], arguing that the predicted TTS on the hardware
introduced in Ref. [14] would achieve orders of magnitude improvements over those reported for digital annealing.
The remainder of this paper is structured as follows. We provide an overview of probabilistic computing in Sec. 2 and
present our main findings in Sec. 3. In particular, we detail the ability of the VCPC to solve the set cover, TSP, and
spin-glass problems, highlighting the advantage of virtual connectivity for problems presenting complex geometries
that would otherwise necessitate prohibitive overheads in physically constrained hardware.
2
An overview of probabilistic computing
A probabilistic computer consists of stochastic binary units, termed probabilistic bits (p-bits), which are interconnected
to form a network. We can associate an energy function with a given network by tuning the network’s interactions to
equate the p-bits’ states with energy contributions. In the case of a quadratic energy function, a probabilistic computer
represents an Ising model. The stochasticity of a p-bit is controlled by it’s input, I, and the corresponding output s,
fluctuates between −1 and 1 with a probability determined by I:
s = sgn{σ(βI) + randU[−1, +1]},
(1)
where σ is the sigmoid function, β is the inverse temperature, randU[−1, +1] is a uniform random number on the
interval [−1, +1]. Bipolar outputs can be converted to binary outputs via the mapping s
2 + 1
2 [10]. The input of a given
2


<!-- page 3 -->
A VCPC as a solver for higher-order, densely connected, or reconfigurable COPs
A PREPRINT
p-bit is determined by the outputs of it’s neighbours in the network. When such interactions are pairwise, the input of
the ith p-bit is computed as
Ii =
X
j
Jijsj + hi,
(2)
where Jij is the coupling matrix describing, the interaction between p-bits i and j, and hi is the constant bias contribution.
We also refer to an input Ii as the ith p-bit’s update drive when solving a specific problem, which we detail in App. C.
In this work, we will also consider higher-order networks, which for three-way interactions can be represented as:
Ii =
X
jk
J3
ijksjsk +
X
j
J2
ijsj + hi,
(3)
and can be generalised to higher-order polynomials. The network will, if J is symmetric, after a sufficient number of
p-bit updates, be described by the Boltzmann distribution with state probabilities:
p(E(s)) = e−βE(s)
Z
(4)
where the energy E of a configuration of p-bit states s = (s1, s2, ..., sn) is given by
E(s) = −
X
i<j
Jijsisj −
X
i
hisi,
(5)
and Z is the partition function. This energy function can be expressed in QUBO form via the Ising to QUBO mapping
that we show explicitly in App. C.3.1. Again, this can be generalised to higher-order interacting systems – three-way
and beyond. In this case the energy function is known as a higher-order unconstrained binary optimisation problem
(HUBO).
A system which natively approximates the Boltzmann distribution is useful for optimisation, because encoding the
solution to a problem in the ground state enables the system naturally visit the optimal state with the highest probability,
in accordance with the relation in Eqn. 4. There are a number of algorithms, to be discussed in the next section, which
have been developed in the field of energy-based optimisation to assist the system in finding the global ground state, by
avoiding convergence to local minima.
The update drive in Eqn. 2 can be determined by calculating the energy difference between the zero and one states of
the ith p-bit, and is expressed as:
Ii = E(s|si = 0) −E(s|si = 1).
(6)
While Eqn. 2 implies that the bias of a p-bit is updated through its direct interaction with neighbouring p-bits in the
network, it is also possible to mimic such interactions with virtual connections. By supplementing p-bits with a central
computing unit which is able to read the p-bits’ states required to perform the computation in Eqn. 2 or Eqn. 6, and
is also able to bias p-bit i, we can avoid directly implementing physical connections between p-bits. An in-depth
discussion of a photonic probabilistic computer which utilises virtual connections in this manner was recently introduced
in Ref. [14], and a brief discussion is provided in App. B.
2.1
Optimisation methods
There are a number of approaches to increase the solution quality of Boltzmann-like architectures, including SA; PT,
which may or may not be adaptive; and simulated quantum annealing. In this work we explore SA and non-adaptive PT.
Simulated annealing: A standard optimisation algorithm, where an external temperature variable is introduced, and
slowly decreased. At high temperatures, the system samples large subsections of its configuration space, while at low
temperatures the system is less likely to move states, and samples smaller subspaces. Conventionally, we define β to
denote to the inverse temperature of the system. We let βstart and βend indicate the start and end points of this interval,
and S indicate the number of steps in this interval. These steps are linearly separated within the interval; although
one can alternatively use geometric spacing. At each step, a p-bit group is chosen at random I times, and the update
drives of the incumbent p-bits are calculated in parallel. This can be done if p-bits in a group are independent, meaning
that the state of one p-bit is not needed to calculate the update drive of another p-bit (see Eqn. 2). Usually, finding
these groups is itself and NP-hard problem since it is equivalent to a graph or hypergraph colouring problem, but we
employ greedy algorithms which still offer dramatic speed-up through this group parallelisation. Since the algorithm is
probabilistic, we repeat the run R times.
3


<!-- page 4 -->
A VCPC as a solver for higher-order, densely connected, or reconfigurable COPs
A PREPRINT
Parallel tempering: Another optimisation algorithm, which involves introducing P copies of the system at different tem-
peratures, referred to as replicas, and probabilistically exchanging the states between two adjacent replicas periodically.
The probability of swapping two replicas’ states is given by the Metropolis criteria:
Pi,i+1 = min{1, exp{(∆β∆E)}},
(7)
where ∆β = βi+1 −βi and ∆E = Ei+1 −Ei are the differences in inverse temperature and energy between
neighbouring replicas, respectively. This ensures that low temperature states do not get stuck in local minima, by
propagating other potentially promising states from higher temperatures. The PT algorithm is often favoured in
the probabilistic computing literature [23, 24], although there is not yet a thorough understanding of when PT or
SA is favourable. Nevertheless in certain cases PT has been shown to perform better, such as on Ising spin-glasses
with Gaussian distributed couplings [25]. Here, we employ linear PT, where the replicas’ temperatures are linearly
interpolated between (βstart, βend). Similarly to SA, we perform I iterations on each replica, swapping replicas at an
interval of S. Again, at each iteration a group of p-bits is selected and their update drives are calculated in parallel.
We refer to SA and PT on the simulated VCPC as PC-SA and PC-PT respectively, to highlight that we refer to the
algorithms described above.
2.2
Scaling of the heuristic probabilistic algorithm
We now give a description of our probabilistic algorithm’s complexity, and highlight the speed-ups gained by paralleli-
sation. Considering SA, suppose we have S temperature steps with I iterations at each step (let the total number of
iterations be I = I × S). Without any parallelisation, suppose that the number of iterations needed to reach a solution
of quality q with probability p is Iq,p. Then, the total number of operations is
Oq,p = Iq,p · O(p(n))
(8)
where the prefactor indicates the number of times an update drive calculation is completed, and O(p(n)) indicates
that the update drive calculation scales at some polynomial p(n) that is dependent on the problem parameters n. Now
suppose we run R repeats, and, further, that we group p-bits into a list of groups G such that any set g ∈G contains
only p-bits which do not occur in their update drive calculation. Then,
Oq,p = Iq,p · O(p(n)) · R
¯G
.
(9)
Assuming we are able to run R repeats in parallel, this simply becomes
Oq,p = Iq,p · O(p(n))
¯G
.
(10)
where ¯G is the average size of groups in G. If every group is a singleton (only containing one p-bit), we have ¯G = 1
and no speed-up is obtained. Note, therefore, that the scaling of the group sizes is extremely important for obtaining
speed-ups when implementing these algorithms, as well as the form of the polynomial p(n). We may also express
this in terms of the number of groups in the set G, since |G|/N = 1/ ¯G. It was recently shown by a subset of the
authors, that perfect parallelisation in this sense can result in favourable scaling in cubic and biclique geometries for the
spin-glass problem [26].
Note that although parallelisation is important for ensuring the algorithm runs efficiently, such parallelisation procedures
are not specific to the hardware and can be done also on CPUs and GPUs. The expected speed-up when implementing
this on a VCPC originates from increases in sampling speed, which is discussed in Ref. [27]. Nevertheless, parallelisation
is an important consideration for reducing the runtime of probabilistic algorithms.
2.3
Advantages of a virtually connected probabilistic computer
As already emphasised, hardware dedicated to heuristic optimisation is often restricted to QUBO formulated energy
functions:
E(s) = −
X
i,j∈V
i<j
Ji,jsisj −
X
i∈V
hisi.
(11)
Here, we use s = (s1, s2, ..., sn) to emphasise that this is the general case, without reference to probabilistic computing.
We may encounter optimisation problems which are represented by higher dimensional energy functions:
E(s) = −
X
i1,i2,...,ik∈V
i1<i2...<ik
Jk
i1,i2,...,iksi1si2...sik −
X
i1,i2,...,ik−1∈V
i1<i2<...<ik−1
Jk−1
i1,i2,...,ik−1si1si2...sik−1 −... −
X
i∈V
hisi.
(12)
4


<!-- page 5 -->
A VCPC as a solver for higher-order, densely connected, or reconfigurable COPs
A PREPRINT
(a)
(b)
Figure 1: (a) The Chimera architecture of D-Wave Inc., a quantum annealer which is often employed to demonstrate
quantum annealing performance. In order to represent higher-order energy functions as a Chimera graph, or other
imposed architectures, an expensive quadratisation process must be performed. When converting a densely connected
problem, embedding or sparsification processes must also be employed. (b) A random sparse architecture. The process
of sparsification transforms a dense graph into a sparse graph, by reducing each vertex’s number of allowed connections.
Here, we refer to k as the dimension of the problem. QUBOs and Ising models are therefore of dimension 2, and a
dimension k problem contains k tensors in the Hamiltonian from dimension k to dimension 1 (the h vector). For a
finite Jl
i1,i2,...,il, the corresponding entities si1 through to sik share a non-zero l–way interaction.
Typically, hardware is restricted to certain interaction geometries; for D-Wave quantum annealers this is a Chimera
graph, as shown in Fig. 1a. More generally, at least in the case that only interactions of order two are implemented, the
allowed interaction geometry of a device is represented by a graph G, which is defined by an adjacency matrix Aij.
Entries where Aij = 0 indicate that the hardware is not able to directly implement an interaction between si and sj. As
such, anytime that (i) interactions are of order higher than 2, or (ii) Jij is non-zero but Aij = 0, requires additional
pre-processing to transform the problem into a representation that the hardware can manage. The down-side of these
processes is that they introduce additional variables, called auxiliary variables, effectively increasing the resources
required to perform the minimisation. The processes that deal with (i) and (ii), respectively, are quadratisation, and
embedding or sparsification. While embedding directly finds a map to target hardware, sparsification, which is closely
related to embedding, has no target topology in mind and simply aims to reduce the number of neighbours that each
vertex has at the cost of increasing the number of vertices. An example of a sparse random graph (random here referring
to the fact that edges are randomly placed) is shown in Fig. 1b.
1. Quadratisation is the process of turning a higher-order energy function into a quadratic energy function. Note
that usually, after this step, an embedding into the hardware graph still needs to be implemented. There is a
standard process for turning higher-order terms into quadratic terms, where a term of order k introduces k −2
additional terms. As such, if there are n terms of order k, n(k −2) auxiliary variables are needed. It is known
that quadratisation causes not only an increased in the size of the problem, but also results in a rugged energy
landscape [28].
2. Embedding is the process of mapping the logical variables and pairwise interactions of a problem to the
physical qubits and couplers of the hardware interactions graph G. Typically the hardware graph is sparse,
so finding an embedding, often requiring groups of physical qubits to represent a single logical variable, is
computationally expensive and consumes a large number of physical qubits. This overhead drastically limits
the effective problem size; for instance a 1000-qubit Chimera processor may only accommodate a problem of
∼45 variables [29]. As the number of required number of physical qubits scales poorly with the problem
size, it is rarely feasible to embed large or dense problems directly. Instead, the problem is broken down into
smaller subproblems, each of which is solved separately [30].
3. Sparsification is the process of reducing the connectivity of a dense problem’s interaction graph to a sparse,
energetically degenerate surrogate [31]. Fewer neighbouring interactions per vertex are desirable as in general
graphs with lower average connectivity (where connectivity refers to the number of vertices a single vertex is
connected to) are easier to implement in hardware and can be coloured using less update groups. The drawback
of this approach is that it leads to a significant expansion in the number of variables relative to the original
problem formulation.
5

[CAPTION] Figure 1: (a) The Chimera architecture of D-Wave Inc., a quantum annealer which is often employed to demonstrate


<!-- page 6 -->
A VCPC as a solver for higher-order, densely connected, or reconfigurable COPs
A PREPRINT
3
Problems
Here we outline three problems which we solve to evaluate the performance of the VCPC algorithm. Each problem is
known to be NP-hard and presents different challenges in hardware realisation.
3.1
Set cover
(a)
V1
V2
V3
...
Vn
0
0
1
0
...
1
1
0
0
1
...
1
2
1
0
0
...
0
3
0
1
0
...
0
...
...
...
...
...
...
m
0
1
0
...
0
(b)
Figure 2: (a) A hypergraph over twelve vertices with five hyperedges. The minimal hitting set is highlighted in white.
(b) An alternative tabular representation of the set cover problem.
The set cover problem asks, given a set X, and a selection {Vi}i∈I of subsets of X, such that these subsets cover X, or
S
i∈I Vi = X, what is the minimal subset R of the sets Vi which covers X? It can be visualised in tabular form, as
shown in Fig. 2b, where the problem becomes how best to choose a set of columns such that every row has at least one
‘1’ appearing in one of the columns.
We solve this problem by instead solving the hitting set problem, since the hitting set and set cover problems can be
seen to be equivalent by exchanging the “is an element of" relation with “is contained in". More formally, given a set Y
of vertices, and a collection H of subsets of Y , the hitting set is a subset Z ⊆Y such that every R ∈H has at least one
element in Z. We will refer to this collection H as a hypergraph. The size of the largest set in H is the dimension of the
hypergraph, and its size is the number of vertices. An example of a hypergraph is shown in Fig. 2a. It consists of twelve
vertices and five hyperedges, with the dimension of each edge indicated by the colour of that edge. The minimum
hitting set is highlighted in white. The two problems shown in Fig. 2 are equivalent.
An overview of algorithms used to approximate solutions to set cover or the hitting set, and an analysis of their
performance, can be found in Ref. [15]. An important result was reported by Lovász who introduced an algorithm which
can find a cover with at most a factor of 1 + ln d vertices greater the optimal solution, where d is the maximum degree
of the hypergraph. In terms of solving the problem with energy-based solvers, a QUBO formulation was introduced
in Ref. [1], and the related problem of set cover with pairs was solved for m = 17 using quantum annealing and
compared against SA, where no quantum speed-up was observed over classical runtimes when averaging over different
instances [32]. Others have used quantum annealing and neuromorphic computing to study vertex cover, which is a
specified case of the hitting set when every edge has dimension 2 [30,33], where the hypergraph is simply a graph.
A higher-order binary optimisation problem can be obtained by generalising the energy function for vertex cover, giving
the following:
E(s) = A
X
r∈R
Y
v∈r
(1 −sv) + B
X
v∈V (G)
sv,
(13)
where sv is the state of the p-bit at vertex v and s = (sv)v∈Y is a choice of state for each vertex in the hypergraph R.
The hyperparameters A and B control the contribution of the two terms, and it can be shown that A > B must be taken
to ensure that the solution found is a valid cover. Note that Eqn. 13 can be put in the form of Eqn. 12 by grouping
together terms of each order and determining the elements of the Jk tensors for the different orders k. Also note that
for a hypergraph of dimension k, the maximum product occurring in the energy is of order k, making the energy a
polynomial of order k. We show the formulation of Eqn. 13 explicitly in App. C.1.1.
6

[CAPTION] Figure 2: (a) A hypergraph over twelve vertices with five hyperedges. The minimal hitting set is highlighted in white.


<!-- page 7 -->
A VCPC as a solver for higher-order, densely connected, or reconfigurable COPs
A PREPRINT
100
200
300
400
500
Hypergraph Vertices
50
100
150
200
250
300
350
Iterations
PC-SA (k=5, q = 1.05)
PC-SA (k=5, q = 1.1)
PC-SA (k=5, q = 1.2)
PC-SA (k=10, q = 1.2)
PC-SA (k=10, q = 1.3)
(a)
200
400
600
800
Number of variables of HUBO
0
5000
10000
15000
20000
25000
Number of variables of QUBO
k=5
k=10
(b)
200
400
600
800
Hypergraph Vertices NV
0.98
1.00
1.02
1.04
1.06
1.08
Solution quality q
PC-SA (HUBO, I = rNV)
SA (QUBO, I = rNV)
SA (QUBO, I = rNV *)
(c)
Figure 3: (a) The number of iterations required to reach certain solution qualities for hypergraph dimensions k = 5
and k = 10, where now a single iteration updates a group of independent p-bits in parallel. (b) The growth in problem
size for randomly generated instances for hypergraph dimensions k = 5 and k = 10, when quadratising the HUBO into
a QUBO. (c) The performance of the PC-SA algorithm, as measured by solution quality, against a standard QUBO SA
solver, where the cooling schedules are equivalently scaled (see text).
In App. C.1.2 we show that p-bits’ update drive for the hitting set problem is given by:
Ik = A
X
r∈R|
k∈r
Y
v∈r|
v̸=k
(1 −sv) −B.
(14)
For each p-bit, this scales as at most as O(m(k −1)), where m is the number of hyperedges in H and k is the
dimension of H. In practice, this computation is often significantly more efficient, as the product can be truncated to
zero immediately upon encountering any zero variable
Hypergraph dimensions of k = 2 reduce naturally to QUBO form, so in order to test the performance of the PC-SA and
PC-PT algorithms on higher-order problems we require that k > 2. We conducted our initial performance tests using
k = 5 and examine the impact of increasing k on algorithmic performance in App. E. We generated 100 instances for
different choices of size in the range of 50 vertices to 1000 vertices. The results for k = 5 and k = 10 are shown in
Fig. 3a. The hyperparameters used are listed in App. A.1. Figure 3 shows the solution quality q, where for this example
7


**[Table p7.1]**
| PC-SA (k=5, q = 1.05) 350 PC-SA (k=5, q = 1.1) PC-SA (k=5, q = 1.2) 300 PC-SA (k=10, q = 1.2) PC-SA (k=10, q = 1.3) 250 Iterations 200 150 100 50 100 200 300 400 500 Hypergraph Vertices |  |
| --- | --- |
|  | 25000 k=5 QUBO k=10 20000 of variables 15000 10000 of Number 5000 0 200 400 600 800 Number of variables of HUBO |

[CAPTION] Figure 3: (a) The number of iterations required to reach certain solution qualities for hypergraph dimensions k = 5


<!-- page 8 -->
A VCPC as a solver for higher-order, densely connected, or reconfigurable COPs
A PREPRINT
q is defined as:
q = HSA
HSref
,
(15)
where HSA refers to the size of the hitting set found by algorithm A. Usually, we expect q ≥1 as the algorithm will
return a value larger than or equal to that of the reference algorithm, although, q < 1 is also possible here since the
reference size was calculated using SetCoverPy [34], which is itself a heuristic algorithm.
Figure 3a depicts the average iterations required to reach target solution qualities for k = 5 and k = 10 using a greedy
hypergraph colouring algorithm for parallelisation. For the k = 5 data, the iteration requirement increases only modestly
when shifting from 20% to 10% above the reference solution. In contrast, reaching the 5% quality threshold requires a
significant jump in the number of iterations, likely reflecting the heuristic nature of the simulated annealing algorithm
in finding high-accuracy solutions. For the k = 10 case, the increased complexity of the energy landscape demands
significantly more iterations to achieve a comparable improvement in solution quality. Specifically, shifting from 30%
to 20% above the reference necessitates a much larger increase in the iteration count than what was required for the
k = 5 instance.
The general trend shown in Fig. 3a is that the number of iterations required to reach a particular solution quality appears
to scale linearly with vertex size for the ranges investigated. The small number of iterations required is a reflection of
the large reduction due to hypergraph colouring. In App. D we investigate how the number of groups, and therefore the
speed-up factor, changes with the dimension of the hypergraph and number of edges. We find that for lower hypergraph
dimension the best speed-ups are obtained, regardless of the number of edges. On the other hand, for very large
dimensions, greedy algorithms perform well, which makes such parameter space less interesting. While group updates
reduce the number of iterations required to reach a particular solution quality, we also expect, considering the bias
shown in Eqn. 14, that the number of clock cycles required to perform a single iteration, or group update, will depend
on m and k. A careful analysis of which regimes the speed-up due to group updates undercuts the increase in time
needed to compute the bias will be left to future work.
While VCPC is able to work directly with HUBO energy functions, other hardware platforms with physically connected
p-bits necessitate quadratisation. To this end, we convert the test instances considered for k = 5 and k = 10 to quadratic
form, using standard quadratisation procedures, and investigate how (i) the performance difference, for an equal number
of iterations, and (ii) the scaling of this conversion process. In Fig. 3b, we show the growth of the problem size with the
quadratisation process for dimension size k = 5 and k = 10. This growth for k = 5 is already quite significant, with
the number of QUBO variables corresponding to more than double the number of HUBO variables. For k = 10, the
rate of this growth increases drastically. In particular, the largest problem sizes of 1000 vertices for the HUBO problem
were converted to, on average, 25000 vertices for the QUBO.
Figure 3c shows the performance of the PC-SA algorithm, against a PC-SA solver using the QUBO representation. For
each, we have the same starting and ending temperature in the cooling schedule. The strength parameter, which must
necessarily be chosen when quadratising an energy function, was optimised for the best performance. The iterations
for the PC-SA calculation were taken as I = rNV , where NV is the number of hypergraph vertices of the HUBO and
as detailed in App. A, and we set r = 5. For the QUBO forms, we performed the calculation with I = rNV , as for
the HUBO case, as well as I = rNV ∗. Here NV ∗is the number of vertices of the QUBO after quadratisation (see
Fig. 3b), and as such the number of iterations is significantly higher. As shown in Fig. 3c, the solution quality for both
formulations remains stable as the number of vertices increases, though we note that the iteration count scales linearly
with problem size to maintain this performance. Despite this scaling, the QUBO algorithm consistently underperforms
the HUBO approach. Crucially, this performance gap persists even when the iteration count for the QUBO is increased
to account for the larger number of variables introduced by quadratisation. It is likely that the quadratisation process
induces a more rugged energy landscape, as noted in previous studies [28]. This increased complexity explains why
simply scaling the iteration count does not allow the QUBO formulation to match the solution quality achieved by the
native HUBO.
3.2
Travelling salesperson problem
The TSP is an established NP-hard combinatorial optimisation task. Solving the TSP involves finding the minimum
tour between N cities {A, B, C, ...}, such that each city is visited exactly once while ensuring that one returns to the
starting city at the end of the tour. It is convenient to define a distance matrix, Dij, showing the associated cost of
8


<!-- page 9 -->
A VCPC as a solver for higher-order, densely connected, or reconfigurable COPs
A PREPRINT
travelling between cities i →j:
Dij =




0
dAB
dAC
. . .
dBA
0
dBC
. . .
dCA
dCB
0
. . .
...
...
...
...




(16)
We assume a symmetric problem where Dij = Dji for simplicity. The energy function for the TSP is given by:
E(S) = A
" N−1
X
i=0
  N−1
X
k=0
Sik −1
 2
+
N−1
X
k=0
  N−1
X
i=0
Sik −1
 2#
+ B
" N−1
X
i=0
N−1
X
j=0
N−2
X
k=0
Dij Sik Sj,k+1 +
N−1
X
i=0
N−1
X
j=0
Dij Si,N−1 Sj,0
#
(17)
where S is an N × N matrix with rows corresponding to the cities that can be visited, and columns corresponding
to the order in which cities are visited1. We explain how the ground state of Eqn 17 encodes solutions to the TSP in
App. C.2.1, and provide the update drive derivation in App. C.2.2. The difficulty in solving the TSP lies in the scaling
of the number of possible tours, which is well known to be (N −1)!/2 for a symmetric problem. Consequently, the
brute-force approach becomes unfeasible for problems involving N > 15 cities, making alternative methods necessary
to solve TSP problems within a reasonable timeframe.
One approach is to use KMC to reduce the number of p-bits needed to solve a TSP instance [35,36]. KMC is a shallow
unsupervised machine learning algorithm which partitions data points into K groups. In the context of solving the TSP,
we pre-process the problem by recursively clustering the city coordinates m times. Here, clustering refers to a process
by which the centroid position of a group of coordinates, which may be cities or clusters of cities, is computed and used
as the coordinate in the next round, thereby coarsening the data. Letting Ki denote the ith cluster of cities, where K1
represents the original problem, the clustering is performed as K1 →K2 →· · · →Km where K1 > K2 > · · · > Km,
meaning that we are simplifying the problem at each step.
Coarsening the problem to a TSP with Km centroids (where typically Km ∼4) enables the solver to easily find
the minimum tour of the reduced problem. This result allows one to construct a mask matrix, Mm−1, for the Km−1
centroid problem, which is a binary matrix of shape Km−1 × Km−1. This procedure is depicted in Fig. 4. The purpose
of M is to dictate which p-bits in S need to be modelled, where 1s indicate that the p-bit should be included, while 0s
correspond to clamping the specified p-bit to the 0 state.
(a)














1
1
1
1
0
0
0
0
0
0
1
1
1
1
0
0
0
0
0
0
1
1
1
1
0
0
0
0
0
0
1
1
1
1
0
0
0
0
0
0
0
0
0
0
1
1
1
0
0
0
0
0
0
0
1
1
1
0
0
0
0
0
0
0
1
1
1
0
0
0
0
0
0
0
0
0
0
1
1
0
0
0
0
0
0
0
0
1
1
0
0
0
0
0
0
0
0
0
0
1














(b)
Figure 4: (a) Illustration of a single KMC process on an arbitrary 10 city TSP. (b) The mask, M0, generated from
solving the K1 problem, showing how KMC has reduced the number of p-bits required from 100 →29 [35].
The existence of M can be justified by assuming that the optimal tour for the Km−1 problem will not comprise moves
between different clusters until all constituent cities have been visited in a given cluster. Ignoring suboptimal routes
using this method vastly reduces the search space of the Km−1 problem, allowing better approximate solutions. This
process is then iterated until we arrive at the original N city TSP, K1, where we use the mask generated from solving
the K2 problem to return a final tour.
In addition to reducing the number of p-bits, the KMC pre-processing also reduces the number of colour groups needed
to cover S. For example, using KMC on the 10 city problem shown in Fig. 4 reduces the number of colour groups from
1While before we used s to refer to the global state, highlighting that it was a vector, in this context since it is a matrix we use the
conventional upper case letter.
9

[CAPTION] Figure 4: (a) Illustration of a single KMC process on an arbitrary 10 city TSP. (b) The mask, M0, generated from


<!-- page 10 -->
A VCPC as a solver for higher-order, densely connected, or reconfigurable COPs
A PREPRINT
22 →8. This reduction is due to a combination of using fewer total p-bits as well as fewer conflicts in the remaining
p-bits’ update drives. Reducing the number of colour groups is significant, as it allows a greater proportion of the global
system to be updated in parallel.
The reconfigurable geometry of the VCPC is essential for this process to be implemented on a probabilistic annealer.
As each iteration of KMC produces a different set of city groupings and corresponding connectivity patterns, the solver
must dynamically adjust which couplings are active. Annealing hardware with fixed physical connections cannot
accommodate these changing geometries, whereas the VCPC can reprogram its virtual couplings in real time, enabling
adaptation to the evolving problem geometry. The VCPC’s reconfigurable geometry enables us to employ KMC in
order to improve the solution quality, where both PC-SA and PC-PT are sub-methods used in the KMC procedure. In
both cases, we show in Table 1, the KMC pre-processing greatly reduces the average tour cost and enables convergence
nearer to the ground state in the best case scenario. It is also worth noting that we are now able to reduce our penalty
weight parameter A, while maintaining a high fidelity of valid solutions, as there are fewer states contributing to the
possible invalid tours available. We show our hyperparameter choices in App. A.2.
Table 1: Performances on TSPLIB [37] benchmark instances using different solvers. Each solver was run 100 times
with different seeds, and the results are shown as a fraction of the known optimal solutions. We do not count invalid
tours in these averages, where the number of valid tours per method can be seen in the "Valid" columns.
Instance
PC-SA
PC-PT
Best
Ave.
Valid
Best
KMC
Ave.
KMC
Valid
KMC
Best
Ave.
Valid
Best
KMC
Ave.
KMC
Valid
KMC
burma14
1.071
1.237
100
1.000
1.079
99
1.089
1.222
100
1.000
1.082
100
ulysses16
1.169
1.303
100
1.007
1.072
81
1.117
1.238
100
1.007
1.070
83
ulysses22
1.306
1.559
100
1.010
1.074
91
1.262
1.416
100
1.010
1.072
91
berlin52
2.221
2.543
99
1.091
1.197
99
2.551
2.912
42
1.049
1.199
99
More specifically, in Table 1 we show that for smaller instances: burma14, ulysses16 and ulysses22, KMC enables
both SA and PT to converge to within 1% of the optimal tour for the best case scenario. As well as this, the average
tour is consistently ∼7.5% greater than the optimal tour which represents a vast improvement on the standard PC-SA
and PC-PT solvers. For the largest instance considered, berlin52, the KMC procedure approximately halves the best
and average tour costs for both PC-SA and PC-PT. Remarkably, the fidelity of valid tours is consistently > 80% for
every instance and method investigated with the exception of standard PC-PT on berlin52. Despite this, the addition
of the KMC procedure in the PT solver enables a best tour cost of 1.049 to be achieved. Our scores for berlin52 are
comparable to digital annealing results as reported in Ref. [38]. Two improvements that could be made would be
to utilise the recently published 2D-PT [39] and Transformer-Augumented-PT [40] algorithms, to ensure constraint
satisfaction and propose pre-learned global moves, respectively.
3.2.1
Sparsification
It is known that dense graphs are particularly difficult to embed into hardware [13]. Relatively little is reported in the
literature related to the geometry of the TSP problem, both with and without KMC. Since embeddings or sparsification
must be carried out to implement the problem on hardware, understanding this geometry helps to shed light on the
difficulty of doing so.
The density of a graph G with NE edges (NE ≈2N 3) and NV vertices (NV = N 2) is defined as M =
NE
NV ×(NV −1),
where NV × (NV −1) is the maximum possible number of edges. As such, the density of the TSP problem can be
expressed as:
M ≈2N 3
N 4 = 2
N ,
(18)
where N is the number of cities (or equivalently, the number of time steps). Since M →0 as N →∞, the problem is
not strictly dense, but decays more slowly than any other class of non-dense graphs.
Another geometrical property affecting the embeddability is the size of the largest connected component in the geometry,
corresponding to the largest subset of vertices which forms a complete graph. This structure is particularly difficult to
embed, and places a lower bound on the growth of the problem when embedding is performed. We find that, for the
berlin52 instance with KMC, the biggest connected component is of order 100 vertices. Therefore, both the high number
10

[CAPTION] Table 1: Performances on TSPLIB [37] benchmark instances using different solvers. Each solver was run 100 times


<!-- page 11 -->
A VCPC as a solver for higher-order, densely connected, or reconfigurable COPs
A PREPRINT
of connections and large connected components would make these difficult problems to find efficient embeddings for,
or to sparsify without a substantial increase in problem size.
One way to circumvent such embedding difficulties is to break the problem down into subproblems. Although, in
principle, one could break down the problem at each step in KMC, the geometry of the problem dynamically changes
at each step due to the uniqueness of the obtained mask. This means that the embedding must be calculated m times,
and cannot be pre-computed, significantly increasing the complexity when running on restricted physically connected
hardware. This also applies to sparsification, as the problem must be sparsified for each and every KMC step.
We now consider a graph sparsification procedure recently published in Ref. [31], in the context of solving the TSP.
The purpose of graph sparsification is to reduce the density of a problem by enforcing a limit on each p-bit’s number
of neighbours, at the cost of introducing additional auxiliary p-bits. This connectivity regulation is crucial for non-
virtually connected architectures, where large scale fully connected graphs become unfeasible in hardware. The graph
sparsification algorithm introduces copy nodes to limit the connectivity to the hyperparameter k. Ferromagnetic edges
are then introduced between a source node (the original p-bit) and it’s copy nodes in order to correlate their behaviour.
In Fig. 5a the relationship between the number of required p-bits and the connectivity is shown. As expected, for a
lower neighbour count, the number of p-bits is significantly higher. The unmasked architecture is also more difficult to
sparsify, as for equal neighbour counts, the number of p-bits required is always greater than that of the KMC masked
version. It can be seen in Fig. 5a, for example, that constraining each p-bit to have at most 10 neighbours results in
a problem of size of ∼1000 p-bits for the unmasked version and ∼500 for the masked version, while the original
problem size — indicated by the grey dotted line — required only 196 p-bits.
We introduce two parameters to further help visualise the scaling of the sparsification procedure on TSP instances. If G
denotes the connectivity of the original TSP problem, and GS denotes the sparsified graph, then we let rN be the ratio
of the number of vertices in GS to the number of vertices in G:
rN = NV (GS)
NV (G) .
(19)
We also introduce a sparsification ratio, rS, which shows how the maximum number of neighbours decreases as
auxiliary variables are included. Specifically,
rS = N(G)
N(GS)
(20)
where N(G) is the maximum number of neighbours in a graph G.
In Fig. 5b we show how the ratio rN varies, for both the unmasked and KMC masked methods, as the ratio rS increases.
As expected, the KMC masked p-bit array scales more favourably than the unmasked case, as the KMC masked problem
is initially denser. For a graph which is 10 times more sparse than the original, the problem increases in size by a factor
of at least 5. It is also worth noting that both cases start at the point (1, 1), corresponding to the unsparsified case.
11


<!-- page 12 -->
A VCPC as a solver for higher-order, densely connected, or reconfigurable COPs
A PREPRINT
0
10
20
30
40
50
60
Number of neighbours k
0
1000
2000
3000
4000
5000
Number of p-bits
Unmasked
KMC Masked
Unsparsified
(a)
0
5
10
15
20
25
Sparsification ratio rs
0
5
10
15
20
25
Increase in problem size rN
Unmasked
KMC Masked
(b)
Figure 5: Sparsification results for the burma14 instance. (a) The number of p-bits required to reach an architecture
with a specified number of neighbours, with a low neighbour count requiring more p-bits. (b) The relationship between
the ratios rN and rS representing the increase in problem size and the sparsification of the graph, respectively (for
k = 2). Results are included for both KMC masked and unmasked p-bit arrays [31].
3.3
Spin-glass
The final NP-hard problem we consider is the problem of finding low-energy states of a spin-glass lattice, which is
defined as a system of N spins with a symmetric N × N coupling matrix J. The elements of the matrix encode
ferromagnetic and anti-ferromagnetic couplings between lattice sites, the frustration of which makes the problem
complex. Starting from the Ising Hamiltonian with an external field:
H = −
X
i<j
Jijσiσj −
X
i
hiσi
(21)
where σi ∈{−1, +1}, we map this to a QUBO problem by applying the substitution σi = 2si −1 where si ∈{0, 1}.
This transformation results in the QUBO energy function:
E(s) =
X
i<j
Qijsisj +
X
i
bisi
(22)
where Qij (for i < j) and bi are the transformed coupling and external bias terms dervied from Jij and hi, under the
bipolar to binary mapping (shown explicitly in App. C.3.1). The goal is to find the combination of values of s that
minimise the energy of this lattice according to these connections, or a close approximation of it. This problem can also
be phrased in terms of the weighted Max-Cut problem, where one optimises for splitting a graph in two, such that the
highest summed weight of edges crosses the divide. We show the p-bit update drive derivation in App. C.3.2.
Different lattice topologies can be encoded by populating J differently, ranging from the Edwards-Anderson (EA)
model [41–44] of nearest-neighbour-interactions in a D-dimensional lattice to the Sherrington-Kirkpatrick model of
full-connectivity [43, 45, 46]. In Ref. [26] spin-glasses were solved for cubic (sparse) lattices and biclique (dense)
lattices with probabilistic computing and for this section we focus more specifically on the effect of graph connectedness
on performance.
To this end, we focus on Erd˝os–Rényi graphs [47], where all possible edges have a probability of p of being included in
the graph. We consider a range of p-values, taking the graphs through different degrees of connectivity. In the limit
of p →1, the graph becomes fully-connected, whereas the p = 0 graph is disconnected and its optimal configuration
is trivial. In any p ̸= 0 configuration, the topology of the lattice is dense in that as N →∞the ratio of obtained
versus possible edges approaches p, not zero. Within a given topology, different Max-Cut versions can be formulated
depending on how the values Jij are chosen. For this work we follow Ref. [6] in considering edges sampled from a set
{−1, +1} with equal probability.
Finding the update-groups of Erd˝os–Rényi graphs is itself an NP-hard problem of colouring a random graph. For
practical purposes, we use a greedy colouring algorithm. While this approach does not guarantee an optimal colouring,
12


**[Table p12.1]**
| Unmasked 5000 KMC Masked Unsparsified 4000 p-bits 3000 of Number 2000 1000 0 0 10 20 30 40 50 60 Number of neighbours k | Unmasked 25 KMC Masked N r size 20 problem 15 in 10 Increase 5 0 0 5 10 15 20 25 Sparsification ratio r s |
| --- | --- |

[CAPTION] Figure 5: Sparsification results for the burma14 instance. (a) The number of p-bits required to reach an architecture


<!-- page 13 -->
A VCPC as a solver for higher-order, densely connected, or reconfigurable COPs
A PREPRINT
(a)
(b)
(c)
Figure 6: Spin-glass topologies of (a) Edwards–Anderson model on a 2D lattice (sparse). (b) Erd˝os–Rényi random
graph (dense). (c) Sherrington–Kirkpatrick all-to-all model (dense & maximally connected).
0.2
0.4
0.6
0.8
1.0
Erd s-Rényi p
101
101
102
102
103
103
Iterations
q = 0.5
q = 0.8
PC-SA
PC-PT
N = 100
N = 500
N = 750
N = 1024
(a) Iteration scaling with density at various sizes
200
400
600
800
1000
Graph size N
102
102
103
103
Iterations
q = 0.5
q = 0.8
PC-SA
PC-PT
(b) Iteration scaling with graph size at p = 1
Figure 7: The number of iterations required to attain solution qualities of q = 0.5, 0.8 over (a) the Erd˝os–Rényi graph
density p and (b) the graph size N at a fixed p = 1. Note we redacted the N = 250 data in (a) for better visual clarity.
it operates in polynomial time and ensures the use of no more than ∆+ 1 colors, where ∆denotes the maximum
degree of the graph. This makes it a practical and efficient fallback when exact graph colouring is computationally
intractable [48]. In 1976 Bollobás [49] proved that the chromatic number of an Erd˝os–Rényi graph with constant p and
n vertices is with very high likelihood
χ
 G(n, p)
 
=
n
2 log n log
 
1
1 −p
 
(1 + o(1)).
(23)
This increases the efficacy of probabilistic computers when either a low probability or a low number of vertices lead to
a low number of update groups, with the limit of a bipartite graph being the optimal configuration.
We consider spin-glasses on Erd˝os–Rényi graphs for probabilities ranging from p = 0.1 to p = 1. We choose solution
qualities of 0.5 and 0.8 in order to directly compare to results reported in Ref. [12], where SA, PT, and digital annealing
were considered. The authors found that the TTS for digital annealing showed a significant improvement over SA and
PT algorithms, with the TTS of graph sizes up to N = 1024 being below 1 second for digital annealing, and growing
up to order 103 seconds for parallel tempering.
We therefore consider lattice sizes N = 100, 250, 500, 750, 1024 and take q = 0.5 and q = 0.8, computing the number
of iterations required to reach such qualities2 shown in Fig. 7a. In Fig. 7b, we show how the number of iterations scales
with increasing N when the probability of including an edge is fixed at p = 1. We approximate the ground states using
large PC-PT simulations (see App. A.3 for the number of iterations and replicas used).
Note that for SA, the number of iterations is the total number I, and in particular we take I = 1, so that at each β-step
only one iteration is performed. On the other hand, for PC-PT the number of replicas is not accounted for in the iteration
2Note that in Ref. [12] they use the terminology ‘cutoff’ rather than solution quality, but the two are equivalent.
13


**[Table p13.1]**
| q = 0.5 PC-SA N = 100 N = 750 q = 0.8 PC-PT N = 500 N = 1024 103 103 Iterations 102 102 101 101 0.2 0.4 0.6 0.8 1.0 Erd s-Rényi p | 103 103 Iterations q = 0.5 q = 0.8 PC-SA PC-PT 102 102 200 400 600 800 1000 Graph size N |
| --- | --- |

[CAPTION] Figure 6: Spin-glass topologies of (a) Edwards–Anderson model on a 2D lattice (sparse). (b) Erd˝os–Rényi random

[CAPTION] Figure 7: The number of iterations required to attain solution qualities of q = 0.5, 0.8 over (a) the Erd˝os–Rényi graph


<!-- page 14 -->
A VCPC as a solver for higher-order, densely connected, or reconfigurable COPs
A PREPRINT
count. This is because replicas can in theory be run in parallel in any hardware implementation. Based on the number
of iterations and placing only moderate assumptions on hardware, we can make estimates from Fig. 7 about the TTS we
expect when running such simulations on dedicated probabilistic hardware.
3.3.1
Time to solution estimation
Recall from Eqn. 10 that the relationship for estimating the scaling of operations to reach a solution of quality q with
probability p for a particular problem is:
Oq,p = Iq,p · O(p(n))
¯G
,
(24)
assuming all R repeats can be done in parallel. We can similarly get an estimate for the TTS if, rather than computing
the number of operations needed to perform the update drive calculation once, we compute the number of clock cycles
required by the hardware to compute the update drive, which we will denote ncycles(n). Then, given that the hardware
runs at a frequency f, which is to say in one second it performs f cycles, the total TTS is:
TTSq,p = Iq,p · f −1 · ncycles(n)
¯G
(25)
where recall that n represents problem parameters, which for spin-glass would be the size of the problem, the density
of edges, and so forth. Given an equation for computing the update drive, estimates can be made about the scaling of
ncycles(n). We compute that the number of clock cycles required to do a single update drive calculation is at most:
ncycles = log2(N) + O(10),
(26)
where the first term comes from the adder tree algorithm for computing multiplications, and the second term is a
constant overhead accounting for miscellaneous background processes. Taking a clock speed of 2.7 GHz, which was
provided in Ref. [14], Eqn. 25 gives:
TTSp,q = 1
¯G · Ip,q ·
1
2.7 × 109 · [log2(N) + O(10)] .
(27)
The factor of 1
¯
G · Ip,q for the largest lattice size of N = 1024 can be read off of Fig. 7b, and is approximately 2000
(note that group updates were used in calculating the number of iterations). Although the number of repeats do not in
theory affect this scaling, since they can be run in parallel, it is important to note hardware limitations. However, here
we have only used a single repeat and therefore need not consider hardware limitations in this respect. As shown in
Fig. 8, we estimate the TTS for q = 0.8 as ∼10−5 seconds for the largest graph size of 1024, outperforming the digital
annealing algorithm considered in Ref. [6] by several orders of magnitude.
200
400
600
800
1000
Graph size N
10
6
10
6
10
5
10
5
10
4
10
4
10
3
10
3
10
2
10
2
10
1
10
1
100
100
Time to solution (s)
q = 0.5
q = 0.8
PC-SA
DA (Ref.[6])
Figure 8: Comparison between TTS estimates for PC-SA using Eqn. 27 and reported digital annealing values from
Ref. [6], for Erd˝os–Rényi graphs with p = 1.
14

[CAPTION] Figure 8: Comparison between TTS estimates for PC-SA using Eqn. 27 and reported digital annealing values from


<!-- page 15 -->
A VCPC as a solver for higher-order, densely connected, or reconfigurable COPs
A PREPRINT
3.3.2
Sparsification
Finally, we consider the sparsification of spin-glasses. For Erd˝os–Rényi random graphs, the density M is given by:
M = pN(N −1)
N(N −1) = p.
(28)
Consequently, when p = 1 these graphs are fully connected, with each node having N −1 neighbours. The difficulty
of embedding such graphs into hardware is well-documented [13]. Using efficient embedding algorithms on D-Wave’s
2000-qubit system, the maximum embeddable problem size scales with density: it is approximately 275 variables for
sparse graphs (p = 0.01) and decreases to 60 variables for complete graphs (p = 1).
Here, as with the TSP, we focus on sparsification results since these are hardware independent. In Fig. 9 we show the
effect of sparsification on the graph size for an N = 100 test instance, with p = 0.1, 0.5, 1. Figure 9a shows the number
of p-bits needed for a specified number of neighbours in the sparsified graph. As the number of allowed neighbours
increases, the number of p-bits in the unsparsified problem is eventually recovered. For the p = 1 instance, restricting
the number of neighbours to less than 10, requires over 1000 to implement the sparsified problem. Many hardware
implementations would suffer subject to these constraints, including the Chimera graph structure used by D-Wave Inc.,
where each qubit is limited to 6 neighbours. Figure 9b shows how the ratio rN, representing the increase in problem
size compared to the unsparsified version, increases as the sparsification ratio rs increases. Again, for p = 1, the growth
is most significant: in the worst case, sparsifying the original graph by a factor of 50 results in an increase in the size of
the problem by a factor of 100.
0
10
20
30
40
50
60
Number of neighbours k
0
2000
4000
6000
8000
10000
Number of p-bits
N = 100, p = 0.1
N = 100, p = 0.5
N = 100, p = 1.0
Unsparsified
(a)
0
10
20
30
40
50
Sparsification ratio rs
0
20
40
60
80
100
Increase in problem size rN
N = 100, p = 0.1
N = 100, p = 0.5
N = 100, p = 1.0
2
4
2
4
(b)
Figure 9: The sparsification results for an Erd˝os–Rényi spin-glass instance with N = 100. (a) The number of p-bits
required for a specified maximum number of neighbours for the sparsified graph (b) shows the relationship between the
ratios rN and rS representing the increase in problem size and the sparsification of the graph (for k = 2).
15


**[Table p15.1]**
| 10000 N = 100, p = 0.1 N = 100, p = 0.5 N = 100, p = 1.0 8000 Unsparsified p-bits 6000 of Number 4000 2000 0 0 10 20 30 40 50 60 Number of neighbours k | 100 N = 100, p = 0.1 N = 100, p = 0.5 N N = 100, p = 1.0 r 80 size problem 60 4 40 2 in Increase 2 4 20 0 0 10 20 30 40 50 Sparsification ratio r s |
| --- | --- |

[CAPTION] Figure 9: The sparsification results for an Erd˝os–Rényi spin-glass instance with N = 100. (a) The number of p-bits


<!-- page 16 -->
A VCPC as a solver for higher-order, densely connected, or reconfigurable COPs
A PREPRINT
PC
DA
QA
VCPC
Dense
✗
✓
✗
✓
Reconfigurable
✗
✓
✗
✓
Higher-order
✗
✗
✗
✓
Table 2: A comparison of the ability of different Nature inspired computing platforms to implement the types of
problem considered in this study, without resorting to quadratisation, embedding, or sparsification. We compare the
VCPC with (non virtually connected) probabilistic computing (PC), digital annealing (DA), and quantum annealing
(QA) platforms.
4
Conclusion
We have presented an argument for the use of a VCPC on problems with geometries that are difficult to implement on
hardware lacking virtual connections.
In Section 3, we demonstrated the performances of the PC-SA and PC-PT algorithms on NP-hard problems by
conducting VCPC simulations. For the hitting set problem, we showed that the VCPC achieved solutions qualities
consistently within 1% of the reference solution for hypergraphs up to 1000 vertices (for dimension k = 5). Crucially,
this performance relies on greedy hypergraph colouring to facilitate parallel updates. We found that for hypergraph
dimensions k ≤6, the number of update groups is typically less than 10, providing an order of magnitude speed-up for
many instances.
For the TSP, we showed that the addition of KMC significantly improved the solution quality. For the 52-city problem
(berlin52), the KMC procedure reduced the best and average tour costs be ∼50% compared to standard PC-SA and
PC-PT, with a valid tour fidelity across all instances exceeding 80%. In our study of the spin-glass problem, we
estimated the TTS for Erd˝os–Rényi graphs up to N = 1024. Using a 2.7 GHz clock frequency, we estimate that the
VCPC could achieve a TTS of order 10−5 seconds, outperforming the digital annealing results reported in Ref. [6] by
orders of magnitude.
In Table 2, we emphasise these results by highlighting the advantages of VCPC over other platforms. While fully
connected digital annealers are capable of implementing any QUBO, they necessitate quadratisation, which we showed
leads to a problem size increase by a factor of 25. Also, in our hitting set study, we showed that QUBO formulations
struggle to reach the same solution quality as the native HUBO formulation. Furthermore, both quantum annealers and
physically connected probabilistic architectures suffer from server scaling limitations due to the overhead of embedding
and sparsification. By circumventing these pre-processing requirements and enabling reconfigurable, all-to-all virtual
connectivity, the VCPC provides a scalable solution for complex, higher-order, and dynamically changing combinatorial
optimisation problems.
Author Contributions
A.J.S. conceptualised the project. A.J.S. and H.Y. performed simulations and authored the manuscript, while A.J.S.,
H.Y., and F.H. wrote python implementations. A.M. oversaw hardware latency estimates, and M.L. and R.A. supervised
and secured funding for the project. All authors contributed to editing of the manuscript and substantially to discussions
on the results and ideas contained therein.
Acknowledgments
This work was supported by the European Innovation Council’s (EIC) grant number 101248376 (Q-TASTic).
16


**[Table p16.1]**
|  | PC DA QA VCPC |
| --- | --- |
| Dense Reconfigurable Higher-order | ✗ ✓ ✗ ✓ ✗ ✓ ✗ ✓ ✗ ✗ ✗ ✓ |

[CAPTION] Table 2: A comparison of the ability of different Nature inspired computing platforms to implement the types of


<!-- page 17 -->
A VCPC as a solver for higher-order, densely connected, or reconfigurable COPs
A PREPRINT
A
Hyperparameters
We indicate which hyperparameters were used for each problem, to ensure reproducibility of the results. When the
number of repeats is not specified, it was set to 1.
A.1
Hitting set problem
For the hitting set calculations in Fig. 3, we use the hyperparameters indicated in Table 3, where N is the number of
vertices in the hypergraph. In Fig. 3a, iterations refers to the product I = I × S, and we keep I fixed at one and only
vary S. Thus, I = S. All other parameters are as in Table 3. In all experiments we took A = 13 and B = 9.
Table 3: The hyperparameters used for hitting set instances.
Iters.
Steps
Reps.
Repls.
Temp. range
Swap
SA
5 × N
100
20
-
0.01 −1.1
-
PT
50 × N
-
10
20
0.5 −10
25
A.2
Travelling salesperson problem
For the TSP, hyperparameters are displayed in Table 4. Km denotes the number of clusters used at each level and Am is
the corresponding penalty weight used in the solver. A0 represents the penalty weight for the final N-city problem. The
penalty weight for the SA and PT experiments without clustering was set to A0. We used B = 1 for all experiments.
The SA and PT parameters shown were used at each clustering step. For experiments without clustering, the number of
iterations was multiplied by the number of clustering steps to ensure both methods were run for equal total update cycles.
The inverse temperature ranges for both the SA and PT simulations were linearly separated between 0.0001 −0.01.
Table 4: Hyperparameters used for the TSP simulations.
Instance
A0
Clustering hyperparameters
SA
PT
K1
A1
K2
A2
K3
A3
K4
A4
Steps
Iters.
Iters.
Swap Repls.
burma14
1000
4
1400
-
-
-
-
-
-
200
1000
10000
100
20
ulysses16 1500
8
2500
4
3000
-
-
-
-
200
1000
10000
100
20
ulysses22 1500
16
2000
8
2500
4
3000
-
-
200
1000
10000
100
20
berlin52
1000
32
1000
16
1500
8
1500
4
2000
200
1000
10000
100
20
A.3
Spin-glass problem
Finally, for the spin-glass instances, the hyperparameters are shown in Table 5. We include the parameters, PT-baseline,
which are the parameters used in the large PT run to estimate the baseline energies.
Table 5: The hyperparameters used for spin-glass simulations, including the baseline calculations.
Graph size
SA
PT
PT-baselines
Temp. range
Temp. range
Swaps
Repls.
Temp. range
Swaps
Repls.
Iters.
Reps.
N = 100, 250, 500
0.074 −0.74
1.0 −5.0
10
5
1.0 −5.0
10
10
3000
20
N = 750, 1024
0.074 −0.74
1.0 −5.0
10
10
0.3 −10
10
20
3000
20
17

[CAPTION] Table 3: The hyperparameters used for hitting set instances.

[CAPTION] Table 4: Hyperparameters used for the TSP simulations.

[CAPTION] Table 5: The hyperparameters used for spin-glass simulations, including the baseline calculations.


<!-- page 18 -->
A VCPC as a solver for higher-order, densely connected, or reconfigurable COPs
A PREPRINT
B
Strategies for implementing a VCPC in hardware
An efficient implementation of a VCPC relies on a hybrid optoelectronic architecture, as detailed in Ref. [14]. Unlike
conventional Ising machines that rely on physical coupling between components (such as capacitive coupling in CMOS
or inductive coupling in superconducting loops), the VCPC decouples the stochastic bit generation from the network
topology. The hardware consists of three primary subsystems: (1) a high-bandwidth photonic entropy source, (2) a
high-speed digital logic unit (such as an FPGA or ASIC) for bias (update drive) calculation, and (3) memory banks for
state and weight storage.
Entropy generation
The stochasticity of the p-bits is derived from a photonic Quantum Random Number Generator
(QRNG). The optical signal is detected and digitised to provide a continuous stream of random numbers, randU[−1, +1],
which are distributed to the update logic. The self-correcting nature and ultra-high speed of the entropy source allows
for high-quality randomness without the need to engineer microscopic noise sources at every individual node.
Virtual connectivity and bias calculation
The definition of the problem geometry is stored entirely in the digital
memory. For a system of N p-bits, the connectivity matrix J (or tensor for higher-order problems) is stored in
high-throughput memory (e.g., high bandwidth memory or on-chip SRAM). The calculation of the bias for a specific
p-bit is performed by the digital logic unit.
To update a p-bit i, the logic unit retrieves the current states m of all connected neighbours and the corresponding
coupling weights. The computing unit then performs a dot-product operation:
Ii =
X
j
Jijsj + hi.
(29)
Crucially for the timing estimates in Sec. 3.3.1, this summation is implemented in hardware using a parallel adder tree
architecture. In an adder tree, pairs of terms are summed in parallel at each stage, reducing the number of values to be
summed by half at each level of the tree. Consequently, the depth of the logic circuit—and thus the number of clock
cycles required to compute the sum—scales logarithmically with the number of connections, i.e., O(log2 N) for a fully
connected graph.
Parallel updates and clock frequency
As discussed in Sec. 2, graph colouring allows independent sets of p-bits
to be updated simultaneously. The system operates at a clock frequency f (e.g., 2.7 GHz as cited in Ref. [14]).
Since the connectivity is virtual, changing the problem geometry (reconfigurability) or increasing the interaction order
(higher-order problems) does not require physical rewiring; it requires only an update to the weights stored in memory
and the routing logic in the adder tree. This architecture directly informs the TTS derived in Sec. 3.3.1 where the time
per iteration is dominated by the latency of the adder tree depth.
C
Derivation of the Hamiltonians and update drives
C.1
Hitting set problem
C.1.1
QUBO formulation
In the hitting set problem, we seek a subset of vertices S ⊆V of minimum size such that every hyperedge r ∈R
contains at least one vertex from S. We represent this with binary variables sv ∈{0, 1}, where sv = 1 if vertex v ∈S
and sv = 0 otherwise. The constraint that every hyperedge r must be covered is equivalent to requiring that it is not the
case that all vertices in r are outside of S. Mathematically, for each r ∈R, we require:
Y
v∈r
(1 −sv) = 0.
(30)
To enforce this as an optimisation problem, we assign a large penalty A to any violated constraint and minimise the
total size of the set with weight B:
E(s) = A
X
r∈R
Y
v∈r
(1 −sv) + B
X
v∈V
sv.
(31)
Setting A > B ensures that any valid vertex cover (where the first term is zero) will always have a lower energy than
any invalid configuration.
18


<!-- page 19 -->
A VCPC as a solver for higher-order, densely connected, or reconfigurable COPs
A PREPRINT
C.1.2
Update drive
We deduce from Eqn. 31 that the update drive for the hitting set problem is:
Ik = E(s|sk = 0) −E(s|sk = 1)
= A


X
r∈R
k/∈r
Y
v∈r
(1 −sv) +
X
r∈R
k∈r
Y
v∈r
v̸=k
(1 −sv)

+ B
X
v∈V (G)
v̸=k
sv
−A


X
r∈R
k/∈r
Y
v∈r
(1 −sv)

−B
X
v∈V (G)
v̸=k
sv −B
= A
X
r∈R|
k∈r
Y
v∈r|
v̸=k
(1 −sv) −B
(32)
Note the dependence on the number of vertices in a hyperedge, or the dimension k, as well as the number of hyperedges,
or m.
C.2
Travelling salesperson problem
C.2.1
QUBO formulation
In order to formulate the TSP as a QUBO problem, we initially define a p-bit state matrix, S, which one-hot encodes
our Hamiltonian cycle:
S =




1
0
0
. . .
0
0
1
. . .
0
1
0
. . .
...
...
...
...




(33)
The rows represent the city visited {A, B, C, ...}, and the columns denote the tour ordering. The example shown in
Eqn. 33 corresponds to the tour: A →C →B. Calculating the associated tour cost is straightforward using the distance
matrix D:
Ecost =
N−1
X
i=0
N−1
X
j=0
N−2
X
k=0
Dij Sik Sj,k+1 +
N−1
X
i=0
N−1
X
j=0
Dij Si,N−1 Sj,0
(34)
where the second term adds an implicit return to the starting city. We also need to introduce penalty terms to ensure S
converges to a valid tour. There are two types of invalid tours: (a) a tour which visits more or less than one city in a
single visit and (b) a tour that visits the same city more or less than once. These constraints take the form:
Epen =
N−1
X
i=0
  N−1
X
k=0
Sik −1
 2
+
N−1
X
k=0
  N−1
X
i=0
Sik −1
 2
(35)
where the first and second terms correspond to constraints (a) and (b) respectively. Therefore, one can define the total
energy function, E, for a TSP by linearly combining Epen and Ecost:
E(S) = A
" N−1
X
i=0
  N−1
X
k=0
Sik −1
 2
+
N−1
X
k=0
  N−1
X
i=0
Sik −1
 2#
+ B
" N−1
X
i=0
N−1
X
j=0
N−2
X
k=0
Dij Sik Sj,k+1 +
N−1
X
i=0
N−1
X
j=0
Dij Si,N−1 Sj,0
#
(36)
where A and B are hyperparameters. The ratio between A and B specifies the comparative importance of minimising
the tour cost and ensuring the solver yields a valid tour. Conventionally, we set B = 1 and A is instance dependent,
which is set subject to A ≥max(D) [1]. Due to the competitive nature of the terms comprising E, it is practically
difficult to find the minimum tour using a choice of A which consistently yields a valid tour. Moreover, it is well
reported in Ref. [35] that the probability of finding the optimal tour, using standard SA, approaches zero for problems
involving N > 12 cities. This necessitates additional procedures to assist the solver in finding the ground state.
19


<!-- page 20 -->
A VCPC as a solver for higher-order, densely connected, or reconfigurable COPs
A PREPRINT
C.2.2
Update drive
Using the TSP energy function from Eqn. 36 we can deduce the update drive:
Iik = E(Sik = 0) −E(Sik = 1)
= A






N−1
X
k′=0
k′̸=k
Sik′ −1




2
−




N−1
X
k′=0
k′̸=k
Sik′




2

+ A






N−1
X
i′=0
i′̸=i
Si′k −1




2
−




N−1
X
i′=0
i′̸=i
Si′k




2

−B
" N−1
X
j=0
Dij Sj,k+1 +
N−1
X
i′=0
Di′i Si′,k−1
#
(37)
Note that we can infer the three colouring rules by inspecting each term in Iik. The first and second terms forbid
p-bits in the same row and column to be in the same update group, respectively. While the third term couples Iik to all
variables in Si,k±1, which means p-bits in adjacent columns are directly dependent and cannot be included in the same
update group.
C.3
Spin-glass problem
C.3.1
Ising to QUBO mapping
To map the Ising Hamiltonian H = −P
i<j Jijσiσj −P
i hiσi with σi ∈{−1, +1} to a binary QUBO problem
si ∈{0, 1}, we use the transformation σi = 2si −1:
E(s) = −
X
i<j
Jij(2si −1)(2sj −1) −
X
i
hi(2si −1)
= −
X
i<j
Jij(4sisj −2si −2sj + 1) −
X
i
hi(2si −1)
=
X
i<j
(−4Jij)sisj +
X
i

2
X
j̸=i
Jij −2hi

si + C.
(38)
Defining E(s) = P
i<j Qijsisj + P
i bisi + C, the parameters map as:
Qij = −4Jij
(39)
bi = 2
X
j̸=i
Jij −2hi,
(40)
where C is the constant energy offset. Minimising E(s) is equivalent to finding the ground state of the original Ising
system.
C.3.2
Update drive
Lastly, we show the update drive for the spin-glass problem:
20


<!-- page 21 -->
A VCPC as a solver for higher-order, densely connected, or reconfigurable COPs
A PREPRINT
Ik = E(s|sk = 0) −E(s|sk = 1)
=


X
i<j,i,j̸=k
Qijsisj +
X
i̸=k
bisi

−

X
i<j
Qijsisj +
X
i
bisi


sk=1
= −

X
j>k
Qkjsj +
X
i<k
Qiksi + bk


= −

X
j̸=k
Qkjsj + bk

.
(41)
Clearly, the number of terms which needs to be calculated is highly dependent on the density of Q.
D
Average group numbers for the hitting set problem
An important factor in considering which hypergraphs are efficient to run with the VCPC is the average group size when
partitioning p-bits into independent groups. Recall that for a graph, independent nodes are nodes which do not share an
edge; we generalise this for hypergraphs by defining independent nodes to be any group of nodes which, among them,
share no hyperedges.
The number of iterations I required to reach a solution quality q is reduced by a factor N/|G|, or the number of p-bits
divided by the number of groups. Equivalently, this speed-up factor is given by the average group size, ¯G.
In Fig. 10 we plot temperature profiles of the number of groups |G| for hypergraphs of size N = 500, 1000, and 5000,
for hypergraph dimensions in the range [2, 20] and for varying numbers of hyperedges. Interestingly, they appear almost
identical for the different hypergraph sizes; we can therefore safely assume that provided k is not too large (size 6 or
less) for all values of m, the number of groups is less than 10. As discussed in the main text, the factor of speed-up is
then N
10, which for bigger hypergraphs, is clearly a larger speed-up due to group updates. This however must offset the
increase in the time per iteration, incurred by the increased m and k for larger values of N.
E
The effect of dimension on the conversion of the simulated annealing algorithm
We investigate how the hypergraph dimension k of the hitting set problem affects the solution quality for the SA
algorithm. Recall that increasing the dimension of the hypergraph corresponds to increasing the order of the energy
function to be solved. To investigate this, in Fig. 11 we plot the solution quality found by the PC-SA algorithm for
hypergraphs of size 500 vertices and 200 edges, for varying values of k and varying number of iterations. Note that
group updates are not included here in the number of iterations.
For each point we sample 20 hypergraphs, and run 20 repeats in parallel3. While for k = 10 we see that the quality
approaches 1.05, for larger values of k the solution quality is worse for an equal number of iterations. This is possibly
due to the energy landscape becoming significantly more complex for higher-order problems, and therefore more
difficult to sample.
3We avoid calling these replicas, since for the case of PT replicas refers to the number of chains of different temperatures
21


<!-- page 22 -->
A VCPC as a solver for higher-order, densely connected, or reconfigurable COPs
A PREPRINT
100
200
300
400
500
m (number of hyperedges)
2.5
5.0
7.5
10.0
12.5
15.0
17.5
20.0
k (hyperedge size)
20
40
60
80
|G|
N = 500
(a)
200
400
600
800
1000
m (number of hyperedges)
2.5
5.0
7.5
10.0
12.5
15.0
17.5
20.0
k (hyperedge size)
20
40
60
80
|G|
N = 1000
(b)
1000
2000
3000
4000
5000
m (number of hyperedges)
2
4
6
8
10
12
14
16
18
20
k (hyperedge size)
10
20
30
40
50
60
70
80
90
|G|
N = 5000
(c)
Figure 10: Temperature plots showing the variation in the number of colour groups |G| with the hypergraph dimension
k and the number of hyperedges m, for a fixed number of vertices N. The chosen values of N were (a) N = 500, (b)
N = 1000, and (c) N = 5000.
F
Iterations against solution quality for the TSP problem
In Fig. 12 we compare the number of iterations taken to reach various solution qualities for the burma14 instance with
the four methods considered. We observe a similar trend in both Fig. 12a and Fig. 12b, where PC-PT is able to find
lower solutions of a lower quality in several orders of magnitude fewer iterations than PC-SA. This is to be expected,
as the lower temperature states are only explored by PC-SA during the latter iterations of the annealing process. The
PC-PT algorithm however, is immediately exposed to these lower temperature states, and therefore can explore them
constantly. Despite this initial improvement, on average the PC-SA reaches the best solution in less iterations than the
PC-PT, both with and without KMC. This is most likely because the SA schedule evolves β smoothly, allowing more
consistent convergence toward low-energy states, whereas the rigid temperature exchanges in the 20 replica PT can
interrupt local relaxation.
22

[CAPTION] Figure 10: Temperature plots showing the variation in the number of colour groups |G| with the hypergraph dimension


<!-- page 23 -->
A VCPC as a solver for higher-order, densely connected, or reconfigurable COPs
A PREPRINT
0
20000
40000
60000
80000 100000 120000 140000
Iterations
1.05
1.10
1.15
1.20
1.25
1.30
1.35
1.40
1.45
Solution quality q
PC-SA (k = 40)
PC-SA (k = 30)
PC-SA (k = 20)
PC-SA (k = 10)
Figure 11: The average solution quality q against iterations I obtained by the PC-SA algorithm for varying hypergraph
dimensions. Dimensions k = 10, 20, 30 and 40 were tested. Other parameters, such as the number of vertices and
edges of the hypergraph, were kept constant. Each point was averaged over the best obtained over 20 repeats for 20
hypergraphs.
We also show how the distribution of tours obtained changes with the addition of KMC in Fig. 13. The shift in the
distribution shows clearly the advantage of employing KMC.
20000
40000
60000
80000
100000
Total iterations 
1
2
3
4
5
6
7
Solution quality q
PC-SA
PC-PT
(a)
20000
40000
60000
80000
100000
Total iterations 
1.0
1.5
2.0
2.5
3.0
Solution quality q
KMC - PC-SA
KMC - PC-PT
(b)
Figure 12: The best solution qualities at various numbers of total iterations I of the SA and PT algorithms without (a)
and with (b) KMC. The error bars shows the range of solution qualities obtained for 100 runs on the burma14 instance.
23


**[Table p23.1]**
| PC-SA 7 PC-PT 6 q quality 5 4 Solution 3 2 1 20000 40000 60000 80000 100000 Total iterations | KMC - PC-SA KMC - PC-PT 3.0 q quality 2.5 Solution 2.0 1.5 1.0 20000 40000 60000 80000 100000 Total iterations |
| --- | --- |

[CAPTION] Figure 11: The average solution quality q against iterations I obtained by the PC-SA algorithm for varying hypergraph

[CAPTION] Figure 12: The best solution qualities at various numbers of total iterations I of the SA and PT algorithms without (a)


<!-- page 24 -->
A VCPC as a solver for higher-order, densely connected, or reconfigurable COPs
A PREPRINT
6
8
10
12
14
16
18
20
22
24
Tour cost (×1000)
0.0
0.1
0.2
0.3
0.4
0.5
0.6
Probability
Optimal
PC-SA
KMC - PC-SA
Figure 13: Performances of standard PC-SA and KMC - PC-SA solvers on the berlin52 instance [37], where the
optimal tour length is 7542.
24


**[Table p24.1]**
|  | Optimal PC-SA KMC - PC-SA |
| --- | --- |

[CAPTION] Figure 13: Performances of standard PC-SA and KMC - PC-SA solvers on the berlin52 instance [37], where the


<!-- page 25 -->
A VCPC as a solver for higher-order, densely connected, or reconfigurable COPs
A PREPRINT
References
[1] Lucas, A. Ising formulations of many np problems. Front. Phys. 2, DOI: 10.3389/fphy.2014.00005 (2014).
[2] Patel, S., Chen, L., Canoza, P. & Salahuddin, S. S. Ising model optimization problems on a fpga accelerated
restricted boltzmann machine. ArXiv abs/2008.04436 (2020).
[3] Leleu, T. et al. Scaling advantage of chaotic amplitude control for high-performance combinatorial optimization.
Commun. Phys. 4, 266, DOI: 10.1038/s42005-021-00768-0 (2021).
[4] Feynman, R. P. Simulating physics with computers. Int. J. Theor. Phys. 21, 467–488, DOI: 10.1007/BF02650179
(1982).
[5] Ushijima-Mwesigwa, H., Negre, C. F. A. & Mniszewski, S. M. Graph partitioning using quantum annealing on
the d-wave system (2017). 1705.03082.
[6] Aramon, M. et al. Physics-inspired optimization for quadratic unconstrained problems using a digital annealer.
Front. Phys. Volume 7 - 2019, DOI: 10.3389/fphy.2019.00048 (2019).
[7] Moy, W. et al. A 1,968-node coupled ring oscillator circuit for combinatorial optimization problem solving. Nat.
Electron. 5, 310–317, DOI: 10.1038/s41928-022-00749-3 (2022).
[8] Finocchio, G. et al. The promise of spintronics for unconventional computing. J. Magn. Magn. Mater. 521,
167506, DOI: 10.1016/j.jmmm.2020.167506 (2021).
[9] Grollier, J. et al. Neuromorphic spintronics. Nat. Electron. 3, 360–370, DOI: 10.1038/s41928-019-0360-9 (2020).
Publisher Copyright: © 2020, Springer Nature Limited.
[10] Camsari, K. Y., Faria, R., Sutton, B. M. & Datta, S. Stochastic p-bits for invertible logic. Phys. Rev. X 7, 031014,
DOI: 10.1103/PhysRevX.7.031014 (2017).
[11] Valiante, E., Hernandez, M., Barzegar, A. & Katzgraber, H. G. Computational overhead of locality reduction in
binary optimization problems. Comput. Phys. Commun. 269, 108102, DOI: 10.1016/j.cpc.2021.108102 (2021).
[12] Nikhar, S., Kannan, S., Aadit, N. A., Chowdhury, S. & Camsari, K. Y. All-to-all reconfigurability with sparse and
higher-order ising machines. Nat. Commun. 15, 8977, DOI: 10.1038/s41467-024-53270-w (2024).
[13] Okada, S., Ohzeki, M., Terabe, M. & Taguchi, S. Improving solutions by embedding larger subproblems in a
d-wave quantum annealer. Sci. Reports 9, 2098, DOI: 10.1038/s41598-018-38388-4 (2019).
[14] Aboushelbaya, R., Moslein, A., Azar, H., Tanhaei, H. & von der Leyen, M. Self-correcting high-speed opto-
electronic probabilistic computer (2025). 2511.04300.
[15] Caprara, A., Toth, P. & Fischetti, M. Algorithms for the set covering problem. Annals Oper. Res. 98, 353–371,
DOI: 10.1023/A:1019225027893 (2000).
[16] Beasley, J. An algorithm for set covering problem. Eur. J. Oper. Res. 31, 85–93, DOI: https://doi.org/10.1016/
0377-2217(87)90141-X (1987).
[17] Gohil, A., Tayal, M., Sahu, T. & Sawalpurkar, V. Travelling salesman problem: Parallel implementations &
analysis (2022). 2205.14352.
[18] Bock, S., Bomsdorf, S., Boysen, N. & Schneider, M. A survey on the traveling salesman problem and its variants
in a warehousing context. Eur. J. Oper. Res. 322, 1–14, DOI: https://doi.org/10.1016/j.ejor.2024.04.014 (2025).
[19] Osaba, E., Yang, X.-S. & Del Ser, J. Chapter 9 - traveling salesman problem: a perspective review of recent
research and new results with bio-inspired metaheuristics. In Yang, X.-S. (ed.) Nature-Inspired Computation
and Swarm Intelligence, 135–164, DOI: https://doi.org/10.1016/B978-0-12-819714-1.00020-8 (Academic Press,
2020).
[20] Handley, L. B., Petigura, E. A. & Misic, V. V. Solving the traveling telescope problem with mixed integer linear
programming (2023). 2310.18497.
[21] Kochenberger, G. et al. The unconstrained binary quadratic programming problem: a survey. J. Comb. Optim. 28,
58–81, DOI: 10.1007/s10878-014-9734-0 (2014).
[22] Karp, R. M. Reducibility among combinatorial problems. In Miller, R. E., Thatcher, J. W. & Bohlinger, J. D.
(eds.) Complexity of Computer Computations: Proceedings of a symposium on the Complexity of Computer
Computations, held March 20–22, 1972, at the IBM Thomas J. Watson Research Center, Yorktown Heights, New
York, and sponsored by the Office of Naval Research, Mathematics Program, IBM World Trade Corporation, and
the IBM Research Mathematical Sciences Department, 85–103, DOI: 10.1007/978-1-4684-2001-2_9 (Springer
US, 1972).
25


<!-- page 26 -->
A VCPC as a solver for higher-order, densely connected, or reconfigurable COPs
A PREPRINT
[23] Grimaldi, A. et al. Spintronics-compatible approach to solving maximum-satisfiability problems with probabilistic
computing, invertible logic, and parallel tempering. Phys. Rev. Appl. 17, 024052, DOI: 10.1103/PhysRevApplied.
17.024052 (2022).
[24] Raimondo, E. et al. High-performance and reliable probabilistic ising machine based on simulated quantum
annealing (2025). 2503.13015.
[25] Romá, F., Risau-Gusman, S., Ramirez-Pastor, A., Nieto, F. & Vogel, E. The ground state energy of the ed-
wards–anderson spin glass model with a parallel tempering monte carlo algorithm. Phys. A: Stat. Mech. its Appl.
388, 2821–2838, DOI: 10.1016/j.physa.2009.03.036 (2009).
[26] Hasselgren, F., Al-Hasso, M. O., Searle, A., Tindall, J. & von der Leyen, M. Probabilistic computing optimization
of complex spin-glass topologies (2025). 2510.23419.
[27] Chowdhury, S. et al. Pushing the boundary of quantum advantage in hard combinatorial optimization with
probabilistic computers. Nat. Commun. 16, 9193, DOI: 10.1038/s41467-025-64235-y (2025).
[28] Dobrynin, D. et al. Energy landscapes of combinatorial optimization in ising machines. Phys. Rev. E 110, DOI:
10.1103/physreve.110.045308 (2024).
[29] Albash, T., Spedalieri, F., Hen, I., Pudenz, K. & Tallant, G. Solving large optimization problems with restricted
quantum annealers. IEEE High Perform. Extrem. Comput. (2016).
[30] Pelofske, E., Hahn, G. & Djidjev, H. Solving large minimum vertex cover problems on a quantum annealer.
In Proceedings of the 16th ACM International Conference on Computing Frontiers, CF ’19, 76–84, DOI:
10.1145/3310273.3321562 (ACM, 2019).
[31] Sajeeb, M. M. H. et al. Scalable connectivity for ising machines: Dense to sparse. Phys. Rev. Appl. 24, 014005,
DOI: 10.1103/kx8m-5h3h (2025).
[32] Cao, Y., Jiang, S., Perouli, D. & Kais, S. Solving set cover with pairs problem using quantum annealing. Sci.
Reports 6, 33957, DOI: 10.1038/srep33957 (2016).
[33] Corder, K., Monaco, J. V. & Vindiola, M. M. Solving vertex cover via ising model on a neuromorphic processor. In
2018 IEEE International Symposium on Circuits and Systems (ISCAS), 1–5, DOI: 10.1109/ISCAS.2018.8351248
(2018).
[34] Zhu, G. A new view of classification in astronomy with the archetype technique: An astronomical case of the
np-complete set cover problem (2016). arXiv:1606.07156.
[35] Dan, A., Shimizu, R., Nishikawa, T., Bian, S. & Sato, T. Clustering approach for solving traveling salesman
problems via ising model based solver. In 2020 57th ACM/IEEE Design Automation Conference (DAC), 1–6, DOI:
10.1109/DAC18072.2020.9218695 (2020).
[36] Jaradat, A., Matalkeh, B. & Diabat, W. Solving traveling salesman problem using firefly algorithm and k-means
clustering. In 2019 IEEE Jordan International Joint Conference on Electrical Engineering and Information
Technology (JEEIT), 586–589, DOI: 10.1109/JEEIT.2019.8717463 (2019).
[37] Reinelt, G. Tsplib—a traveling salesman problem library. ORSA J. on Comput. 3, 376–384 (1991).
[38] Ayodele, M. Penalty weights in qubo formulations: Permutation problems. In Pérez Cáceres, L. & Verel, S. (eds.)
Evolutionary Computation in Combinatorial Optimization, 159–174 (Springer International Publishing, Cham,
2022).
[39] Delacour, C., Sajeeb, M. M. H., Hespanha, J. a. P. & Camsari, K. Y. Two-dimensional parallel tempering for
constrained optimization. Phys. Rev. E 112, L023301, DOI: 10.1103/mr2n-qqrb (2025).
[40] Bunaiyan, S., Delacour, C., Chowdhury, S., Lee, K. & Camsari, K. Y. Isingformer: Augmenting parallel tempering
with learned proposals (2025). 2509.23043.
[41] Mézard, M., Parisi, G. & Virasoro, M. A. Spin Glass Theory and Beyond: An Introduction to the Replica Method
and Its Applications, vol. 9 of World Scientific Lecture Notes in Physics (World Scientific Publishing Company,
Singapore, 1987).
[42] Edwards, S. F. & Anderson, P. W. Theory of spin glasses. J. Phys. F: Met. Phys. 5, 965, DOI: 10.1088/0305-4608/
5/5/017 (1975).
[43] Sherrington, D. & Kirkpatrick, S. Solvable model of a spin-glass. Phys. Rev. Lett. 35, 1792–1796, DOI:
10.1103/PhysRevLett.35.1792 (1975).
[44] Binder, K. & Young, A. P. Spin glasses: Experimental facts, theoretical concepts, and open questions. Rev. Mod.
Phys. 58, 801–976, DOI: 10.1103/RevModPhys.58.801 (1986).
26


<!-- page 27 -->
A VCPC as a solver for higher-order, densely connected, or reconfigurable COPs
A PREPRINT
[45] Rodríguez-Camargo, C. D., Mojica-Nava, E. A. & Svaiter, N. F. Sherrington-kirkpatrick model for spin glasses:
Solution via the distributional zeta function method. Phys. Rev. E 104, 034102, DOI: 10.1103/PhysRevE.104.
034102 (2021).
[46] Panchenko, D.
The sherrington-kirkpatrick model:
An overview.
J. Stat. Phys. 149, DOI: 10.1007/
s10955-012-0586-7 (2012).
[47] Erd˝os, P. & Rényi, A. On the evolution of random graphs. Publ. Math. Inst. Hung. Acad. Sci. 5, 17 (1960).
[48] Borowiecki, P. Computational aspects of greedy partitioning of graphs. J. Comb. Optim. 35, 641–665 (2018).
[49] Bollobás, B. On complete subgraphs of different orders. Math. Proc. Camb. Philos. Soc. 79, 19–24, DOI:
10.1017/S0305004100052063 (1976).
27