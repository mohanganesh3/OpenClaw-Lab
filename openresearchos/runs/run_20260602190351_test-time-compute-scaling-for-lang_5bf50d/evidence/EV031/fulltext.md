[2311.17007] Computational Hypergraph Discovery, a Gaussian Process framework for connecting the dots

†

†  † Department of Computing and Mathematical Sciences, California Institute of Technology, Pasadena, CA 91125, USA

†

†  ∗ Jet Propulsion Laboratory, California Institute of Technology, Pasadena, CA 91109, USA

†

† E-mail addresses:  tbourdai@caltech.edu ,  pau@caltech.edu ,  yxjmath@caltech.edu ,

rsb@caltech.edu ,  nicolas.f.rouquette@jpl.nasa.gov ,  owhadi@caltech.edu

Computational Hypergraph Discovery, a Gaussian Process framework for connecting the dots

Théo Bourdais † , Pau Batlle † , Xianjin Yang † , Ricardo Baptista † , Nicolas Rouquette ∗

and Houman Owhadi †

Abstract.

Most scientific challenges can be framed into one of the following three levels of complexity of function approximation.  Type 1:  Approximate an unknown function given input/output data.  Type 2:  Consider a collection of variables and functions, some of which are unknown, indexed by the nodes and hyperedges of a hypergraph (a generalized graph where edges can connect more than two vertices). Given partial observations of the variables of the hypergraph (satisfying the functional dependencies imposed by its structure), approximate all the unobserved variables and unknown functions.  Type 3:  Expanding on Type 2, if the hypergraph structure itself is unknown, use partial observations of the variables of the hypergraph to discover its structure and approximate its unknown functions.
While most Computational Science and Engineering and Scientific Machine Learning challenges can be framed as Type 1 and Type 2 problems, many scientific problems can only be categorized as Type 3. Despite their prevalence, these Type 3 challenges have been largely overlooked due to their inherent complexity. Although Gaussian Process (GP) methods are sometimes perceived as well-founded but old technology limited to Type 1 curve fitting, their scope has recently been expanded to Type 2 problems.
In this paper, we introduce an interpretable GP framework for Type 3 problems, targeting the data-driven discovery and completion of computational hypergraphs.
Our approach is based on a kernel generalization of (1) Row Echelon Form reduction from linear systems to nonlinear ones and (2) variance-based analysis. Here, variables are linked via GPs, and those contributing to the highest data variance unveil the hypergraph’s structure.
We illustrate the scope and efficiency of the proposed approach with applications to (algebraic) equation discovery, network discovery (gene pathways, chemical, and mechanical), and raw data analysis.

1.  Introduction.

1.1.  The three levels of complexity of function approximation.

Alfred North Whitehead stated in 1911: "Civilization advances by extending the number of important operations we can perform without thinking about them." The automation of arithmetic and calculus through the introduction of calculators and computers serves as a testament to such transformative shifts.
In line with this perspective, the resolution of most scientific challenges could be facilitated by automating the resolution of the following three levels of increasing complexity of function approximation (see Fig.

1  ).

•

Type 1 (Regression):  Approximate an unknown function given (possibly noisy) input/output data.

•

Type 2 (Hypergraph Completion):  Consider a collection of variables and functions, some of which are unknown, indexed by the nodes and hyperedges of a hypergraph (a generalized graph where edges can connect more than two vertices). Given partial observations of the variables of the hypergraph (satisfying the functional dependencies imposed by its structure), approximate all the unobserved variables and unknown functions.

•

Type 3 (Hypergraph Discovery):  Expanding on Type 2, if the hypergraph structure itself is unknown, use partial observations (of subsets of the variables of the hypergraph) to discover its structure (the hyperedges and possibly the missing vertices) and then approximate its unknown functions and unobserved variables.

Figure 1 .

The three levels of complexity of function approximation.

The scope of these problems is immense.
Numerical approximation, Supervised Learning, and Operator Learning can all be formulated as type 1 problems (possibly with functional inputs/outputs spaces).
Type 2 problems include (see Fig.

4  ) solving and learning (possibly stochastic) ordinary or partial differential equations, Deep Learning, dimension reduction, reduced-ordered modeling, system identification, closure modeling, etc.
The scope of Type 3 problems extends well beyond Type 2 problems and includes applications involving model/equation/network discovery and reasoning with raw data. While most problems in Computational Sciences and Engineering (CSE) and Scientific Machine Learning (SciML) can be framed as Type 1 and Type 2 challenges, many problems in science can only be categorized as Type 3 problems. Despite their prevalence, these Type 3 challenges have been largely overlooked due to their inherent complexity.

Causal inference methods and probabilistic graphs, as discussed in Section

3.3  , and sparse regression methods  [ 11 ,  4 ] , offer potential avenues for addressing Type 3 problems. However, it is important to note that their application to these problems necessitates additional assumptions.
Causal inference models, for instance, typically assume randomized data and some level of access to the data generation process or its underlying distributions. We emphasize that our proposed approach does not seek to replace causal inference methods but rather seeks to encode a different type of information into the structure of the graph, i.e., the functional dependencies between variables (see Sec.

3.3

for a more detailed discussion).
Sparse regression methods, on the other hand, rely on the assumption that functional dependencies have a sparse representation within a known basis.
In this paper, we do not impose these assumptions, and thus, these particular techniques may not be applicable.

Figure 2 .

Formal description of Type 2 problems.

1.2.  Type 2 problems: Formal description and GP-based Computational Graph Completion

1.2.1.  Formal description of Type 2 problems

Consider a computational graph (as illustrated in Fig.

2  .(a)) where nodes represent variables and edges are directed and they represent functions. These functions may be known or unknown. In Fig.

2  .(a), edges associated with unknown functions (

f

5  ,  1

subscript  𝑓

5  1

f_{5,1}

,

f

1  ,  2

subscript  𝑓

1  2

f_{1,2}

,

f

3  ,  6

subscript  𝑓

3  6

f_{3,6}

) are colored in red, and those associated with known functions (

f

2  ,  5

subscript  𝑓

2  5

f_{2,5}

) are colored in black.
Round nodes are utilized to symbolize variables, which are derived from the concatenation of other variables (e.g, in Fig.

2  .(a),

x  3

=

(

x  2

,

x  4

)

subscript  𝑥  3

subscript  𝑥  2

subscript  𝑥  4

x_{3}=(x_{2},x_{4})

). Therefore, the underlying graph is, in fact, a hypergraph where functions may map groups of variables to other groups of variables, and we use round nodes to illustrate the grouping step.
Given partial observations derived from

N

𝑁

N

samples of the graph’s variables, we introduce a problem, termed a Type 2 problem, focused on approximating all unobserved variables and unknown functions.
Using Fig.

2  .(a)-(b) as an illustration we call a vector

(

X

s  ,  1

,  …  ,

X

s  ,  6

)

subscript  𝑋

𝑠  1

…

subscript  𝑋

𝑠  6

(X_{s,1},\ldots,X_{s,6})

a sample from the graph if its entries are variables satisfying the functional dependencies imposed by the structure of the graph (i.e.,

X

s  ,  1

=

f

5  ,  1

​

(

X

s  ,  5

)

subscript  𝑋

𝑠  1

subscript  𝑓

5  1

subscript  𝑋

𝑠  5

X_{s,1}=f_{5,1}(X_{s,5})

,

X

s  ,  2

=

f

1  ,  2

​

(

X

2  ,  s

)

subscript  𝑋

𝑠  2

subscript  𝑓

1  2

subscript  �