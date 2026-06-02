[2510.09567] Safe, Untrusted, “Proof-Carrying” AI Agents: toward the agentic lakehouse Thanks to [1] for coming up with a great title (a long time ago, for a different type of agents).

Safe, Untrusted, “Proof-Carrying” AI Agents:

toward the agentic lakehouse

†

†  thanks:  Thanks to  [ 1 ]  for coming up with a great title (a long time ago, for a different type of agents).

Jacopo Tagliabue

Ciro Greco

Abstract

Data lakehouses run sensitive workloads, where AI-driven automation raises concerns about trust, correctness, and governance. We argue that API-first, programmable lakehouses provide the right abstractions for  safe-by-design , agentic workflows. Using Bauplan as a case study, we show how data branching and declarative environments extend naturally to agents, enabling reproducibility and observability while reducing the attack surface. We present a proof-of-concept in which agents repair data pipelines using correctness checks inspired by proof-carrying code. Our prototype demonstrates that untrusted AI agents can operate safely on production data and outlines a path toward a fully agentic lakehouse.

I

Introduction

The data lakehouse is the  de facto  cloud architecture for analytics and Artificial Intelligence (AI) workloads  [ 2 ,  3 ] , thanks to storage-compute decoupling, multi-language support, and unified table semantics. As reasoning and tool usage in Large Language Models (LLMs) improve  [ 4 ] , autonomous decisions (“AI agents”) are both supported by, and targeted at, cloud infrastructure: to what extent can agents manage the data lifecycle in a lakehouse?

Prima facie , the question appears both too hard and too broad. On one hand, lakehouses are distributed systems built for the collaboration of human teams on sensitive production data, not point-wise tasks immediately suitable for end-to-end automation. On the other, it is unclear how to prioritize agentic use cases across such heterogeneous platforms.  This  paper is a preliminary answer to these challenges: we detail lakehouse abstractions suitable for automation, and operationalize a prototype for an important use case:  repairing data pipelines .

Pipelines are a compelling case study for three reasons: first, they cover a large portion of lakehouse workloads, measured both by developer time  [ 5 ]  and overall compute  [ 6 ] . Second, data engineers spend a significant amount of their time fixing them  [ 7 ,  8 ] . Finally, repairing pipelines is a canary test for agent penetration in high-stakes non-trivial scenarios, which are often hard even for expert humans  [ 9 ,  10 ] . We summarize our contributions as follows:

1.

we introduce abstractions to model the data life-cycle in a programmable lakehouse  [ 11 ] , i.e. building and executing cloud pipelines entirely through  code . We argue that traditional systems resist automation mostly because of heterogeneous interfaces and complex access patterns, while code is the  lingua franca  suitable for agents, cloud systems, and human supervisors;

2.

we review common objections to automating high-stakes workloads in light of the proposed abstractions: in particular, we argue that our model promotes trustworthiness and correctness both in data and code artifacts;

3.

we release working code  1

1  1 Open source code is available at https://github.com/BauplanLabs/the-agentic-lakehouse.

, showing a proof of concept for self-repairing pipelines using  Bauplan  as a lakehouse and an agentic loop. Starting from this prototype, we conclude by outlining practical next steps for a full agentic lakehouse.

The paper is organized as follows. After reviewing agent-friendly abstractions (Section

II  ), we address key safety objections for high-stakes scenarios (Section

III  ). Once safety is established, we describe a ReAct  [ 12 ]  loop built on these abstractions (Section

IV  ). We put forward our working prototype as a feasibility demonstration of safe-by-design data agents, not as a full-fledged experimental benchmark.

We believe that sharing working code is of great value to the community, especially in times of quickly shifting mental models. However, it is important to remember that our fundamental insights – programmability and safety – can be replicated independently of the chosen APIs. For these reasons, we believe our paper to be valuable to a wide range of practitioners: on one hand, those looking for a new mental map of this uncharted territory; on the other, those looking to be inspired by tinkering with existing implementations and inspecting systems working at scale.

II

A programmable lakehouse

In a  programmable  lakehouse, the  entire  data life-cycle – data, user and infrastructure management, pipeline and query execution, runtime observability – is exposed through code abstractions: server-side APIs, SDK methods, CLI shortcuts. In the rest of the paper,  Bauplan  snippets will be used as sample implementation, but the platform’s composable nature makes it easy to replicate these functionalities in different architectures.  2

2  2 We refer the interested reader to existing papers, in particular  [ 11 ,  13 ,  14 ] .

We break down the pipeline life-cycle into two major components: pipeline definition and pipeline execution.

II-A

Pipeline definition

A pipeline is a DAG of transformations. A DAG starts from source tables, which are progressively cleaned, augmented, aggregated through the transformation code (expressed in SQL or Python). A successful execution produces intermediate and final data assets, which are then consumed by downstream systems  [ 15 ] . Fig.

2

shows a pipeline (hence, P) used as a recurring example throughout the paper. Taking two tables from the NYC taxi dataset  [ 16 ] , P defines two new tables (A and B), based on two transformations (1 and 2). The Bauplan implementation for P is as follows:  3

3  3 Snippets are simplified in the interest of space: full code is available in the open source repository.

Listing 1:  P as the Python file  p.py

⬇

@bauplan  .  model  (  materialization  =  "REPLACE"  ,

name  =  "A"  )

@bauplan  .  python  (  "3.10"  ,

pip  ={  "pandas"  :

"2.0"  })

def

join_and_filter  (

trips  =  bauplan  .  Model  (  "taxi_trips"  ),

zones  =  bauplan  .  Model  (  "taxi_zones"  )

):

#

some

transformation

here...

return

trips  .  join  (  zones  ).  do_something  ()

@bauplan  .  model  (  materialization  =  "REPLACE"  ,

name  =  "B"  )

@bauplan  .  python  (  "3.11"  ,

pip  ={  "pandas"  :

"1.5.3"  })

def

clean_and_transform  (

data  =  bauplan  .  Model  (  "join_and_filter"  )

):

return

data  .  do_something  ()

Two important design choices are worth highlighting in connection to our safety discussion (Section

III  ):

•

Function-as-a-service (FaaS) abstractions : business logic is expressed in the body of plain vanilla functions with the signature

T  ​  a  ​  b  ​  l  ​  e  ​

(  s  )

→

T  ​  a  ​  b  ​  l  ​  e

Table(s)\rightarrow Table

. DAGs are functions chained through naming convention. These abstractions naturally map to a serverless runtime, which can execute the requested computation efficiently  [ 11 ] ;

•

declarative I/O and infrastructure : functions are fully isolated (e.g. two functions, two versions of  pandas ) and their Python environment is specified declaratively  [ 17 ] . Reading tables and writing artifacts back to the lake is also fully declarative: users specify the needed inputs and desired output, the platform performs the corresponding physical operations.

II-B

Pipeline execution

A human (or an agent) with the proper access can execute  p.py  by simply installing the  bauplan  package, and running it from the terminal, without any additional steps – no Docker, no Terraform, no JDBC clients:  4

4  4 To get a first-person perspective on the developer experience, the reader is invited to pause and watch a recorded run before continuing: https://www.loom.com/share/99ac0d5b5f944fc9aeef132bf