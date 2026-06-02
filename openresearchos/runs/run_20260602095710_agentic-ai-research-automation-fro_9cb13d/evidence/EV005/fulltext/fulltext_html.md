[2511.16402] Trustworthy AI in the Agentic Lakehouse: from Concurrency to Governance

Trustworthy AI in the Agentic Lakehouse: from Concurrency to Governance

Jacopo Tagliabue 1 ,
Federico Bianchi 2 ,
Ciro Greco 1

Corresponding author.

Abstract

Even as AI capabilities improve, most enterprises do not consider agents trustworthy enough to work on production data. In this paper, we argue that the path to trustworthy agentic workflows begins with solving the infrastructure problem first: traditional lakehouses are not suited for agent access patterns, but if we design one around  transactions , governance follows. In particular, we draw an operational analogy to MVCC in databases and show why a direct transplant fails in a decoupled, multi-language setting. We then propose an agent-first design,  Bauplan , that  reimplements  data and compute isolation in the lakehouse. We conclude by sharing a reference implementation of a self-healing pipeline in  Bauplan , which seamlessly couples agent reasoning with all the desired guarantees for  correctness  and  trust .

1  Introduction

The lakehouse is the enterprise standard for data and AI workloads in the cloud  ( Zaharia2021LakehouseAN ) . Notwithstanding the steady progress in coding and tool usage by Large Language Models (LLMs)  ( shen2024llm ) , production deployments of AI agents have rarely, if ever, targeted lakehouse use cases: even before considering the specificity of data  vs.  software engineering, the primary obstacles in industry are trust and governance: it is unclear how a human-centered lakehouse can provide the necessary guarantees for an agent-first world. To make a vivid example, imagine empowering a code assistant with access to  your  lakehouse: what if the agent drops a table? Or if it pollutes the lake with hallucinated data?

On the research side, the data management community is raising similar concerns: it is unlikely that data systems designed for small, expensive human teams can successfully adapt to the access patterns of a  swarm  of cheap AI agents  ( tagliabue2025safeuntrustedproofcarryingai ) . In particular, managing agents that run queries and build data pipelines on a lakehouse means managing concurrency on a different scale than what traditional OLAP systems are designed for  ( liu2025supportingaioverlordsredesigning ) .

In this position paper, we argue that the industry and the research concerns above are best thought of as  two sides of the same coin . In a nutshell, correct concurrent workloads require us to solve the isolation of data and compute through a unified API, reducing the governance challenge to the well-known pattern of API-based access control. If we focus our engineering effort on making sure multiple agents can work on the lakehouse without catastrophic consequences, we will get an easy and principled path to governance as well.

We know that this path is not impossible because – at the very least –  it has already happened once ; in particular, we first establish a connection with the theory of multi-version concurrency control (MVCC)  ( 10.5555/12518 )  in databases. Monolithic, SQL-only transactional databases (e.g.,  Postgres ) have evolved as a sophisticated answer to concurrency issues – declarative abstractions, isolated processes, data snapshots allow correctness in the face of multiple users, with RBAC access control layered on top for governance  ( 10.1145/501978.501980 ) . The lessons from the data management field are precious, but a direct mapping of the existing techniques will  not  work in a distributed, heterogeneous system such as a lakehouse. Alas, not all hope is lost. We describe novel abstractions and isolation primitives for an agent-first lakehouse  Bauplan . We show how to accommodate the usage patterns of swarms of agents, and easily derive the required governance: a concise, narrow API surface is much easier to protect as opposed to the plethora of tools in traditional lakehouses.

This position paper is organized as follows: Section

2

introduces concurrency ideas from the MVCC literature, to provide a concrete, extremely successful mental model that guides our search for similar lakehouse primitives. In Section

3  , we show that naively mapping MVCC concepts won’t work, before describing our proposal for lakehouse-native concepts in Section

4  . We conclude with an agentic, open-source implementation of a complex data engineering task: as the landscape is evolving quickly, we do believe that a working system is a valuable reference for practitioners coming from both the agentic and the data management field; our principles, however, are indeed tool-agnostic.

2  The MVCC mental model

Borrowing the metaphor from

10.5555/3299537  , we could think of the job of a database as giving each user the “illusion” of being the only user in play, while juggling safely and transparently the workloads of many such users. Correctness in the presence of concurrency is handled through  transactions

( 10.5555/12518 ) : readers only see a consistent view of the data, and writers atomically publish all the changes or, in case of error, none of them. For the current purposes, we carry along the entire paper a three-way partition of the conceptual space: data isolation, compute isolation and programming abstractions.

Figure 1:

The MVCC mental model :

U  1

U_{1}

starts his transaction, which at the end returns the value of

B  B

– effectively, his code works  as if

U  2

U_{2}

’s transaction had never happened.

2.1  Data Isolation

While the exact details are complex  ( cerone_et_al:LIPIcs.CONCUR.2015.58 ) , the basic model of data isolation is straightforward. Inside of transaction boundaries, processes read data always  as if  the database was frozen at the start; in order to do so, the system needs to maintain a reference to the state of the data at any point in time (a  snapshot ). In Fig.

1  ,

U  1

U_{1}

’s read of

U  2

U_{2}

’s value will return

500  500

as if

U  2

U_{2}

’s update never started. On the write path, transactions are used to prevent incorrect updates and race conditions: two transactions won’t both commit conflicting writes to the same key based on stale reads. Importantly, none of these storage-level implementations is exposed to the final user – a crucial design point to which we will return to below.

2.2  Compute Isolation

Each transaction gets processed in isolation, with no data sharing in between – from the point of view of the single transaction, it is  as if  that is the only process active at that time. In a SQL database there is no equivalent of package management, resulting in a small toolchain as opposed to a more heterogeneous compute environment where isolation is not as obvious.

2.3  Programming Abstractions

Traditional databases leverage a declarative language such as SQL which simplifies the relationship between the system and its users. Users express  what  data they wish to read or write, the query engine and the storage engine figure out  how  that should happen. This is a crucial design choice for at least two independent reasons:

•

correctness : the complexity of coupling data and compute isolation within transactions is offloaded entirely to the platform, avoiding the common pitfalls of  ad hoc  transactions that plague even popular open source projects  ( 10.1145/3638553 ) ;

•

performance : while each user is given the illusion of being alone, the platform may re-use transparently work done for one user for another one (e.g., caching).

A final consequence deserves a special mention,  trust . Since the only way to interact with the system is at the logical layer, protecting the physical representation of the data is now as easy as implementing RBAC over users – unauthorized code does not even go beyond the planning module, ensuring no contamination at the physical layer.

As we shall see in the ensuing section, the properties that make MVCC workable in a monol