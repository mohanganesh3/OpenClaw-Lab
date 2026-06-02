[2503.21602] GenEdit: Compounding Operators and Continuous Improvement to Tackle Text-to-SQL in the Enterprise

GenEdit: Compounding Operators and Continuous Improvement to Tackle Text-to-SQL in the Enterprise

Karime Maamari

\institution Distyl AI
 \city  \country

karime@distyl.ai

,

Connor Landy

\institution Distyl AI
 \city  \country

connor@distyl.ai

and

Amine Mhedhbi

\institution Polytechnique Montreal
 \city  \country

amine.mhedhbi@polymtl.ca

Abstract.

Recent advancements in Text-to-SQL, driven by large language models, are democratizing data access.
Despite these advancements, enterprise deployments remain challenging due to the need to capture business-specific knowledge,
handle complex queries, and meet expectations of continuous improvements.
To address these issues, we designed and implemented  GenEdit : our Text-to-SQL generation system that improves with user feedback.
 GenEdit  builds and maintains a company-specific knowledge set, employs a pipeline of operators decomposing SQL generation, and uses feedback to update its knowledge set to improve future SQL generations.

We describe  GenEdit ’s architecture made of two core modules:
(i) decomposed SQL generation; and
(ii) knowledge set edits based on user feedback.
For generation,  GenEdit  leverages compounding operators to improve knowledge retrieval and to create a plan as chain-of-thought steps that guides generation.
 GenEdit  first retrieves relevant examples in an initial retrieval stage where original SQL queries are decomposed into sub-statements, clauses or sub-queries. It then also retrieves instructions and schema elements.
Using the retrieved contextual information,
 GenEdit  then generates step-by-step plan in natural language on how to produce the query. Finally,  GenEdit  uses the plan to generate SQL, minimizing the need for model reasoning, which enhances complex SQL generation. If necessary,  GenEdit  regenerates the query based on syntactic and semantic errors.
The knowledge set edits are recommended through an interactive copilot, allowing users to iterate on their feedback and to regenerate SQL queries as needed. Each generation uses staged edits which update the generation prompt.
Once the feedback is submitted, it gets merged after passing regression testing and obtaining an approval, improving future generations.

1.  Introduction

Recent advancements in Text-to-SQL have broadened access to data for a wider range of users while enabling faster query authoring using database management systems (DBMSs).
At the core of these advancements are large language models (LLMs),
which achieve unprecedented accuracy through in-context learning (ICL)

DBLP:conf/nips/PourrezaR23  ;  DBLP:journals/corr/abs-2310-17342

and fine-tuning

DBLP:journals/pvldb/GaoWLSQDZ24  ;  10.1145/3654930  ;  DBLP:journals/corr/abs-2408-07702/death-schema

.
Nevertheless many of these approaches are not readily deployable in the enterprise setting.
From our experience with customer deployments, a Text-to-SQL solution ought to:
i) understand  external knowledge ,  e.g. , a company’s specific terminology and processes;
ii) handle  large query complexity  due to the schemas of data warehouses and the inherent complexity of the queries themselves; and
iii) improve  systematically over time .

Consider the following query, denoted as

Q

f  ​  i  ​  n

−

p  ​  e  ​  r  ​  f

Q_{fin-perf}

, which will serve as a running example.
This query originates from a data analyst working at a holding company with ownership in multiple sports organizations.  1

1  1 While

Q

f  ​  i  ​  n

−

p  ​  e  ​  r  ​  f

Q_{fin-perf}

represents the actual structure of a query in production, we mask the domain and any relevant customer or user details.

The query is defined as follows:

⬇

Identify

our

5

sports

organisations

with

the

best

and

worst

QoQFP

in

Canada

for

Q2

2023.

The query asks for QoQFP,  i.e. , quarter-over-quarter financial performance.
The QoQFP acronym within this company has a very specific meaning and associated calculations;
the query cannot be answered without understanding such specifics.
An LLM might in fact have a different interpretation of this acronym due to its pre-training.
Furthermore, the query’s equivalent SQL has very high complexity.
The equivalent SQL to

Q

f  ​  i  ​  n

−

p  ​  e  ​  r  ​  f

Q_{fin-perf}

is shown in Appendix §  A

with appropriate domain and data masking. Such highly complex queries have been reported as a core Text-to-SQL challenge

DBLP:journals/corr/abs-2406-08426/nextgen-db-interfaces-survey

and differ significantly to the common queries found in current popular public benchmarks

li2023llm  ;  yu2019spider

.
Finally, data analysts expect that even if the initial generated SQL fails,
the query generation should improve over time.

GenEdit  is our purpose-built system to address the challenges of enterprise scenarios similar to the prior example.
First,  GenEdit  captures the specific context of a company by building and maintaining a knowledge set.
The set is a  view  containing pairs of: i) natural language; and ii) SQL examples, natural language instructions (or hints) for generation, and database schemas.
Second,  GenEdit  handles the complexity challenge by relying on a multi-operator pipeline using LLM calls to decompose the problem and generates a step-by-step plan thereby reducing the need for LLM reasoning.
Finally,  GenEdit  improves over time as users provide free text feedback leading to editing suggestions to the knowledge set and prompts.
This in turn improves future generations.
 GenEdit  contains an edits recommendation module that supports subject matter experts in improving the system without understanding its internals.

Previous work on Text-to-SQL shares similarities with our approach. For example,  GenEdit  pipelines are decomposed into operators and rely on few-shot examples for generation

DBLP:conf/nips/PourrezaR23  ;  DBLP:conf/icde/RenFHHDHJZYW24  ;  wang2024macsql

. Our approach however deviates in important ways.  GenEdit  operators do not retrieve or produce separate context fed to the LLM for generation;
instead, they compound,  e.g. , the choice of relevant examples informs the choice of instructions to retrieve.
 GenEdit  also imposes a very different intermediate representation fed to the model to generate SQL.
Instead of using full Text-to-SQL pairs as examples,  GenEdit  decomposes examples into smaller clauses, which are used for the chain-of-thought (CoT) optimization

DBLP:conf/nips/Wei0SBIXCLZ22  ;  DBLP:journals/corr/abs-2310-17342

.
 GenEdit  first generates a CoT plan in which one or more steps describe a Common Table Expression (CTE) expected to be part of the output candidate query.
 GenEdit  then constructs its final output by combining these CTEs as described in the plan with an added SELECT-FROM-WHERE instruction.

In the rest of the paper, we overview  GenEdit ’s architecture (§  2  ). We primarily cover our insights designing  GenEdit , which are:

•

Compounding Operators  (§  3.1  ):

1  1

.

Context Expansion  (§  3.1.1  ): we decompose SQL generation into multiple operators where the output of the prior operator is used in the subsequent one.
For example, consider a list of operators retrieving relevant contextual information for the input query from the knowledge set.
Let an initial operator select relevant examples of the task,  i.e. , (natural language query, SQL) pairs.
The selection of these examples informs that of relevant instructions, which are natural language descriptions in the form of hints of what to do or definition of certain terms; the selection of both then informs the choice of relevant schema elements and so on.
Such context expansion is akin to  query expansion  and improves the performance of subsequent retrieval operators by adding more relevant context to the query and hence finding more relevant elements in the knowledge set.

2  2

.

Planning