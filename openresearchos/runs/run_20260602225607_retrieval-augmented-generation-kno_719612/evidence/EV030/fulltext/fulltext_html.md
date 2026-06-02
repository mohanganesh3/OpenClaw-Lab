[2509.11708] From Evaluation to Enhancement: Large Language Models for Zero-Knowledge Proof Code Generation

From Evaluation to Enhancement: Large Language Models for Zero-Knowledge Proof Code Generation

Zhantong Xue

0009-0000-3419-9431

Hong Kong University of Science and Technology  Hong Kong  China

zxueai@cse.ust.hk

,

Pingchuan Ma

0000-0001-7680-2817

Hong Kong University of Science and Technology  Hong Kong  China

CipherInsight Limited  Hong Kong  China

pmaab@cse.ust.hk

,

Zhaoyu Wang

0009-0009-6892-1264

Hong Kong University of Science and Technology  Hong Kong  China

zwangjz@cse.ust.hk

and

Shuai Wang

0000-0002-0866-0308

Hong Kong University of Science and Technology  Hong Kong  China

CipherInsight Limited  Hong Kong  China

shuaiw@cse.ust.hk

(11 September 2025)

Abstract.

Zero-knowledge proofs (ZKPs) are increasingly deployed in domains such as
privacy-preserving authentication, blockchain scalability, and secure finance.
However, authoring ZK programs remains challenging: unlike mainstream
programming, ZK development requires reasoning about finite field arithmetic,
constraint systems, and gadgets, making it knowledge-intensive and error-prone.
While large language models (LLMs) have demonstrated strong code generation
capabilities in general-purpose languages, their effectiveness for ZK
programming, where correctness hinges on both language mastery and gadget-level
reasoning, remains unexplored. To address this gap, we propose  ZK-Eval ,
a domain-specific evaluation pipeline that probes LLM capabilities at three
levels: language knowledge, gadget competence, and end-to-end program
generation. Our evaluation of four state-of-the-art LLMs reveals that models
excel at surface-level syntax but struggle with gadget usage and semantic
correctness, often yielding incorrect programs. Based on these insights, we
introduce  ZK-Coder , an agentic framework that augments LLMs with
constraint sketching, guided retrieval, and interactive repair. Experiments on
Circom and Noir show substantial gains, with success rates improving from
17.35% to 83.38% and from 32.21% to 90.05%, respectively. With
 ZK-Eval  and  ZK-Coder , we establish a foundation for
systematically measuring and augmenting LLMs in ZK code generation to lower
barriers for practitioners and advance trustworthy computation.

Zero-Knowledge Proofs, Large Language Models, Code Generation

†

†  copyright:  acmlicensed

†

†  journalyear:  2025

†

†  doi:  XXXXXXX.XXXXXXX

†

†  journalvolume:  X

†

†  journalnumber:  X

†

†  article:  X

†

†  journalyear:  2025

†

†  publicationmonth:  9

†

†  copyright:  none

†

†  ccs:  Software and its engineering Automatic programming

†

†  ccs:  Security and privacy Cryptography

†

†  ccs:  Software and its engineering Domain specific languages

†

†  ccs:  Computing methodologies Natural language processing

1.  Introduction

Zero-knowledge proofs (ZKPs)  are a powerful cryptographic primitive that
allow one party to prove knowledge of a statement without revealing the
underlying witness. In practice, this means proving that a computation or claim
is correct while keeping sensitive inputs private. ZKPs have moved far beyond
theory: they are now central to privacy-preserving authentication, blockchain
scalability, and emerging applications across finance and security
domains  (Gupta,  2025 ; Lavin et al.,  2024 ; team,  2024 ) .
As demand for trustworthy computation grows, the ability to author, test, and
verify ZKP programs has become a first-class concern in the software engineering
community  (Takahashi et al.,  2025 ; Hochrainer et al.,  2024 ; Xu et al.,  2025 ; Pailoor et al.,  2023 ; Jiang et al.,  2025a ) .

Figure 1 .

Illustrative Comparison Between the Process of Mainstream and ZK Programming.

However, authoring ZK programs is challenging. Unlike mainstream programming,
which builds on the von Neumann model and imperative execution, ZKP programs
boil down to specifying and proving mathematical  constraints  over finite
fields. Its development requires reasoning about finite field arithmetic,
constraint systems, and  gadgets  — a novel concept advocated by modern
ZKP frameworks to encode reusable, domain-specific building blocks. As shown in
Fig.

1

and
Table

1  , ZK programs do not describe how to
 execute  a computation but how to  prove  its correctness by
compiling constraints into arithmetic circuits. Developers must specify
equations over inputs, outputs, and intermediate variables, carefully wiring
gadgets to enforce correctness. Even small omissions or miswirings can yield
satisfiable but unsound constraint systems, leading to errors that compilers
rarely flag. This makes ZK development knowledge-intensive, error-prone, and
inaccessible to non-experts, despite significant progress in proof systems and
verification tools  (Chaliasos et al.,  2024 ; Pailoor et al.,  2023 ; Takahashi et al.,  2025 ) .

Table 1 .

Comparison of Mainstream and ZK Programming.

Aspect

Mainstream

ZK Program

Paradigm

Imperative/OOP/Func

Constraint Systems

Comp. Model

von Neumann Arch

Arithmetic Circuits

Purpose

to Compute

to Prove

Data Types

Int/Float/Str/…

Finite Field Elements

Mutability

Dynamic

Static

Turing-complete?

Yes

No

Control Flow

If/Loop/Recursion

Limited

Side Effects

Yes

No, Pure Math

Figure 2 .

Overview of  ZK-Eval  and  ZK-Coder .

As a result, ZK programming demands significantly more time and expertise than
mainstream software development, posing a steep barrier to entry for many
practitioners. In contrast, recent advances in large language models (LLMs) have
lowered this barrier in general-purpose languages by demonstrating strong code
generation capabilities  (Jiang et al.,  2025b ) , achieving impressive results on
benchmarks such as HumanEval  (Chen et al.,  2021 )  and
MBPP  (Research,  2021 ) , and powering widely adopted tools like GitHub
Copilot  (GitHub,  2025 )  and Cursor  (team,  2025 ) . This contrast naturally raises
the question:  to what extent can modern LLMs assist in generating ZK code
from natural language specifications?  Answering this question requires more
than surface-level benchmarking. Compared to programming in mainstream
languages, success in ZK development hinges on two competencies: (i) knowledge
of domain-specific languages (DSLs) and their toolchains, and (ii) mastery of
gadgets as building blocks for constraint systems. Existing benchmarks focus
only on mainstream languages and overlook these unique requirements.

To address this gap, we introduce  ZK-Eval , a domain-specific evaluation
pipeline that probes LLM capability at three levels: language knowledge,
gadget-level competence, and end-to-end program code generation (left panel of
Fig.

2  ). We anticipate that this design would provide
a structured way to pinpoint where models succeed, where they fail, and why,
rather than treating code generation as a monolithic task.
Applying  ZK-Eval  to four state-of-the-art LLMs (including advanced
reasoning models like GPT-o4-mini and GPT-o3) reveals a clear pattern. While
models perform strongly on surface-level language knowledge, their accuracy
drops significantly when reasoning about gadgets or assembling complete
programs, often producing code that compiles but fails semantically. These
results underscore both the promise of LLMs in lowering barriers to ZK
development and the need for augmentations to bridge the gap between syntactic
knowledge and reliable code generation.
Motivated by these findings, we present  ZK-Coder  (right panel of
Fig.

2  ), an agentic framework that augments LLMs with
constraint sketching, retrieval, and interactive repair. By grounding generation
in constraint reasoning and guided gadget usage,  ZK-Coder  achieves
substantial gains in reliability, improving success rates from 17.35% to
83.38% on Circom and from 32.21% to 90.05% on Noir. In summary, our
contributions are as follows:

(1)

We propose  ZK-Eval , a three-stage eva