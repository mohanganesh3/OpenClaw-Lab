# EV020: Learning to Allocate Time-Bound and Dynamic Tasks to Multiple Robots Using Covariant Attention Neural Networks

URL: https://www.semanticscholar.org/paper/0047fe61294019af30f3476ceb40f4637aabd575
Year: 2024
Source: semantic_scholar
Arxiv: n/a

## Abstract


 In various applications of multi-robotics in disaster response, warehouse management, and manufacturing, tasks that are known apriori and tasks added during runtime need to be assigned efficiently and without conflicts to robots in the team. This multi-robot task allocation (MRTA) process presents itself as a combinatorial optimization (CO) problem that is usually challenging to be solved in meaningful timescales using typical (mixed)integer (non)linear programming tools. Building on a growing body of work in using graph reinforcement learning to learn search heuristics for such complex CO problems, this paper presents a new graph neural network architecture called the Covariant Attention Mechanism (CAM). CAM can not only generalize but also scale to larger problems than that encountered in training, and handle dynamic tasks. This architecture combines the concept of Covariant Compositional Networks used here to embed the local structures in task graphs, with a context module that encodes the robots' states. The encoded information is passed onto a decoder designed using Multi-head Attention mechanism. When applied to a class of MRTA problems with time deadlines, robot ferry range constraints, and multi-trip settings, CAM surpasses a state-of-art graph learning approach based on the attention mechanism, as well as a feasible random-walk baseline across various generalizability and scalability tests. Performance of CAM is also found to be at par with a high-performing non-learning baseline called BiG-MRTA, while noting up to a 70-fold improvement in decision-making efficiency over this baseline.
