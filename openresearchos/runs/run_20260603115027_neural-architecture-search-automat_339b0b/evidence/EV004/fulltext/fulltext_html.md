[2211.04343] Abstract

\DeclareAcronym

est
short = EST,
long = expressed sequence tags

\DeclareAcronym Xl
short =  X. laevis ,
long =  Xenopus laevis

\DeclareAcronym Xg
short =  X. gilli ,
long =  Xenopus gilli

\thesistitle Quantum Deep Dreaming: A Novel Approach for Quantum Circuit Design  \supervisor Dr. Alán  Aspuru-Guzik

\examiner  Undergraduate Physics Thesis

\addresses

\subject Physics

\university  McMaster University

\department  Department of Physics and Astronomy

\group  The Matter Lab

\faculty  Arts and Science Program

Quantum Deep Dreaming: A Novel Approach for Quantum Circuit Design

\ttitle

By  \authorname ,

A Thesis Submitted to the School of Undergraduate Studies in the Partial Fulfillment of the course PHYSICS 4P06:  \degreename

\univname

©  Copyright by  \authorname

April 19, 2022

\univname

\degreename

(2024)

Hamilton, Ontario ( \deptname )

TITLE:  \ttitle

AUTHOR:  \authorname

( \univname )

SUPERVISOR:  \supname

NUMBER OF PAGES:

Abstract  ,  LABEL:LastPage

Abstract

\addchaptertocentry

Abstract
One of the challenges currently facing the quantum computing community is the design of quantum circuits which can efficiently run on near-term quantum computers, known as the quantum compiling problem. Algorithms such as the Variational Quantum Eigensolver (VQE), Quantum Approximate Optimization Algorithm (QAOA), and Quantum Architecture Search (QAS) have been shown to generate or find optimal near-term quantum circuits. However, these methods are computationally expensive and yield little insight into the circuit design process. In this paper, we propose Quantum Deep Dreaming (QDD), an algorithm that generates optimal quantum circuit architectures for specified objectives, such as ground state preparation, while providing insight into the circuit design process. In QDD, we first train a neural network to predict some property of a quantum circuit (such as VQE energy). Then, we employ the Deep Dreaming technique on the trained network to iteratively update an initial circuit to achieve a target property value (such as ground state VQE energy). Importantly, this iterative updating allows us to analyze the intermediate circuits of the dreaming process and gain insights into the circuit features that the network is modifying during dreaming. We demonstrate that QDD successfully generates, or ‘dreams’, circuits of six qubits close to ground state energy (Transverse Field Ising Model VQE energy) and that dreaming analysis yields circuit design insights. QDD is designed to optimize circuits with any target property and can be applied to circuit design problems both within and outside of quantum chemistry. Hence, QDD lays the foundation for the future discovery of optimized quantum circuits and for increased interpretability of automated quantum algorithm design.

Acknowledgements.

\addchaptertocentry  \acknowledgementname  To my advisor, Prof. Alán Aspuru-Guzik, for his continuing support and the opportunity to take part in the inextricably amazing field of quantum computing. It is an honour and privilege to be able to take part in the research at the University of Toronto Matter Lab. In addition, this research would not have been possible without the support and collaboration of Dr. Chong Sun, Dr. Jakob Kottmann, Dr. Thi Ha Kyaw, Dr. Sukin Sim, and Abhinav Anand.
To my mother, who encourages me to pursue the best version of myself. To my father who shows me how to be kind to other people. To my brother who shows me how to enjoy life.

{declaration}  \addchaptertocentry  \authorshipname

I,  \authorname , declare that this thesis titled,  “ \ttitle ”  and the work presented in it are my own. I confirm that:

•

I wrote every chapter and section of this thesis

•

I conducted the training and dreaming of the Quantum Deep Dreaming models described in this report

•

I conducted all of the experiments in Chapters 3 and 4, as well as gathered, cleaned, reported, and analyzed the results in Chapters 3 and 4

•

I developed the analytic techniques for evaluating the performance of Quantum Deep Dreaming and gaining insights from the Quantum Deep Dreaming process, described and used in Chapters 2.3, 3, and 4

•

I created all figures, images, and tables in this document

Conversion to HTML had a Fatal error and exited abruptly. This document may be truncated or damaged.

◄

Feeling lucky?

Conversion report

Report an issue

View&nbsp;original on&nbsp;arXiv  ►

Copyright

Privacy Policy

Generated  on Thu Mar 14 06:34:14 2024 by

L a T e

XML