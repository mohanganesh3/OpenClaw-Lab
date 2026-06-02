[2404.10876] Course Recommender Systems Need to Consider the Job Market

Course Recommender Systems Need to Consider the Job Market

Jibril Frej

jibril.frej@epfl.ch

0009-0009-0631-0636

ML4ED Lab, IC, EPFL  Switzerland

,

Anna Dai

anna.dai@epfl.ch

0009-0003-7250-1234

NLP Lab, IC, EPFL  Switzerland

,

Syrielle Montariol

syrielle.montariol@epfl.ch

0000-0003-1355-8778

NLP Lab, IC, EPFL  Switzerland

,

Antoine Bosselut

antoine.bosselut@epfl.ch

0000-0001-8968-9649

NLP Lab, IC, EPFL  Switzerland

and

Tanja Käser

Tanja.kaser@epfl.ch

0000-0003-0672-0415

ML4ED Lab, IC, EPFL  Switzerland

(2024; 20 February 2007; 12 March 2009; 5 June 2009)

Abstract.

Current course recommender systems primarily leverage learner-course interactions, course content, learner preferences, and supplementary course details like instructor, institution, ratings, and reviews, to make their recommendation.
However, these systems often overlook a critical aspect: the evolving skill demand of the job market.
This paper focuses on the perspective of academic researchers, working in collaboration with the industry, aiming to develop a course recommender system that incorporates job market skill demands.
In light of the job market’s rapid changes and the current state of research in course recommender systems, we outline essential properties for course recommender systems to address these demands effectively, including explainable, sequential, unsupervised, and aligned with the job market and user’s goals.
Our discussion extends to the challenges and research questions this objective entails, including unsupervised skill extraction from job listings, course descriptions, and resumes, as well as predicting recommendations that align with learner objectives and the job market and designing metrics to evaluate this alignment.
Furthermore, we introduce an initial system that addresses some existing limitations of course recommender systems using large Language Models (LLMs) for skill extraction and Reinforcement Learning (RL) for alignment with the job market.
We provide empirical results using open-source data to demonstrate its effectiveness.

Recommender System, Course Recommendation, Entity linking

†

†  journalyear:  2024

†

†  copyright:  acmlicensed

†

†  conference:  Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval; July 14–18, 2024; Washington, DC, USA

†

†  booktitle:  Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’24), July 14–18, 2024, Washington, DC, USA

†

†  doi:  10.1145/3626772.3657847

†

†  isbn:  979-8-4007-0431-4/24/07

†

†  ccs:  Computing methodologies Information extraction

†

†  ccs:  Information systems Recommender systems

†

†  ccs:  Applied computing Learning management systems

1.  Introduction

The contemporary job market is dynamic and rapidly evolving  (Deming and Noray,  2020 ) , necessitating continuous adaptation of individual skill sets to maintain relevance and competitiveness.
This evolution introduces a unique challenge: guiding learners in selecting educational courses that enhance their expertise and align with their career objectives and with job market demands.
However, a notable mismatch exists between the skills learners possess, the skills taught, and those in demand in the job market  (Palmer,  2017 ) . This mismatch can be explained by various factors such as the lag between demand in the market and the adaptation from course providers, unequal access to training, or the lack of information from training providers on the type of skills needed in the job market  (Palmer,  2017 ) .
This issue significantly limits the employability and career progression of individuals, impacting both their personal development and, at a more systemic level, economies dependent on a skilled workforce.

However, existing course recommender systems often focus solely on learner-course dynamics  (Sanguino Perez et al . ,  2022 ; Sakboonyarat and Tantatsanawong,  2019 ) , neglecting the crucial aspect of aligning course recommendations with real-time job market trends.
This limitation leads to a mismatch between the skills acquired through recommended courses and those in demand in the job market.
Moreover, while some approaches to career path recommendation or skill recommendation consider the job market, users’ skills, and goals for their recommendations  (Sun et al . ,  2021 ; Ghosh et al . ,  2020 ) , they do not recommend specific courses to help users achieve their goals.
Finally, to our knowledge, a single study proposes to use job postings and resumes in a course recommender system  (Bothmer and Schlippe,  2023 ) . However, this approach does not directly consider market trends or users’ goals, making this domain largely unexplored.

In this paper, we present the perspective of academic researchers working in collaboration with industry practitioners aiming to develop and deploy job-market-oriented course recommender systems. We argue that rethinking course recommender systems to consider the job market has the potential for significant economic and societal impact. We outline the properties these systems must have: ( P1 ) aligned with the latest job market trends to prioritize courses teaching high-demand skills; ( P2 ) unsupervised to avoid the resource-intensive process of collecting and annotating up-to-date data; ( P3 ) sequential to recommend a sequence of courses where each course builds upon the knowledge acquired in the preceding ones; ( P4 ) aligned with users’ goals such as attaining a specific role or increase their marketability; ( P5 ) explainable to ensure user trust and engagement.
We also highlight research directions and areas for future research to address the challenges in this field: ( RD1 ) Addressing the scarcity of course recommendation datasets by creating or providing datasets to the community for training and evaluation; ( RD2 ) Designing evaluation metrics to take into account alignment with the job market when evaluating the recommendations. ( RD3 ) Estimating user’s goal progress based on their profile and the job market to tailor the recommendations to their needs; ( RD4 ) Developing Skill-based explainable models and visualization techniques; ( RD5 ) Developing Unsupervised Skill Matching models to estimate up-to-date skill demand on the job market; ( RD6 ) Unsupervised Taxonomy Construction to include new emerging skills without human labeling.

In this work, we also develop a skill extraction and matching ( SEM ) method to identify skills and proficiency levels from learners’ resumes, course content, and job descriptions. We also develop an unsupervised, sequential skill-based Job-Market-Oriented Course Recommender system ( JCRec ) that uses the skills extracted by  SEM  to determine a candidate’s course options and to estimate the number of job opportunities available to them.  JCRec  then uses Reinforcement Learning (RL) to recommend a sequence of courses that maximizes the number of job opportunities available to the user. Our system meets all of the properties we previously outlined and lays out further steps for the research directions we’ve mentioned.

The key contributions of our paper include  1

1  1 Our code is available at  https://github.com/Jibril-Frej/JCRec

:

•

Identification of the desirable properties a job-market-oriented course recommender system should have.

•

Identification of the challenges that developing such systems will pose along with research direction for the community to address these challenges.

•

A few-shot skill extraction method to find skills from resumes, job postings, and course descriptions.

•

A formulation of job-market-oriented course recommendation as a Markov Decision Process

•

A first job-market-oriented course recommender system.

2.  Perspective

This paper presents the perspective of academic researchers worki