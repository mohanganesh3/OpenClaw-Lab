# EV019: Constructing and predicting school advice for academic achievement: a comparison of item response theory and machine learning techniques

URL: https://www.semanticscholar.org/paper/00475e45e3a5a5e0fbb03d6fafd9ee4e728861b6
Year: 2020
Source: semantic_scholar
Arxiv: n/a

## Abstract

Educational tests can be used to estimate pupils' abilities and thereby give an indication of whether their school type is suitable for them. However, tests in education are usually conducted for each content area separately which makes it difficult to combine these results into one single school advice. To help with school advice, we provide a comparison between both domain-specific and domain-agnostic methods for predicting school types. Both use data from a pupil monitoring system in the Netherlands, a system that keeps track of pupils' educational progress over several years by a series of tests measuring multiple skills. A domain-specific item response theory (IRT) model is calibrated from which an ability score is extracted and is subsequently plugged into a multinomial log-linear regression model. Second, we train domain-agnostic machine learning (ML) models. These are a random forest (RF) and a shallow neural network (NN). Furthermore, we apply case weighting to give extra attention to pupils who switched between school types. When considering the performance of all pupils, RFs provided the most accurate predictions followed by NNs and IRT respectively. When only looking at the performance of pupils who switched school type, IRT performed best followed by NNs and RFs. Case weighting proved to provide a major improvement for this group. Lastly, IRT was found to be much easier to explain in comparison to the other models. Thus, while ML provided more accurate results, this comes at the cost of a lower explainability in comparison to IRT.
