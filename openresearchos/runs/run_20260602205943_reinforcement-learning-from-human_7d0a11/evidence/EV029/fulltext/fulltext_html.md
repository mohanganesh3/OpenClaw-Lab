[2411.08734] Recommender systems and reinforcement learning for building control and occupant interaction: A text-mining driven review of scientific literature

Recommender systems and reinforcement learning for building control and occupant interaction: A text-mining driven review of scientific literature

Wenhao Zhang

National University of Singapore, Department of the Built Environment

Matias Quintana

Singapore-ETH Centre, Future Cities Lab Global Programme

Clayton Miller

Abstract

The indoor environment significantly impacts human health and well-being; enhancing health and reducing energy consumption in these settings is a central research focus. With the advancement of Information and Communication Technology (ICT), recommendation systems and reinforcement learning have emerged as promising approaches to induce behavioral changes to improve the indoor environment and energy efficiency of buildings. This study aims to employ text-mining and Natural Language Processing (NLP) techniques to thoroughly examine the connections among these approaches in the context of building control and occupant interaction. The study analyzed around 27,000 articles from the ScienceDirect database, revealing extensive use of recommendation systems and reinforcement learning for space optimization, location recommendations, and personalized control suggestions. Although these systems are broadly applied to specific content, their use in optimizing indoor environments and energy efficiency remains limited. This gap likely arises from the need for interdisciplinary knowledge and extensive sensor data. Traditional recommendation algorithms, including collaborative filtering, content-based and knowledge-based methods, are commonly employed. However, the more complex challenges of optimizing indoor conditions and energy efficiency often depend on sophisticated machine learning (ML) techniques like reinforcement and deep learning. Furthermore, this review underscores the vast potential for expanding recommender systems and reinforcement learning applications in buildings and indoor environments. Fields ripe for innovation include predictive maintenance, building-related product recommendation, and optimization of environments tailored for specific needs, such as sleep and productivity enhancements based on user feedback. The study also notes the limitations of the method in capturing subtle academic nuances. Future improvements could involve integrating and fine-tuning pre-trained language models to better interpret complex texts.

keywords:

Reinforcement Learning , Recommender Systems , Building Operations , Building Energy Efficiency , Word Embeddings , Natural Language Processing

†

†  journal:  Energy and Buildings

Highlights

Recommender systems and reinforcement learning for building control and occupant interaction: A text-mining driven review of scientific literature

Wenhao Zhang, Matias Quintana, Clayton Miller

•

27,000 full-text articles related to recommendation systems and the built environment have been extracted for text mining.

•

Detailed relationships among recommender algorithms, system types, input data, interventions, and platforms have been elucidated.

•

Word embeddings model has been used to extract the relationship between keywords.

•

The most commonly used algorithms, well-established applications, and emerging opportunities in building control and occupant interaction have been discussed.

1  Introduction

1.1  Background

The indoor environment significantly affects human health and well-being, with individuals spending on average approximately 86% of their day indoors  [ 1 ,  2 ] . Addressing how to improve health and well-being in indoor environments while reducing energy consumption is a central question in contemporary research  [ 3 ] . In recent years, smart building control has become an important topic in this regard, with extensive studies demonstrating its ability to significantly reduce energy usage while maintaining indoor comfort levels  [ 4 ,  5 ,  6 ,  7 ] . However, despite these promising results, the adoption rate of smart control technologies remains relatively low  [ 8 ] . The primary barriers to widespread adoption are the high initial costs and complexity of implementation  [ 9 ] . Therefore, there remains a need to explore a more feasible, user-friendly, and cost-effective solution.

In the context of the widespread adoption of Artificial Intelligence (AI), the Internet of Things (IoT) devices, and smart mobile devices, a viable approach is the behavior change facilitated by Information and Communication Technology (ICT)  [ 10 ,  11 ] . This method leverages recommendation systems and smart devices to influence and change human behavior, thus improving indoor environmental quality and reducing energy consumption  [ 10 ] . Central to this strategy is sophisticated recommendation algorithms that analyze user preferences and environmental data to provide personalized suggestions. These algorithms have shown success in various sectors, including mobile health, commonly referred to as Just-In-Time Adaptive Interventions (JITAI) in the domain  [ 12 ] , as well as in online shopping  [ 13 ] , entertainment  [ 14 ,  15 ] , and social networking  [ 16 ] , etc.

In the field of building and built environments, several conventional literature reviews have been conducted recently. These reviews analyze current studies on recommendation systems in various areas, including building performance optimization  [ 17 ,  18 ,  19 ] , IoT  [ 20 ,  21 ,  22 ] , and smart cities  [ 23 ,  24 ,  25 ,  26 ] . These systems are typically utilized to optimize indoor environments  [ 27 ,  28 ]  and enhance building energy efficiency  [ 29 ,  30 ,  31 ,  32 ] . Predominantly, the methods involve leveraging real-time data and federated learning to train models that generate energy-saving suggestions  [ 33 ,  34 ,  35 ] , alongside integrating sensor data with user preference feedback to provide personalized recommendations  [ 36 ,  37 ] . Furthermore, reinforcement learning-based recommender systems are widely utilized in this domain, due to their ability to handle multi-objective optimization tasks  [ 38 ,  39 ] . For example, multi-agent deep reinforcement learning has been developed to optimize energy consumption, occupant comfort, and air quality simultaneously in commercial buildings  [ 40 ] . These systems are typically reactive, meaning they respond to user inputs or environmental sensors to provide suggestions. However, proactive recommendation systems have also been explored, which aim to incrementally alter user habits during specific moments based on the theories of  micro-moments  and  habit loops

[ 41 ] . Additionally, due to the opacity of machine learning (ML)  black box  models, explainable AI has been used to provide explainable and personalized energy efficiency suggestions  [ 42 ] . On an urban scale, the application of recommendation systems has also garnered interest, primarily to improve energy efficiency in smart grids  [ 43 ,  44 ]  and facilitate demand response initiatives  [ 45 ] .

These reviews cover the diverse applications of recommendation systems in various aspects of the built environment. However, due to the limitations of human analytical capacity, conventional literature reviews may not adequately handle the vast volume of publications, struggling to discern the subtle relationships among different studies, particularly in interdisciplinary fields  [ 46 ,  47 ] . Typically, these reviews are restricted to analyzing between 80-100 publications. Given the complexity and interdisciplinarity of integrating recommendation systems within the built environment, a more sophisticated approach is essential to efficiently manage and interpret extensive literature datasets. To address these challenges, text mining-based literature review methods emerge as a key tool.

1.2  Text mining-based literature review

T