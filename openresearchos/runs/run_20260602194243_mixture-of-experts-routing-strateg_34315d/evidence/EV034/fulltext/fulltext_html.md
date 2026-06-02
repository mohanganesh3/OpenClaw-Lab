[2110.10887] A Real-Time Energy and Cost Efficient Vehicle Route Assignment Neural Recommender System

A Real-Time Energy and Cost Efficient Vehicle Route Assignment Neural Recommender System

Ayman Moawad,
Zhijian Li,
Ines Pancorbo,
Krishna Murthy Gurumurthy,
Vincent Freyermuth,
Ehsan Islam,
Ram Vijayagopal,
Monique Stinson,
and Aymeric Rousseau

A. Moawad is with the Vehicle and Mobility Simulation group at Argonne National Laboratory, 9700 S. Cass Ave, Lemont, IL 60439 USA and the department of Statistics, The University of Chicago, 5801 S. Ellis Ave, Chicago, IL 60637 USA. E-mail: amoawad@anl.gov, aymoawad@uchicago.eduKrishna Murthy Gurumurthy, Vincent Freyermuth, Ehsan Islam, Ram Vijayagopal, Monique Stinson, and Aymeric Rousseau are with the Vehicle and Mobility Simulation group at Argonne National Laboratory.Zhijian Li is with the department of Mathematics at The University of California Irvine.Ines Pancorbo is with the department of Mathematics and Statistics at Georgetown University.

Abstract

This paper presents a neural network recommender system algorithm for assigning vehicles to routes based on energy and cost criteria. In this work, we applied this new approach to efficiently identify the most cost-effective medium and heavy duty truck (MDHDT) powertrain technology, from a total cost of ownership (TCO) perspective, for given trips. We employ a machine learning based approach to efficiently estimate the energy consumption of various candidate vehicles over given routes, defined as sequences of links (road segments), with little information known about internal dynamics, i.e using high level macroscopic route information.
A complete recommendation logic is then developed to allow for real-time optimum assignment for each route, subject to the operational constraints of the fleet. We show how this framework can be used to (1) efficiently provide a single trip recommendation with a top-

k

𝑘

k

vehicles star ranking system, and (2) engage in more general assignment problems where

n

𝑛

n

vehicles need to be deployed over

m  ≤  n

𝑚  𝑛

m\leq n

trips. This new assignment system has been deployed and integrated into the POLARIS  1

1  1 POLARIS is an Argonne-based high-performance, open-source agent-based modeling framework for simulating large-scale transportation systems.

Transportation System Simulation Tool for use in research conducted by the Department of Energy’s Systems and Modeling for Accelerated Research in Transportation (SMART) Mobility Consortium  [ 1 ] .

Index Terms:

Neural recommender systems, machine learning, MDHD, trucks, vehicle assignment, energy consumption, cost, TCO.

I

Introduction

Freight companies are facing increasing pressure to decarbonize their fleets. The large number of technology options, the diversity in vehicle usage and economic uncertainties are major hurdles slowing down new vehicle technology adoptions.
Transportation decarbonization across the freight industry is a major challenge for multiple reasons: (1) the large number of vehicle applications, powertrain and component technologies makes it difficult for fleets to decide which vehicles to invest in, and (2) the diverse vehicle usage, both current and future, raises questions as to which technology should be assigned to a particular route. Understanding the techno-economic impact of technologies is an active field of research.
Traditionally, energy consumption and economic impacts have been evaluated using standard drive cycles as a baseline for regulatory purposes  [ 2 ] ,  [ 3 ] ,  [ 4 ] ,  [ 5 ] ,  [ 6 ] . While they are a good standardization for energy benefit studies, regulatory cycles fail to represent real driving conditions, traffic variability and other effects such as future infrastructure and connectivity changes, population density changes, traffic behavior changes, etc.

As a result, the freight industry has been operating under a high level of uncertainty both for longer term technology adoptions and investment as well as day-to-day operations of their current vehicle fleet. Since the industry is heavily driven by cost, we choose to study the optimal assignment problem from a TCO standpoint, assuming that the lowest-cost powertrain option captures driving conditions and number of miles driven over the lifetime of the vehicle under that metric.

To represent diverse vehicle usage as well as the impact of multiple vehicle technologies on energy consumption and cost, both current and future, we leverage an agent-based transportation tools to model truck routes across the city of Chicago and its suburbs under various scenarios, combined a high-fidelity vehicle system simulation tool, to estimate individual truck energy consumption and cost. The results generated serve as a very large backbone dataset of vehicle-route energy outcomes that capture variability in vehicle classes, powertrain fleet distribution, vehicle technology, automation and connectivity levels, population, driving modes, ride-sharing extent, e-commerce impact, etc. All of these affect traffic and driving behavior: For example, connected and automated vehicle (CAV) technologies are likely to have significant effects not only on how vehicles operate in the transportation system but on how individuals behave and use their vehicles. In Section

III-A

we provide high-level details about the data generation process and a brief overview of the data content. For more details about the design of the experiment, the tools involved, and their capabilities, refer to  [ 7 ]  and  [ 8 ] .

The rest of this article describes the development of a machine learning based vehicle-route assignment recommender system for MDHDT to efficiently identify cost effective technologies for different routes, cargo requirements, goods, energy cost, etc. from a total cost of ownership (TCO) perspective (Figure

1  ) with very limited known internal road dynamics (i.e using high level route information). With deployment goals in mind, the system needs to be lightweight, computationally efficient, accurate, and scalable for later integration into subsequent tools, allowing real-time querying. The objective of the paper is to provide and end-to-end deployable tool to support fleet decision making related to investments and technology usage.

Figure 1:  Summary view of the recommender system with the required inputs, assumptions and constraints to output a top- k  ranking of recommended vehicles. For a given trip or set of trips and a flexible list of candidate vehicles, the system outputs a recommendation based on a fast energy consumption prediction that accounts for cost, time and other fleet constraints. Examples of such constraints are operational costs, charging/refueling time delays, delivery window requirements, and truck payload needs.

II

Recommender Systems

II-A

Background

Recommender systems—algorithms that predict users’ preferences among a large set of items based on their feedback—are widely used in many businesses today  [ 9 ] ,  [ 10 ] ,  [ 11 ] . A recommender system learns the systematic relationship between users and items based on past behavior as well as the item attributes involved. These systems are used to establish personalized systems that recommend items to the users likeliest to use or buy them. Table

I

shows a direct similarity to our problem: We want to recommend vehicles for routes. Typically, recommender systems are built to deal with a large matrix of user–item pairs, while our system operates similarly in a very large matrix of vehicle–route possibilities. This approach also works well in a sparse setting, in which there is a large number of users and a big catalogue of items, but not all users interact with all items. In our case, the vehicle–route design matrix is large, but not all vehicles use all routes. Our goal is to predict a rating, in our case TCO, to rate each vehicle–route pairs.

TABLE I:  Typical product recommendation compared to