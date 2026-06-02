[2107.00359] Model Mediated Teleoperation with a Hand-Arm Exoskeleton in Long Time Delays Using Reinforcement Learning

Model Mediated Teleoperation with a Hand-Arm Exoskeleton in Long Time Delays Using Reinforcement Learning

Hadi Beik-Mohammadi  1,2,∗  , Matthias Kerzel  2  , Benedikt Pleintinger  1  , Thomas Hulin  1  ,
 Philipp Reisich  1  , Annika Schmidt  1,3  , Aaron Pereira  1  , Stefan Wermter  2

and Neal Y. Lii  1

1

Institute of Robotics and Mechatronics, German Aerospace Center, Munich, Germany
 (Hadi.Beikmohammadi, Thomas.Hulin, Annika.Schmidt, Benedikt.Pleintinger, Philipp.Reisich, Aaron.Pereira, Neal.Lii)@DLR.de

2  Knowledge Technology Institute, Dept of Informatics, Hamburg University, Hamburg, Germany
 (Kerzel, Wermter) @ informatik.uni-hamnburg.de

3

Technical University of Munich, Munich, Germany
 (An.Schmidt)@TUM.de

Abstract

Telerobotic systems must adapt to new environmental conditions and deal with high uncertainty caused by long-time delays. As one of the best alternatives to human-level intelligence, Reinforcement Learning (RL) may offer a solution to cope with these issues.
This paper proposes to integrate RL with the Model Mediated Teleoperation (MMT) concept. The teleoperator interacts with a simulated virtual environment, which provides instant feedback. Whereas feedback from the real environment is delayed, feedback from the model is instantaneous, leading to high transparency.
The MMT is realized in combination with an intelligent system with two layers. The first layer utilizes Dynamic Movement Primitives (DMP) which accounts for certain changes in the avatar environment. And, the second layer addresses the problems caused by uncertainty in the model using RL methods.
Augmented reality was also provided to fuse the avatar device and virtual environment models for the teleoperator.
Implemented on DLR’s Exodex Adam hand-arm haptic exoskeleton, the results show RL methods are able to find different solutions when changes are applied to the object position after the demonstration. The results also show DMPs to be effective at adapting to new conditions where there is no uncertainty involved.

I

INTRODUCTION

Teleoperation provides the possibility for an operator to interact with a remote environment using an intermediate device. The intermediate device consists of two interconnected parts so-called input and avatar. Using the input device, the teleoperator remotely controls/commands the avatar through a communication channel. Although the avatar is assumed to be passive, due to the bilateral control scheme, it affects the input device using position or force feedback. The bilateral control scheme provides feedback to the operator, which augments the remote environment to be perceived by the operator’s senses, such as haptic, visual, and auditory.
In a long-distance teleoperation scenario, data transmission can take a significant time delay to reach the teleoperator. This delay introduces inconsistency or mismatch between input and avatar system that falsifies the provided feedback. For example, the teleoperator may receive the haptic feedback well after the avatar robot has already collided with an object in the remote environment, causing damage to the target robot system, as well as the remote environment. Furthermore, the visual information provided to the operator may also have significant discrepancies. For example, the object may move in the span of the communication delay, which can cause the task to be compromised. Therefore, relying solely on the data collected from the remote environment is undesirable.

Figure 1:  A teleoperator using the proposed approach with the Exodex Adam hand-arm haptic interface and a Microsoft Hololens teaches the avatar robot to build a tower using three cubes. The teleoperator observes a replica of the remote environment and the avatar robot using the augmented reality.

Grasping or manipulation of objects by teleoperation can be realized in different ways. The robot configurations can be pre-recorded and used as a look-up table. The corresponding configurations can be selected based on the closest object position to the current position  [ 1 ] . To reduce the inconsistency between configurations, end-to-end deep learning with pre-trained neural networks  [ 2 ]  can be used instead of hard-coded look-up tables. Furthermore, deep reinforcement learning can be used as an online solution where the agent learns to reach for grasp by interacting with the environment  [ 3 ,  4 ,  5 ] . The idea of learning from demonstration (LfD) can increase the stability and performance of the grasping and in-hand manipulation  [ 6 ,  7 ] .

In practice, a teleoperation scenario may take place on different levels of abstraction, automation, and shared autonomy. Using a task-driven approach with gesture recognition, it has been demonstrated that in-hand manipulation can be effectively carried out for all six possible Degrees of Freedom (DOF), on free and partially constrained objects  [ 8 ] . German Aerospace Center (DLR) and European Space Agency’s (ESA) METERON SUPVIS Justin  [ 9 ]  was a teleoperation mission with high-level abstraction and complete task autonomy where an astronaut in orbit commanded a robot on the ground to execute a task. In this experiment, the task performance relied mainly on the avatar’s built-in intelligence and the intuitiveness of the task representation for the teleoperator.

Teleoperators continue to demand immersive user experience with haptic feedback of force reflection to feel the object and the environment. Various work has been carried out to study force reflection in-hand telepresence, including an exoskeleton system that uses neural networks  [ 10 ] .

Furthermore, previous work  [ 11 ]  has evaluated and shown the effectiveness of haptic feedback for increasing the in-hand teleoperation performance, as compared to other feedback conditions. By considering the viscosity of the environment, a telepresence system can be extended to interact in a mixed media environment of fluid, gas, and solids up to the fingertips to realize an even more immersive user experience  [ 12 ] .

The bilateral teleoperation architecture usually contains 2-channel  [ 13 ]  or 4-channel  [ 14 ]  communication, which ensures the consistency between two models and the safety of the avatar.
For example, if during a task an object in the remote environment hinders the avatar movement, instant force feedback can inform the operator to prevent colliding with an object or stop penetrating a hard surface. Instant force feedback requires high bandwidth communication channels with no time delay.

As mentioned, time delay can cause teleoperation to become unstable, particularly for high-DOF, long-delay systems. Kontur-2  [ 15 ]  and METERON Haptics-2 experiments [ 16 ]  have tackled this with 20 msec to 800 msec+ of time delays. In robot-assisted remote telesurgery, the operations are handled with delay as long as 100 msec+  [ 17 ] .

Approaches such as passivity-based control  [ 18 ,  19 ,  20 ]  and predictive display  [ 21 ,  22 ]  can reduce the inconsistency, but the problems appear when the time delay increases or becomes variable. A unique way of approaching the time delay problem is introduced by Model Mediated Teleoperation (MMT) approach  [ 23 ,  24 ] . The MMT approach uses a virtual replica of the remote environment on the input-side to provide instant feedback for the teleoperator and has been used in several areas, for example, surgical teleoperation  [ 25 ]  and space robotics  [ 26 ] .

MMT requires an accurate model of the remote environment and avatar device, but modeling a non-linear time-variant system like a remote environment is problematic.
The MMT approaches attempt to recreate the remote environment model using different methods such as neural networks  [ 27 ] .
To tackle problems caused by the time delay, proving a form of timely visual and haptic feedback combined wit