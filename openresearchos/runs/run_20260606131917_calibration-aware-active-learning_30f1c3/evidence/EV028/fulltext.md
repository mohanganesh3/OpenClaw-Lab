[2412.07959] Deep-Learning Control of Lower-Limb Exoskeletons via simplified Therapist Input

Deep-Learning Control of Lower-Limb Exoskeletons via simplified Therapist Input

Lorenzo Vianello  1  , Clément Lhoste  2  , Emek Barış Küçüktabak  7  , Matthew Short  1,4  ,

Levi Hargrove  5,6  , Jose L. Pons  1,3,4,6

1  Legs and Walking Lab of Shirley Ryan AbilityLab, Chicago, IL, USA 2  Rehabilitation Engineering Laboratory, ETH Zürich, CH 3  Center for Robotics and Biosystems of Northwestern University, Evanston, IL, USA 4  Department of Biomedical Engineering, Northwestern University, Evanston, IL, USA 5  Neural Engineering for Prosthetics and Orthotics Lab of Shirley Ryan AbilityLab, Chicago, IL, USA 6  Department of Physical Medicine and Rehabilitation, Northwestern University, Chicago, IL, USA 7  Honda Research Institute, San Jose, CA (USA)This work was conducted when E.B. Küçüktabak and C. Lhoste were with the Northwestern University and Shirley Ryan AbilityLab.

Abstract

Partial-assistance exoskeletons hold significant potential for gait rehabilitation by promoting active participation during (re)learning of “normal” walking patterns. Typically, the control of interaction torques in partial-assistance exoskeletons relies on a hierarchical control structure. These approaches require extensive calibration due to the complexity of the controller and user-specific parameter tuning, especially for activities like stair or ramp navigation.
To address the limitations of hierarchical control in exoskeletons, this work proposes a three-step, data-driven approach: (1) using recent sensor data to probabilistically infer locomotion states (landing step length, landing step height, walking velocity, step clearance, gait phase), (2) allowing therapists to modify these features via a user interface, and (3) using the adjusted locomotion features to predict the desired joint posture and model stiffness in a spring-damper system based on prediction uncertainty.
We evaluated the proposed approach with two healthy participants engaging in treadmill walking and stair ascent and descent at varying speeds, with and without external modification of the gait features through a user interface.
Results showed a variation in kinematics according to the gait characteristics and a negative interaction power suggesting exoskeleton assistance across the different conditions.

I

Introduction

Lower-limb exoskeletons are valuable tools in rehabilitation for individuals with gait impairments. Clinically, these exoskeletons can be classified into two categories: full-assistance and partial-assistance

[ 1 ] . Full-assistance exoskeletons are intended for patients with significant motor impairments who cannot ambulate independently, providing fully autonomous leg movement without requiring user input. They are usually controlled using position control strategies  [ 2 ] . In contrast, partial-assistance exoskeletons are more appropriate for patients with mild to moderate impairments, as they support movement while still requiring the patient’s active participation  [ 3 ] .

This paper focuses on partial-assistance exoskeletons, which have greater potential in rehabilitation settings as they consider the patient’s volitional movements in the control loop. These exoskeletons use approaches like haptic guidance  [ 4 ]  or resistance through error augmentation  [ 5 ]  to promote the (re)learning of walking patterns, depending on the patient’s functional ability.

Both of these strategies, assistance and resistance, depend significantly on the control of interaction torques between the user and the exoskeleton. This is typically achieved through a hierarchical structure comprising high-, mid-, and low-level controls  [ 6 ] . High-level control determines desired interaction torques for specific activities, such as walking on flat surfaces, stairs, or ramps. Mid-level control estimates various states within an activity, like swing and stance phases during walking, and adjusts the desired interaction torque accordingly  [ 7 ] . Finally, low-level control compensates for the exoskeleton’s dynamics and generates motor commands based on the profiles selected in high- and mid-level control.

To provide patients partial-assistance during ambulatory activities, desired interaction torques are typically rendered using virtual springs and dampers, characterized by stiffness and damping parameters, which are attached to reference joint positions. This necessitates the identification of many control parameters specific to each activity and phase of locomotion. As many of these parameters are also user-dependent, this can result in lengthy calibration periods for a single session of therapy. This issue has been well-documented for active prosthetic devices  [ 8 ] , but few studies have quantified the time spent calibrating lower-limb exoskeleton controllers  [ 9 ] . This is likely due to a reliance on predefined trajectories for generating exoskeleton assistance profiles, which reduces the time and complexity of parameter selection, but limits the degree of parameter customization for individual users. Moreover, most research has concentrated on overground walking, with minimal investigation into scenarios involving stairs and ramps  [ 10 ] .

Figure 1:

(1) Data-driven controller:(1.A) Sensor readings are passed to multiple deep-learning models to estimate representative features of the walking pattern. Thes resulting features are fitted to a normal distribution to consider the uncertanty of the locomotion pattern; (1.B) User interface allows to modify the self-selected locomotion features to allow the therapist to adapt the locomotion depending on the needs of each patient; (1.C) The resulting modified features are passed to a combination of models to regress the desired joint configuration of the exoskeleton. At the same time, the uncertainty of the prediction is propagated and used to model the stiffness of the impedance controller of the exoskeleton. Impedance parameters are passed to the low-level controller of the exoskeleton as commanded input. (2) Structure of the feature extractors models. (3) Structure of the command predictor models.

One potential solution to reduce controller complexity and calibration time is to employ data-driven methods that relate sensor readings with command outputs  [ 11 ,  12 ,  13 ,  14 ] . However, these approaches offer limited flexibility for personalization and adjustment of desired behaviors. For example, a therapist may require a progressive increase in foot clearance for a patient during a session of training. Additionally, therapists do not typically conceptualize adjustments in terms of gait cycle kinematics (e.g., trajectories or stiffness/damping parameters). Instead, they focus on clinically relevant features that are easier to interpret, such as step length, time, speed, or height.

For this reason, in this work, we propose a three-step data-driven approach to substitute the hierarchical structure of an exoskeleton controller (Fig.

1  ).
In step (1) we probabilistically regress a short-term history of sensor readings (encoders, force sensors, foot-plate sensors) with a minimal set of clinically relevant gait features, namely step length, and height (to distinguish across ascending/descending surfaces), walking velocity, step clearance, gait phase; in step (2) we allow an operator (e.g., a physical therapist) to modify the self-selected gait features using a user interface and, finally in step (3) we probabilistically regress the current locomotion features to the desired joint posture of the exoskeleton, using prediction uncertainty to model the impedance in a spring-damper system connecting the user and exoskeleton.

II

METHODS

In this section, we present the structure of the proposed approach. In Sec.

II-A

, we present the hardware used. In Sec.

II-B

, we present how we collected data for training the deep-learning models. In Sec.

II