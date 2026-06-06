<!-- page 1 -->
Adaptive Outer-Loop Control of Quadrotors via Reinforcement Learning
Vishnu Saj*
Graduate Research
Assistant
Texas A&M University
College Station, TX, USA
Sushil Vemuri
Graduate Research
Assistant
Texas A&M University
College Station, TX, USA
Dileep Kalathil
Associate Professor
Texas A&M University
College Station, TX, USA
Moble Benedict
Professor
Texas A&M University
College Station, TX, USA
Video: Click here to watch the video
ABSTRACT
Deep Reinforcement Learning (DRL) for quadrotor flight control typically relies on Domain Randomization (DR) for
sim-to-real transfer, resulting in overly conservative policies that struggle with dynamic disturbances. To overcome
this, we propose a novel adaptive control architecture that actively perceives and reacts to instantaneous perturbations.
First, we train an optimal outer-loop policy, then replace its reliance on ground-truth disturbance data with a Residual
Dynamics Predictor (RDP). The RDP estimates the external forces and moments acting on the aircraft in flight online
using only the history of states and control actions. For seamless hardware transfer, we introduce a data-efficient linear
calibration bridge and an online thrust correction mechanism that align the simulated latent space with reality using
mere seconds of flight data. Real-world validations on a Crazyflie micro-quadrotor demonstrate that our adaptive con-
troller significantly outperforms baselines, maintaining precise trajectory tracking under severe uncertainties including
mass variations, asymmetric payloads, and dynamic slung loads.
NOTATION
A,B
Spatial amplitudes for trajectory tracking
A
Reinforcement learning action space
at
Action vector at time t
c(·)
Reward function coefficients
dt, ˆdt
Ground-truth and estimated 6D disturbances
Fbias
Accumulated altitude correction thrust bias
fg
Force due to gravity
fnet,τnet
Net forces and moments acting on the quadrotor
FT,Q
Individual rotor thrust and reaction torque
H
Historical observation window length
Ixx,Iyy,Izz
Diagonal elements of the inertia tensor
KT,KQ
Rotor thrust and torque coefficients
L1,L2
Rotor distances from the geometric center
m
Nominal mass of the quadrotor
N
Total number of samples for RMSE calculation
pt,pdes
t
Actual and desired position vectors at time t
P
Environmental transition dynamics
qt
Attitude represented as a quaternion at time t
rt
Scalar reward at time t
R
Reinforcement learning reward function
S
Reinforcement learning state space
T
Time period for dynamic trajectories
vt
Linear velocity vector at time t
xt
Input feature vector for the RDP
x0,y0
Geometric center offset w.r.t. C.G.
*Corresponding author: vishnu02saj@tamu.edu
Presented at the Vertical Flight Society’s 82nd Annual Forum &
Technology Display, West Palm Beach, FL, USA, May 5–7, 2026.
Copyright © 2026 by the Vertical Flight Society. All rights reserved.
zdes,zref
Desired / reference flight altitude
γ
Reinforcement learning discount factor
η
Adaptive integration gain for altitude correction
θ
Policy parameters / pitch angle
πθ
Stochastic control policy
τm
Motor acceleration/deceleration time constant
φ,θ,ψ
Roll, pitch, and yaw Euler angles
ωt
Angular velocity vector at time t
Ω,Ωc
Actual and commanded rotor angular speeds
INTRODUCTION
Autonomous UAVs are increasingly deployed in complex,
real-world applications that demand high levels of agility, pre-
cision, and operational reliability, such as infrastructure in-
spection, search and rescue, and navigation in cluttered en-
vironments. Autonomous operation in these scenarios places
significant demands on the flight control system. Specifically,
the ability to execute precise maneuvers and maintain accurate
trajectory tracking under dynamic and uncertain environmen-
tal conditions remains a fundamental challenge and is critical
for the safety and efficacy of the overall system.
Traditionally, multirotor flight control relies on classical
architectures such as Proportional-Integral-Derivative (PID)
control, Linear Quadratic Regulators (LQR), or Model Pre-
dictive Control (MPC). While these methodologies are highly
effective in nominal flight regimes with accurately identi-
fied system parameters, they are heavily dependent on ex-
plicit mathematical plant models and linearized approxima-
tions.
Consequently, their tracking performance often de-
grades rapidly when the vehicle is subjected to unmodeled
dynamics, such as severe aerodynamic disturbances, actuator
1
arXiv:2605.16015v2  [cs.RO]  18 May 2026


<!-- page 2 -->
wear, or sudden variations in payload mass. Addressing these
uncertainties using classical methods typically requires ex-
haustive, time-consuming gain scheduling or complex adap-
tive control laws that are difficult to tune for highly nonlinear
flight envelopes.
Recently, deep Reinforcement Learning (RL) has emerged
(Refs. 1–3) as a promising alternative for synthesizing non-
linear control policies without the need for explicit analytical
modeling. By formulating the control problem as a Markov
Decision Process, RL agents can learn complex control map-
pings directly from simulated experience.
However, base
RL controllers trained in ideal simulation environments suffer
heavily from the simulation-to-reality or “sim-to-real” gap;
they often fail catastrophically when deployed on physical
hardware due to inevitable discrepancies between simulated
and real-world dynamics. The most prevalent solution to this
is Domain Randomization (DR), wherein the physical param-
eters of the simulator (e.g., mass, inertia, wind forces) are
heavily randomized during training. While DR yields a more
robust policy, it forces the network to learn a single, conser-
vative control strategy that maximizes the worst-case perfor-
mance across all possible randomized environments. This in-
herently sacrifices agility and optimal performance in nominal
conditions, as the controller remains oblivious to the true, in-
stantaneous state of the environment.
This paper seeks to address the limitations of both baseline
and robust RL controllers. Instead of relying on a static, con-
servative policy, we investigate whether an RL controller can
dynamically adapt to its environment, actively altering its be-
havior in response to the specific disturbances it is experienc-
ing. We pose the following central research question:
How can a learning-based flight controller dynam-
ically estimate unmodeled physical perturbations
from past states and use this latent information to
maintain optimal tracking performance while seam-
lessly bridging the sim-to-real gap?
To answer this, we propose a novel adaptive control architec-
ture. The main contributions of this work are as follows:
• Adaptive RL Controller: We develop a reinforcement
learning-based outer-loop controller trained alongside an
“oracle” that has access to ground-truth environmental
disturbances, allowing the policy to learn optimal re-
sponses to varying external forces and moments.
• Residual Dynamics Predictor (RDP): We design a re-
current neural network RDP that acts as a learned state
estimator. By observing a brief temporal history of ve-
hicle states and control actions, this RDP implicitly pre-
dicts 6D physical perturbations, effectively replacing the
Oracle during real-world deployment.
• Data-Efficient Sim-to-Real Bridge: We introduce a
computationally lightweight linear calibration layer that
maps the simulation-trained estimator’s latent space to
Figure 1. Overview of the proposed adaptive control ar-
chitecture. An ML-based RDP dynamically estimates un-
modeled 6D physical perturbations using past states and
motor commands, allowing the RL-based Adaptive Con-
troller to react to instantaneous disturbances and seam-
lessly bridge the sim-to-real gap.
physical reality using a minimal, few-shot real-world
dataset, effectively neutralizing steady-state sim-to-real
offsets without the need for extensive retraining.
RELATED WORK
RL has increasingly been adopted to synthesize flight con-
trollers that can handle the nonlinear dynamics of multirotor
vehicles (Refs. 4–6). In our previous work (Ref. ?), we trained
a baseline RL controller and demonstrated its superiority over
classical Proportional-Integral-Derivative (PID) architectures
in nominal flight conditions. Specifically, the RL-based policy
exhibited significantly better tracking performance character-
ized by lower phase lag and a higher stability margin than the
PID baseline. However, its performance degrades when ex-
posed to out-of-distribution physical perturbations, motivat-
ing the need for dynamic adaptation.
To address the limitations of static policies in dynamic envi-
ronments, recent breakthroughs in legged robotics introduced
Rapid Motor Adaptation (RMA) (Ref. 7). RMA enables di-
rect deployment to novel environments by using a history of
observed states to continuously infer a latent representation of
the environment. The adaptive control architecture proposed
in this paper is fundamentally inspired by the RMA frame-
work, seeking to translate its success from ground locomotion
to the highly dynamic domain of aerial flight.
The concept of rapid adaptation has recently been extended
to quadrotors. For instance, a recent work (Ref. 8) has suc-
cessfully implemented an RMA-style architecture for aerial
vehicles. However, their methodology relies on abstract latent
variables to encapsulate environmental uncertainties, whereas
our approach is designed to explicitly estimate the physical
6D external forces and moments acting on the vehicle. Fur-
thermore, their policy is trained end-to-end using a trajec-
tory tracking objective. In contrast, our method relies on a
position-hold objective during training, which we found pro-
motes stable, aggressive recovery from severe disturbances
while naturally generalizing to dynamic trajectory tracking.
2

[CAPTION] Figure 1. Overview of the proposed adaptive control ar-


<!-- page 3 -->
Another notable advancement is RAPTOR (Ref. 9), which
presents a foundation policy for quadrotor control. However,
this approach is computationally intensive and complex, re-
quiring the parallel training of hundreds of specialized expert
controllers followed by large-scale imitation learning to distill
them into a single policy. Similar to (Ref. 8), RAPTOR does
not explicitly estimate external forces and moments. Con-
versely, our proposed architecture efficiently trains a single
robust policy paired with an explicit, physics-grounded RDP,
offering both a computationally streamlined training pipeline
and a highly interpretable state estimation mechanism.
PRELIMINARIES AND SYSTEM
MODELING
RL Problem Formulation
We formulate the quadrotor control problem as a Markov De-
cision Process (MDP) defined by the tuple (S ,A ,P,R,γ).
At each discrete time step t, the agent observes the state
st ∈S , samples an action at ∈A from a stochastic pol-
icy πθ(at|st) parameterized by θ, and transitions to a new
state st+1 according to the unknown environmental transition
dynamics P(st+1|st,at). The agent subsequently receives a
scalar reward rt = R(st,at). The objective is to find the op-
timal policy parameters θ ∗that maximize the expected cu-
mulative discounted reward, J(θ) = Eπθ
 
∑T
t=0 γtrt
 
, where
γ ∈[0,1) is the discount factor and T is the episode horizon.
GPU-Accelerated Simulation Environment
We employ NVIDIA Isaac Lab (Ref. 10), a GPU-accelerated
robotics framework, for high-performance rigid-body simula-
tion and vectorized training. To achieve precise control over
modeling fidelity without altering the core physics engine, we
define a custom quadrotor model. Rigid-body kinematics are
handled by Isaac Lab, while aerodynamic forces, moments,
and actuator dynamics are computed via user-defined exter-
nal models and applied as external wrenches at each simula-
tion step. We simulate 8,192 environments in parallel. The
physics engine steps at 500 Hz (∆t = 0.002 s), while the RL
policy operates with a control decimation factor of 10, result-
ing in a control rate of 50 Hz. Training episodes last up to 5.0
seconds, terminating early if safety boundaries are violated.
Quadrotor Dynamics and Actuator Modeling
The modeled platform is based on the Bitcraze Crazyflie 2.X
(Ref. 11) (m = 0.0315 kg, Ixx = Iyy = 1.4 × 10−5 kg · m2,
Izz = 2.17 × 10−5 kg · m2).
Operating primarily in near-
hover regimes, we assume rotor thrust FT and torque Q scale
quadratically with angular speed Ω:
FT = KTΩ2,
Q = KQΩ2,
(1)
where KT and KQ are experimentally identified coefficients.
Individual rotor thrusts are bounded between 0.028 N and
0.148 N.
To bridge the sim-to-real gap, actuator dynamics are modeled
as a first-order lag capturing motor and ESC response times:
˙Ω= 1
τm
(Ωc −Ω),
(2)
where Ωc is the commanded speed and τm = 0.03 s is the time
constant.
Figure 2.
Schematic of the quadrotor body frame and
rotor configuration.
The mathematical model assumes
a symmetric cross configuration with explicitly defined
center-of-gravity offsets and moment arms, which are uti-
lized to dynamically compute the total aerodynamic forces
and moments acting on the vehicle.
Assuming a symmetric cross configuration (Fig. 2) with
center-of-gravity offsets (x0,y0) and moment arms L1 = L2 =
0.028 m, the total aerodynamic force and moments are:
faero =


0
0
KT ∑4
i=1 Ω2
i

,
(3)
τaero =


−KT(L1 −x0)
−KT(L1 +x0)
KT(L2 +x0)
−KT(L2 +x0)
−KQ
KQ
KT(L1 −x0)
KT(L1 +x0)
KT(L2 −x0)
−KT(L2 −x0)
KQ
−KQ




Ω2
1
Ω2
2
Ω2
3
Ω2
4

.
(4)
The net forces and moments acting on the vehicle are obtained
by combining these aerodynamic components with gravity
(fg).
Hardware and Control Architecture
We validate our approach on the physical Crazyflie 2.X micro-
quadrotor. State estimation relies entirely on the onboard Ex-
tended Kalman Filter fusing IMU and Flow Deck (optical
flow and time-of-flight) measurements, eliminating the need
for external motion capture. We employ a cascaded control
architecture (Fig. 4). The RL policy acts as an offboard outer-
loop controller, outputting desired body rates and collective
thrust. These commands are transmitted to the Crazyflie and
3

[CAPTION] Figure 2.
Schematic of the quadrotor body frame and


<!-- page 4 -->
tracked by its onboard, high-bandwidth PID rate controllers.
This modularity isolates the learning-based outer loop while
preserving fast and reliable inner-loop stabilization.
Figure 3.
The Bitcraze Crazyflie 2.X micro-quadrotor
platform utilized for real-world validation experiments.
The vehicle relies entirely on its onboard IMU and Flow
Deck for precise state estimation, enabling fully au-
tonomous operation without relying on an external motion
capture system.
Figure 4. Block diagram of the proposed cascaded control
architecture. The high-level reinforcement learning policy
operates as an offboard outer loop, generating target col-
lective thrust and desired body rates. These setpoints are
subsequently transmitted to the vehicle and tracked by the
high-bandwidth onboard PID rate controllers.
RL State-Action Space
The RL policy functions as a high-level outer-loop controller.
• Action Space (A ⊂R4): Collective target thrust and de-
sired body-frame angular rates (roll, pitch, yaw).
• Observation Space: The instantaneous observation ot ∈
R22 consists of the vehicle’s position error pt ∈R3, the
flattened rotation matrix Rt ∈R9 (used to avoid gimbal
lock), linear velocity vt ∈R3, angular velocity ωt ∈R3,
and the previous action at−1 ∈R4. To account for par-
tial observability, such as sensor latency and unmodeled
transport delays, the final MDP state st ∈R32×22 is con-
structed by stacking a temporal window of the last 32
observations.
• Observation Noise: To simulate real-world estimator
drift, zero-mean Gaussian noise is injected into posi-
tions (0.001 m), orientations (0.0003 rad), linear veloci-
ties (0.003 m/s), and angular velocities (0.002 rad/s).
Reward Formulation
The objective is to maintain hover at the origin while mini-
mizing control effort. The reward at time t is:
rt = cp∥pt∥2 +cψψ2
t +cv∥vt∥2 +cω∥ωt∥2
+cT(∆Tt)2 +c∆ω∥∆ωcmd,t∥2 +cs +cd ·1crash
(5)
Coefficients are empirically tuned:
position cp = −30.0,
yaw cψ = −2.0, linear velocity cv = −3.0, angular velocity
cω = −0.5, thrust smoothing cT = −1.0, and rate smoothing
c∆ω = −0.5. The agent receives a survival bonus cs = 4.0 and
a crash penalty cd = −100.0 (triggered if boundaries > 0.6 m
are exceeded).
Network Architecture and Training
Policies are trained using Proximal Policy Optimization
(PPO) (Ref. 12). To handle partial observability, the policy
and value networks are parameterized as Recurrent Neural
Networks (RNNs) with two layers and a hidden size of 64.
Training uses a learning rate of 0.001, γ = 0.99, and λ = 0.95,
optimized over 8 epochs and 4 mini-batches per update. Sur-
rogate clipping and value loss clipping are both 0.2. Training
converges over 100,000 algorithm timesteps.
Baseline Controllers
To evaluate our adaptive approach, we train two comparative
baselines:
• Base Controller: Trained under ideal conditions with no
external perturbations.
• Robust Controller: Trained using Domain Randomiza-
tion (DR). Random external forces U (−0.1,0.1) N and
torques U (−0.001,0.001) Nm are injected at the center
of mass to encourage a conservative, worst-case mitiga-
tion strategy.
METHODOLOGY
Having established the foundational reinforcement learning
formulation and the baseline robust policy, we now detail our
proposed adaptive control architecture. The core objective is
to move beyond the conservative, worst-case optimization of
Domain Randomization (DR) by enabling the controller to
4

[CAPTION] Figure 3.
The Bitcraze Crazyflie 2.X micro-quadrotor

[CAPTION] Figure 4. Block diagram of the proposed cascaded control


<!-- page 5 -->
dynamically perceive and react to instantaneous environmen-
tal disturbances. This is achieved through a three-stage pro-
gression culminating in a learned disturbance estimator and a
data-efficient sim-to-real calibration bridge.
Adaptive Controller (Oracle Baseline)
To establish an upper bound on achievable flight performance
under uncertain conditions, we first develop an Adaptive Con-
troller trained alongside an “Oracle”. This policy is trained
in the same heavily randomized simulation environment as
the Robust Controller, but its observation space is explic-
itly augmented to include the ground-truth external pertur-
bations. Specifically, the nominal observation vector is ex-
panded to R32×28, appending the 6-dimensional disturbance
vector dt ∈R6 (comprising three force components and three
moment components acting on the vehicle’s center of grav-
ity) at each timestep. By operating under the assumption of
full state observability, the policy is no longer forced to learn
a generalized, conservative compromise. Instead, it learns a
specialized control manifold that maps specific dynamic vari-
ations to optimal targeted corrective actions. While this ora-
cle controller demonstrates superior tracking performance, it
cannot be deployed directly onto physical micro-UAVs, which
lack the necessary force-moment sensors to measure 6D aero-
dynamic disturbances in real-time.
Residual Dynamics Predictor (RDP)
To bridge the gap between the Oracle policy and the limita-
tions in sensing the external forces and moments, we intro-
duce the RDP: a Recurrent Neural Network (RNN) designed
to function as a nonlinear, implicit state estimator. This RDP
estimates the unobserved 6D environmental perturbations, es-
sentially the external forces and moments, purely from a tem-
poral history of vehicle states and low-level actuation sig-
nals. We formulate this system identification problem as a
sequence-to-vector regression task. The training dataset com-
prises approximately 60,000 simulated flight episodes, gen-
erated by deploying the pre-trained Oracle controller under
randomized disturbances. To accurately infer external forces
and moments, the network must observe the residual dynam-
ics of the quadrotor. Thus, the input feature vector xt ∈R26
consists of the 18-dimensional vehicle kinematic state, the
4-dimensional previous outer-loop action, and crucially, the
4-dimensional Pulse Width Modulation (PWM) signals com-
manded to the individual motors by the inner-loop PID con-
troller. Including the PWM signals is vital; higher-level thrust
and body-rate setpoints fail to uniquely capture transient ac-
tuator states, motor lag, and inner-loop saturations necessary
for accurate instantaneous force and moment estimation. The
architecture employs a Gated Recurrent Unit (GRU) (Ref. 13)
to capture the temporal evolution of the disturbances. It pro-
cesses a rolling historical window of H = 64 standardized
observations (Xt−H:t) through two GRU layers (hidden di-
mension of 64), followed by a linear readout head that out-
puts the predicted 6D perturbation vector ˆdt. During real-
world inference, the predicted perturbation is smoothed using
a rolling buffer of length 32 to mitigate high-frequency pre-
diction noise, and the temporal mean is concatenated with the
standard kinematic observation to query the Adaptive Con-
troller.
Data-Efficient Sim-to-Real Calibration Bridge
When transferring the purely simulation-trained RDP to phys-
ical hardware (e.g., the Crazyflie platform), a distinct sim-
to-real gap inevitably emerges.
Unmodeled physical dy-
namics such as slight mass distribution asymmetries, inher-
ent motor wear, degraded battery voltage curves, and near-
surface ground effects cause the GRU to predict a persistent,
non-zero offset for external forces and moments, even in a
nominal, disturbance-free setting. Rather than undertaking
the prohibitively expensive process of collecting large-scale,
ground-truth real-world datasets to fine-tune the recurrent net-
work, we introduce a highly data-efficient linear calibration
bridge. Utilizing a minimal physical dataset of merely three
brief flight samples, we compare the expected nominal states
against the GRU’s real-world predictions and fit a multivariate
linear regression model to the outputs. Crucially, the learned
slope of this regression was found to be approximately unity
(m ≈1.0), demonstrating that the simulated and physical dy-
namic manifolds are structurally isomorphic and differ pri-
marily by a constant affine translation. This linear bridge op-
erates as a computationally negligible final layer that effec-
tively neutralizes the steady-state sim-to-real offset. By apply-
ing this translation, the latent space of the simulation-trained
estimator is precisely aligned with physical reality before it
is passed to the RL policy, ensuring robust and accurate real-
world flight performance.
Furthermore, to address residual thrust mismatches that may
manifest as steady-state altitude errors during physical de-
ployment, we augment the calibration bridge with an online
altitude correction mechanism. This mechanism computes an
accumulated thrust bias, Fbias, based on the real-time altitude
tracking error. At each timestep, the bias is updated according
to:
Fbias ←Fbias +η(zdes −z),
(6)
where zdes is the desired altitude, z is the current measured
altitude, and η is a small adaptive integration gain. This ac-
cumulated bias is continuously added to the estimator’s pre-
diction of the vertical disturbance force, ˆFz. By integrating
the altitude error over time, this mechanism steadily corrects
for unmodeled constant offsets in aerodynamic thrust (such
as battery depletion) and evaluates the precise external ver-
tical force prediction required for the policy to maintain the
desired altitude.
EXPERIMENTS
To evaluate the efficacy of the proposed adaptive architecture,
we conducted a systematic series of real-world experiments.
The primary objectives of these experiments are twofold: (1)
5


<!-- page 6 -->
to quantify the position tracking performance of the adap-
tive approach compared to baselines under varying distur-
bance regimes, and (2) to validate the accuracy of the RDP’s
real-time disturbance estimations against known ground-truth
physics.
Experimental Setup
Evaluated Controllers: We compare four distinct control
configurations across all experimental scenarios:
(i) Base Controller: Trained in an idealized simulation.
(ii) Robust Controller: Trained using Domain Random-
ization (forces/moments).
(iii) Adaptive Controller + Oracle: The theoretical perfor-
mance upper bound, utilizing ground-truth disturbance
information.
(iv) Adaptive Controller + RDP: Our fully deployable sys-
tem, using the GRU-based estimator to infer distur-
bances online.
Evaluation Metrics: The primary performance metric across
all tasks is the Root Mean Square Error (RMSE) of the posi-
tion tracking error, defined as:
RMSE =
s
1
N
N
∑
k=1
∥pk −pdes
k ∥2,
(7)
where pk ∈R3 and pdes
k
∈R3 denote the actual and desired
positions at time step k, respectively, and N is the total num-
ber of samples. In the tabulated results, external disturbance
conditions are quantified by the payload mass expressed as
a percentage increase relative to the nominal vehicle mass
(m = 0.0315 kg).
Evaluation Scenarios:
To rigorously stress-test the con-
trollers, we perform three distinct real-world evaluations. We
explicitly define the testing brackets for each task to evaluate
progressive difficulty:
• Central Payload Mass Variation: An additional payload
is attached directly to the vehicle’s center of gravity. This
introduces a constant, known vertical force perturbation
without altering the rotational dynamics.
In our experi-
ments, the payload mass is explicitly varied across four dis-
crete tiers: 0%, 7.5%, 15%, and 25% of the nominal quadro-
tor mass.
• Asymmetric Payload Distribution: Mass is attached at the
end of one of the quadrotor’s arms. This shifts the center of
gravity and introduces coupled, constant force and moment
perturbations. The payload offsets tested correspond to 0%,
1%, 7%, and 11% relative mass increases.
• Dynamic Trajectory Tracking with Suspended Payload:
A mass is suspended from the drone via a tether. The ve-
hicle is commanded to track a planar figure-8 trajectory de-
fined by a Lissajous curve:
pdes(t) =


Acos
  2πt
T
 
Bsin
  4πt
T
 
zref

,
(8)
where A and B are the spatial amplitudes, zref is the constant
target altitude, and T is the total trajectory duration. Vary-
ing the period T from 15 s down to 3 s scales the required
velocities and accelerations. As the quadrotor maneuvers,
the suspended payload continuously oscillates, generating
complex, unmodeled, and time-varying forces and moments
acting on the airframe.
Experimental Results
Central Payload Mass Variation: Table 1 summarizes the
position hold RMSE under varying central payload masses. A
key insight from this table is the clear divergence in controller
robustness as weight increases. The Base Controller exhibits
progressive degradation, dropping from an RMSE of 0.074
m at 0% to 0.160 m at 25%. This occurs because its fixed
internal dynamics model assumes a constant mass; as the ve-
hicle becomes heavier, the baseline policy outputs insufficient
collective thrust, resulting in altitude sag and large positional
errors.
Notably, the Robust Controller exhibits the worst performance
even in the base scenario (0% variation), yielding an RMSE of
0.111 m. This poor nominal performance is characterized by
severe, high-frequency oscillations. Because the Robust Con-
troller was trained via Domain Randomization to anticipate
extreme, randomized 6-DOF disturbances at every timestep, it
learns an overly conservative, hyper-reactive policy. In a calm
environment, this causes the controller to continuously over-
compensate for minor sensor noise, leading to visible jitter.
Due to this severe oscillatory instability, the Robust Controller
was deemed unsafe for hardware deployment under heavier
payload conditions and was not tested further (denoted by
‘x’).
In contrast, the adaptive controller demonstrates consistently
strong performance across all configurations, with the vari-
ant using the learned RDP outperforming even the oracle-
based version (0.024 m to 0.030 m). While this may appear
counter-intuitive, it can be explained by considering practical
and modeling limitations. The oracle assumes access to exact
disturbance information; however, in real experiments, small
discrepancies inevitably arise due to factors such as slight
variations in payload mass and imperfect placement relative
to the center of mass. These introduce subtle errors in the
computed ”ground-truth” forces and moments. Furthermore,
due to sim-to-real mismatches, directly providing these rigid
physical quantities to the policy may not align optimally with
the dynamics encountered during deployment.
6


<!-- page 7 -->
The learned RDP, on the other hand, produces a latent dis-
turbance representation that implicitly accounts for such in-
consistencies. This behavior is illustrated in Figure 5, where
the estimated mass exhibits a bounded error of approximately
3% to 7% relative to the true value. Although the estimate
does not exactly match the ground truth, it captures the over-
all trend and relative variations effectively. Importantly, pre-
cise disturbance reconstruction is not required; rather, the con-
troller benefits from a consistent and responsive signal that
reflects the underlying system behavior. As a result, the RDP
provides disturbance information that is better aligned with
the actual flight dynamics than idealized oracle inputs.
A similar observation can be made for the baseline con-
troller, where the RMSE varies only marginally for small
mass changes (0% to 7%), indicating that the system is rel-
atively insensitive to minor deviations from nominal condi-
tions. However, as the mismatch increases, the limitations
of fixed dynamics become more pronounced. In this context,
the improved performance of the adaptive controller is further
supported by the Sim-to-Real Calibration Bridge methodol-
ogy introduced earlier. By accounting for systematic discrep-
ancies between simulation and real-world dynamics, this ap-
proach enables the learned disturbance representation to bet-
ter align with the true system behavior. Overall, these results
suggest that the effectiveness of the adaptive approach stems
not from exact physical estimation, but from its ability to pro-
vide control-relevant disturbance information while implicitly
compensating for modeling mismatches.
Controller
0%
7.5%
15%
25%
Base
0.074
0.089
0.101
0.160
Robust
0.111
x
x
x
Adaptive
+ Oracle
0.038
0.033
0.028
0.040
Adaptive
+ RDP
0.024
0.028
0.029
0.030
Table 1. Position hold Root Mean Square Error (RMSE)
under varying central payload mass configurations. The
columns represent the additional payload mass expressed
as a percentage of the nominal vehicle mass.
The Ro-
bust Controller exhibited severe oscillatory instability un-
der nominal conditions and was deemed unsafe for phys-
ical deployment under heavier loads (denoted by ‘x’). In
contrast, the Adaptive Controller paired with the learned
RDP consistently outperformed the baselines, matching or
exceeding the theoretical Oracle performance.
Asymmetric Payload Distribution: When the payload is off-
set to the tip of a single arm, the system experiences coupled
translational (force) and rotational (torque) disturbances, sig-
nificantly increasing the task difficulty. As shown in Table 2,
the baseline controller exhibits a gradual increase in track-
ing error, reaching 0.089 m at 11% asymmetry. However,
the degradation remains relatively moderate compared to the
nominal case, indicating that the baseline controller is able to
Figure 5. Real-time estimation of the added payload mass
by the RDP against true mass during the central payload
variation experiment . The GRU-based network demon-
strates rapid, step-like convergence to the newly altered
physical dynamics upon takeoff, maintaining a bounded
steady-state estimation error of approximately 3% to 7%,
which proves sufficient for the RL policy to maintain opti-
mal tracking.
partially accommodate small moment variations despite lack-
ing an explicit mechanism to decouple the induced torques
from its thrust commands.
Both the adaptive controller with oracle disturbance informa-
tion and the adaptive controller with the learned RDP demon-
strate significantly improved performance compared to the
baseline controller. Their position errors remain consistently
lower across all payload configurations, highlighting their
ability to effectively compensate for the coupled translational
and rotational disturbances. Furthermore, the performance of
the adaptive controller with the learned RDP closely matches
that of the oracle-based variant, indicating that the RDP is
able to provide sufficiently accurate disturbance estimates for
effective control.
Figures 6, 7, and 8 provide crucial insights into how the
RDP resolves this complexity. Figure 6 shows that the to-
tal mass estimation remains stable across different configura-
tions. More importantly, Figures 7 and 8 demonstrate that the
RDP is able to accurately capture the induced roll (Mx) and
pitch (My) moments resulting from the offset center of gravity.
The predicted moments closely match the ground-truth values
with minimal deviation, indicating high-fidelity estimation of
the disturbance-induced torques. This level of accuracy en-
ables the controller to effectively compensate for the coupled
rotational effects, contributing to improved overall tracking
performance.
A key factor enabling this accurate moment estimation is
the inclusion of PWM control inputs in the RDP. These in-
puts provide implicit information about the thrust distribution
across the motors, allowing the RDP to infer the resulting mo-
ments acting on the system. Under the assumption of identi-
7

[CAPTION] Table 1. Position hold Root Mean Square Error (RMSE)

[CAPTION] Figure 5. Real-time estimation of the added payload mass


<!-- page 8 -->
cal motor characteristics, the mapping from PWM signals to
generated forces and moments remains consistent, leading to
reliable disturbance estimation. However, in practical scenar-
ios, motor characteristics may diverge over time due to wear,
manufacturing variability, or partial degradation, introducing
asymmetry in thrust generation. In such cases, PWM signals
alone may no longer serve as a sufficient indicator of the true
forces and moments, potentially leading to degraded estima-
tion accuracy. This highlights a limitation of the current for-
mulation and suggests that incorporating additional informa-
tion, such as motor-specific calibration or online identifica-
tion, could further improve the robustness of the RDP under
long-term real-world operation.
Controller
0%
4%
7%
11%
Base
0.074
0.080
0.082
0.089
Robust
0.111
x
x
x
Adaptive
+ Oracle
0.038
0.037
0.039
0.027
Adaptive
+ RDP
0.024
0.033
0.026
0.039
Table 2. Position hold RMSE under asymmetric payload
configurations, where mass is attached to a single rotor
arm to induce coupled translational and rotational distur-
bances. Columns indicate the percentage of mass increase.
The Adaptive Controller with the learned RDP demon-
strates superior robustness, effectively mitigating the in-
duced roll and pitch moments, whereas the Base Con-
troller’s performance steadily degrades as the asymmetry
increases.
Figure 6. Comparison of the additional mass added pre-
dicted by the RDP and the true value(also represented as
% of drone mass) under asymmetric loading conditions.
Despite the complex coupled dynamics introduced by the
offset payload, the vertical force estimation remains stable
and accurate.
Figure 7. Real-time estimation of the induced roll moment
(Mx) by the RDP during the asymmetric payload experi-
ment. Following a brief transient adaptation phase dur-
ing takeoff, the RDP accurately converges to the ground-
truth steady-state torques utilizing only historical kine-
matics and motor PWM signals.
Figure 8. Real-time estimation of the induced pitch mo-
ment (My) by the RDP under asymmetric loading. The
network successfully isolates the pitch disturbance from
the roll and vertical force components, enabling the policy
to perfectly counterbalance the offset payload.
Dynamic Trajectory Tracking with Suspended Payload:
For dynamic tracking, a 4.7g (∼15%) payload suspended by
a thread of length equal to the arm length introduces continu-
ous, oscillatory disturbances. Figure 9 shows the planar x–
y trajectories of the adaptive controller tracking the figure-
8 reference across decreasing time periods (T).
As T de-
creases from 15 s to 3 s, the required velocities and accelera-
tions increase significantly, resulting in more aggressive mo-
tion. This, in turn, induces larger oscillations of the suspended
payload, generating stronger and more rapidly varying distur-
8

[CAPTION] Table 2. Position hold RMSE under asymmetric payload

[CAPTION] Figure 6. Comparison of the additional mass added pre-

[CAPTION] Figure 7. Real-time estimation of the induced roll moment

[CAPTION] Figure 8. Real-time estimation of the induced pitch mo-


<!-- page 9 -->
Figure 9. Planar x–y trajectory tracking performance of the adaptive controller carrying a 4.7 g suspended payload
attached via a thread of length equal to the arm length. The reference trajectory is a Lissajous figure-8 curve with
decreasing time periods (T). As T decreases from 15 s to 3 s, the required velocities and accelerations increase signifi-
cantly, inducing aggressive pendulum dynamics. Remarkably, the controller The controller maintains stable flight and
continues to follow the reference trajectory despite the presence of significant unmodeled oscillatory disturbances.
bance forces and moments acting on the vehicle.
T (s)
RMSE
Max Err
Std
Delay
Phase
(m)
(m)
(m)
(s)
(deg)
15
0.055
0.108
0.024
-0.140
-3.4
10
0.063
0.112
0.023
-0.080
-2.9
5
0.118
0.192
0.041
0.080
5.8
3
0.141
0.220
0.036
0.100
12.0
Table 3. Trajectory tracking performance under increas-
ing aggressiveness. As the trajectory period T decreases,
tracking error increases, accompanied by growing time
delay and phase lag (computed along the x-axis), indicat-
ing bandwidth-limited response.
Table 3 quantifies the effect of increasing trajectory aggres-
siveness on tracking performance. As T decreases from 15
s to 3 s, the RMSE increases from 0.055 m to 0.141 m, ac-
companied by a rise in the standard deviation of the error,
indicating a more oscillatory response under faster trajectory
execution. This trend is further explained by the observed
variation in tracking delay and phase. The delay and phase
were computed along the x-axis, where the dominant motion
of the trajectory makes tracking errors most pronounced. For
slower trajectories (T = 15 s and 10 s), the delay is small and
slightly negative (−0.14 s and −0.08 s), corresponding to a
slight phase lead. This behavior suggests that the learned pol-
icy is able to implicitly anticipate the system dynamics based
on recent state information, effectively exhibiting predictive
characteristics. As the trajectory becomes more aggressive,
this lead diminishes and transitions into a lag, with the delay
increasing to 0.10 s for T = 3 s and a corresponding phase
lag of 12◦. This indicates that the system increasingly lags
behind the reference as the excitation frequency rises. De-
spite this increase in tracking error, the maximum deviation
remains bounded within a limited range (0.108 m to 0.220 m),
and no divergence is observed across all experiments. This in-
dicates that the system maintains stable closed-loop behavior
even under highly dynamic conditions. Overall, the observed
performance trends are consistent with bandwidth limitations
of the closed-loop system, where higher-frequency inputs re-
duce the controller’s ability to maintain anticipatory behavior
and track the reference with the same fidelity as slower trajec-
tories.
The underlying mechanism enabling this robustness is illus-
trated in Figure 10. The estimated disturbance quantities, in-
cluding the vertical force (Fz) and the induced roll (Mx) and
pitch (My) moments, exhibit strongly time-varying behavior
that reflects the oscillatory dynamics of the suspended pay-
load. As the trajectory becomes more aggressive (lower T),
both the frequency and amplitude of these oscillations in-
crease, indicating stronger and more rapidly varying distur-
bance effects due to intensified payload motion.
Throughout most of the trajectory,
the vertical force
estimate—and
the
corresponding
inferred
payload
mass—remains relatively consistent, suggesting that the
RDP is able to maintain a stable representation of the overall
9

[CAPTION] Figure 9. Planar x–y trajectory tracking performance of the adaptive controller carrying a 4.7 g suspended payload

[CAPTION] Table 3. Trajectory tracking performance under increas-

[CAPTION] Table 3 quantifies the effect of increasing trajectory aggres-


<!-- page 10 -->
Figure 10. Time-series of the estimated disturbance quantities—vertical force (Fz), roll moment (Mx), and pitch moment
(My)—during the dynamic slung load trajectory tracking experiments. The highly oscillatory predictions actively cap-
ture the real-time pendulum physics of the swinging payload, dynamically scaling in amplitude as the trajectory period
T decreases and the maneuver aggressiveness increases.
load. On average, the RDP predicts a mean mass in the range
of 3.4–4.7 g, underestimating the true payload mass of 4.7
g, which is consistent with the estimation error observed in
the previous experiment. However, transient inaccuracies are
observed during takeoff and at the initial phase of trajectory
execution, where the disturbance estimates deviate from
their steady-state values. This behavior is expected, as the
system undergoes rapid changes in dynamics and the RDP
requires a short period to converge. Once the trajectory enters
steady-state motion, the vertical force estimates stabilize and
provide a reliable representation of the disturbance.
In contrast, the estimated moments oscillate about a mean
value, capturing the time-varying moments induced by the
swinging payload. Interestingly, two distinct mean levels are
observed in the moment estimates across different trials, cor-
responding to effective payload offsets of approximately 14
mm and 7 mm from the center. This indicates that the RDP
is able to infer not only the magnitude of the disturbance but
also variations in the effective location of the payload. This
behavior can be attributed to the physical interaction between
the suspended payload and the vehicle structure, where the at-
tachment point is free to shift slightly along the arm. Due to
minor geometric features or contact effects, the payload tends
to settle at different effective positions, resulting in distinct
moment offsets. Despite this variability, the RDP successfully
captures these changes, allowing the controller to account for
both the oscillatory and quasi-static components of the dis-
turbance, thereby maintaining stable and consistent tracking
performance.
Despite these transient and configuration-dependent varia-
tions, the RDP successfully captures both the oscillatory and
quasi-static components of the disturbance, enabling the con-
troller to maintain stable and consistent tracking performance.
CONCLUSIONS AND FUTURE WORK
This work presented a learning-based adaptive control archi-
tecture for micro-quadrotors operating under severe dynamic
uncertainties. By transitioning from conservative domain ran-
domization to an active disturbance-rejection framework, this
research yields several key insights into bridging the sim-to-
real gap:
• Active Adaptation Outperforms Passive Robustness: We
demonstrated that relying exclusively on Domain Random-
ization (DR) to handle unmodeled perturbations forces the
10

[CAPTION] Figure 10. Time-series of the estimated disturbance quantities—vertical force (Fz), roll moment (Mx), and pitch moment


<!-- page 11 -->
policy to learn overly conservative, hyper-reactive behav-
iors. On physical hardware, this passive robustness trans-
lates to severe oscillatory instability even in nominal con-
ditions. Active, real-time adaptation is fundamentally re-
quired to maintain stable and optimal flight tracking.
• Implicit Disturbance Estimation from Onboard Signals:
Our experiments validate that complex 6D external forces
and moments can be accurately reconstructed without spe-
cialized force-torque sensors. A brief temporal history of
standard onboard kinematics coupled with low-level actua-
tor signals (PWM) contains sufficient implicit information
for a recurrent network (the RDP) to isolate and estimate
independent translational and rotational disturbances.
• Control-Relevant vs.
Exact Physical Estimation: Re-
sults from the payload variation experiments indicate that
the RDP does not need to perfectly reconstruct the ground-
truth physical quantities to be highly effective. A bounded
estimation error (e.g. the observed 3% to 7% mass error)
is entirely permissible. As long as the network generates
a consistent and control-relevant representation of the dis-
turbance trends, the RL policy can appropriately scale its
compensatory actions to maintain high-precision tracking.
• Data-Efficient Sim-to-Real Calibration: A key finding of
this work is that the discrepancy between simulated dis-
turbance predictions and real-world behavior can be well-
approximated by a low-order (approximately affine) rela-
tionship. As a result, bridging the sim-to-real gap does not
require complex nonlinear domain adaptation or extensive
real-world retraining. Instead, it can be effectively miti-
gated using a computationally lightweight linear calibration
bridge combined with an online thrust accumulator, requir-
ing only a few seconds of flight data to correct for steady-
state offsets (e.g., battery voltage variations).
• Efficacy of Decoupled, Oracle-Guided Training: Train-
ing an optimal outer-loop policy using a ground-truth “Ora-
cle” before transitioning to a learned estimator circumvents
the instability and sample inefficiency often associated with
end-to-end RL in partially observable environments. This
decoupled approach cleanly separates the control objective
from the state-estimation problem, ensuring that the pol-
icy learns an optimal disturbance-response manifold. This
modularity was crucial for achieving aggressive real-world
stabilization against complex, dynamic slung loads.
Future work will focus on extending this adaptive framework
to handle highly turbulent aerodynamic wind gusts. Address-
ing such severe aerodynamic disturbances is essential for de-
ploying these learned policies in more complex, real-world
missions, such as autonomous ship-deck landings, where the
vehicle must maintain precision within highly unsteady and
turbulent ship wakes. Furthermore, we plan to validate the
scalability of our approach by deploying the controller on
larger quadrotor platforms to assess the RDP’s performance
across different mass, inertia, and actuation scales. Finally,
a broader avenue for future research involves applying this
methodology to alternative aerial configurations, including
fixed-wing aircraft and hybrid vertical take-off and landing
(VTOL) tailsitters, thereby expanding the impact of learned
adaptive control to a wider range of challenging aerospace ap-
plications.
ACKNOWLEDGMENTS
This work is supported by the Office of Naval Research
(ONR) under Grant Number N000142312404. The views and
conclusions contained in this document are those of the au-
thors and should not be interpreted as representing the of-
ficial policies, either expressed or implied, of the Navy or
the U.S. Government. The U.S. Government is authorized to
reproduce and distribute reprints for Government purposes,
notwithstanding any copyright notation herein.
REFERENCES
1.
L. Bauersfeld, E. Kaufmann, P. Foehn, S. Sun, and
D. Scaramuzza, “Neurobem:
Hybrid aerodynamic
quadrotor model,” arXiv preprint arXiv:2106.08015,
2021.
2.
V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves,
I. Antonoglou, D. Wierstra, and M. Riedmiller, “Playing
atari with deep reinforcement learning,” arXiv preprint
arXiv:1312.5602, 2013.
3.
K. Wang, I. Javali, M. Bortkiewicz, B. Eysenbach, et al.,
“1000 layer networks for self-supervised rl: Scaling
depth can enable new goal-reaching capabilities,” arXiv
preprint arXiv:2503.14858, 2025.
4.
J. Hwangbo, I. Sa, R. Siegwart, and M. Hutter,
“Control of a quadrotor with reinforcement learning,”
IEEE Robotics and Automation Letters, vol. 2, no. 4,
pp. 2096–2103, 2017.
5.
J. Yoo, D. Jang, H. J. Kim, and K. H. Johansson, “Hy-
brid reinforcement learning control for a micro quadro-
tor flight,” IEEE Control Systems Letters, vol. 5, no. 2,
pp. 505–510, 2020.
6.
S.
Batra,
Z.
Huang,
A.
Petrenko,
T.
Kumar,
A. Molchanov, and G. S. Sukhatme, “Decentralized con-
trol of quadrotor swarms with end-to-end deep rein-
forcement learning,” in Conference on robot learning,
pp. 576–586, PMLR, 2022.
7.
A. Kumar, Z. Fu, D. Pathak, and J. Malik, “Rma: Rapid
motor adaptation for legged robots,” arXiv preprint
arXiv:2107.04034, 2021.
8.
D. Zhang, A. Loquercio, J. Tang, T.-H. Wang, J. Malik,
and M. W. Mueller, “A learning-based quadcopter con-
troller with extreme adaptation,” IEEE Transactions on
Robotics, 2025.
11


<!-- page 12 -->
9.
J. Eschmann, D. Albani, and G. Loianno, “Raptor: A
foundation policy for quadrotor control,” arXiv preprint
arXiv:2509.11481, 2025.
10.
M. Mittal, P. Roth, J. Tigue, A. Richard, O. Zhang,
P. Du, A. Serrano-Mu˜noz, X. Yao, R. Zurbr¨ugg,
N. Rudin, L. Wawrzyniak, M. Rakhsha, A. Denzler,
E. Heiden, A. Borovicka, O. Ahmed, I. Akinola, A. An-
war, M. T. Carlson, J. Y. Feng, A. Garg, R. Gasoto,
L. Gulich, Y. Guo, M. Gussert, A. Hansen, M. Kulkarni,
C. Li, W. Liu, V. Makoviychuk, G. Malczyk, H. Mazhar,
M. Moghani, A. Murali, M. Noseworthy, A. Poddubny,
N. Ratliff, W. Rehberg, C. Schwarke, R. Singh, J. L.
Smith, B. Tang, R. Thaker, M. Trepte, K. V. Wyk,
F. Yu, A. Millane, V. Ramasamy, R. Steiner, S. Subra-
manian, C. Volk, C. Chen, N. Jawale, A. V. Kuruttuku-
lam, M. A. Lin, A. Mandlekar, K. Patzwaldt, J. Welsh,
H. Zhao, F. Anes, J.-F. Lafleche, N. Mo¨enne-Loccoz,
S. Park, R. Stepinski, D. V. Gelder, C. Amevor, J. Car-
ius, J. Chang, A. H. Chen, P. de Heras Ciechomski,
G. Daviet, M. Mohajerani, J. von Muralt, V. Reutskyy,
M. Sauter, S. Schirm, E. L. Shi, P. Terdiman, K. Vilella,
T. Widmer, G. Yeoman, T. Chen, S. Grizan, C. Li,
L. Li, C. Smith, R. Wiltz, K. Alexis, Y. Chang, D. Chu,
L. J. Fan, F. Farshidian, A. Handa, S. Huang, M. Hut-
ter, Y. Narang, S. Pouya, S. Sheng, Y. Zhu, M. Mack-
lin, A. Moravanszky, P. Reist, Y. Guo, D. Hoeller,
and G. State, “Isaac lab: A gpu-accelerated simula-
tion framework for multi-modal robot learning,” arXiv
preprint arXiv:2511.04831, 2025.
11.
Bitcraze,
“Crazyflie
2.1
nano
quadcopter.”
https://www.bitcraze.io/products/
old-products/crazyflie-2-1/, 2024.
Ac-
cessed: 2026-04-15.
12.
J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and
O. Klimov, “Proximal policy optimization algorithms,”
CoRR, vol. abs/1707.06347, 2017.
13.
J. Chung, C. Gulcehre, K. Cho, and Y. Bengio, “Em-
pirical evaluation of gated recurrent neural networks on
sequence modeling,” arXiv preprint arXiv:1412.3555,
2014.
12