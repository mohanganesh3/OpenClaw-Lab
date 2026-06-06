[2202.11508] AI-enabled mm-Waveform Configuration for Autonomous Vehicles with Integrated Communication and Sensing

AI-enabled mm-Waveform Configuration for Autonomous Vehicles with Integrated Communication and Sensing

Nam H. Chu,
Diep N. Nguyen,
Dinh Thai Hoang, Quoc-Viet Pham,

Khoa T. Phan, Won-Joo Hwang, and Eryk Dutkiewicz

Nam H. Chu, Diep N. Nguyen, Dinh Thai Hoang, and Eryk Dutkiewicz are with the School of Electrical and Data Engineering, University of Technology Sydney, Australia (e-mails: namhoai.chu@student.uts.edu.au, diep.nguyen@uts.edu.au, hoang.dinh@uts.edu.au, and eryk.dutkiewicz@uts.edu.au).Quoc-Viet Pham is with the Korean Southeast Center for the 4th Industrial Revolution Leader Education, Pusan National University, Busan 46241, Korea (e-mail: vietpq@pusan.ac.kr).Khoa T. Phan is with School of Engineering and Mathematical Sciences, Department of Computer Science and Information Technology, La Trobe University, Melbourne, Australia (e-mail: K.Phan@latrobe.edu.au).Won-Joo Hwang is with the Department of Biomedical Convergence Engineering, Pusan National University, Yangsan 50612, Korea (e-mail: wjhwang@pusan.ac.kr).

Abstract

Integrated Communications and Sensing (ICS) has recently emerged as an enabling technology for ubiquitous sensing and IoT applications. For ICS application to Autonomous Vehicles (AVs),
optimizing the waveform structure is one of the most challenging tasks due to strong influences between sensing and data communication functions.
Specifically, the preamble of a data communication frame is typically leveraged for the sensing function.
As such, the higher number of preambles in a Coherent Processing Interval (CPI) is, the greater sensing task’s performance is.
In contrast, communication efficiency is inversely proportional to the number of preambles.
Moreover, surrounding radio environments are usually dynamic with high uncertainties due to their high mobility, making the ICS’s waveform optimization problem even more challenging.
To that end, this paper develops a novel ICS framework established on the Markov decision process and recent advanced techniques in deep reinforcement learning.
By doing so, without requiring complete knowledge of the surrounding environment in advance, the ICS-AV can adaptively optimize its waveform structure (i.e., number of frames in the CPI) to maximize sensing and data communication performance under the surrounding environment’s dynamic and uncertainty. Extensive simulations show that our proposed approach can improve the joint communication and sensing performance up to 46.26% compared with other baseline methods.

Index Terms:

Autonomous vehicles, Internet of Vehicles (IoV) joint communication and sensing, MDP, deep reinforcement learning, waveform structure optimization.

I

Introduction

In Autonomous Vehicles (AVs), e.g., self-driving cars and unmanned aerial vehicles, sensing and data communications are two important functions.
The sensing function enables AVs to detect objects around them and estimate their distance and velocity for safety management (e.g., collision avoidance).
The data communication function allows AVs to exchange information with other AVs or infrastructure via Internet of vehicles (IoV).
For example, they can send/receive safety messages and even their own raw sensing data (e.g., traffic data around the AV) for applications such as transportation safety, transportation monitoring, and user services distributed to the AVs  [ 1 ] .
Although automotive sensing and vehicular communication can share many commonalities (e.g., signal processing algorithms and the system architecture  [ 2 ] ) they are typically designed and implemented separately.
As such, communication and sensing functions require separate hardware components operating at different frequency bands that become increasingly expensive and inefficiency because of an ever-growing number of connected devices and services.
Consequently, this makes the implementation of communication and sensing functions in AVs more costly in hardware, complexity, and radio spectrum resources.

These challenges can be effectively addressed by combining both communication and sensing functions into an unified system, called Integrated Communications and Sensing (ICS).
This system can utilize an existing communication waveform, such as vehicular communication waveforms, e.g., Cooperative Intelligent Transport Systems (C-ITS)  [ 3 ]  and Dedicated Short-Range Communication (DSRC)  [ 4 ] , WiFi waveform, or cellular waveform, to extract sensing information from targets’ echoes.
Note that there is another type of integrated communication and sensing system, where radar waveforms, e.g., Frequency-Modulated Continuous-Wave (FMCW) waveform, can be used to transfer data  [ 5 ] .
However, it cannot provide a high data rate as required by AVs because the communication signal has to be spread to avoid degrading the sensing performance  [ 6 ] .
By sharing the same hardware and signals, ICS significantly reduces the power consumption, cost, spectrum usage, and system size compared to conventional approaches where sensing and communication functions are implemented separately, making it more applicable to AVs.
Hereinafter, an AV with ICS capability is named ICS-AV.

Currently, two standards operating at

5.9

5.9

5.9

GHz for vehicular communication networks are C-ITS based on IEEE 802.11bd in Europe  [ 3 ]  and DSRC based on IEEE 802.11p in the U.S.  [ 4 ] .
Unfortunately, their data rates (i.e., up to 27 Mbps) do not meet the requirements of AVs’ applications.
For example, precise navigation that needs to download a high definition three-dimension map and raw sensor data exchange between AVs to support fully automated driving may require connections up to a few Gbps  [ 7 ] .
In addition, the performance of communication and sensing in ICS systems operating at sub-6 GHz is limited due to the bandwidth availability  [ 6 ] . In this context, millimeter wave (mmWave), whose frequency is from

30

30

30

GHz to

300

300

300

GHz, has been emerging as a promising solution to address the above challenges in ICS systems  [ 8 ] .
First, owing to the high-resolution sensing and small antenna size, mmWave is predominantly utilized for automotive Long-Range Radar (LRR)  [ 9 ] . Second, an mmWave system, e.g., a wireless local area network (WLAN) operating at the

60

60

60

GHz band, can provide a very high data rate to meet AVs’ intensive communication requirements.

However, several challenges are hindering the applications of mmWave ICS systems in AVs.
In particular, unlike the conventional approaches where sensing and communication are separated, the ICS-AV leverages a single waveform for both sensing and communication functions.
Thus, it needs to jointly optimize these two functions simultaneously to achieve high performance of data communication and sensing for ICS systems.
In addition, since the ICS operates while ICS-AVs are moving, the surrounding environments of AVs are highly dynamic and uncertain.
This makes ICS-AVs’ performance unstable as mmWave is more severely impacted by wireless environments than those of the sub-6 GHz bands  [ 10 ] .
Therefore, the highly dynamic and uncertainty of mmWave ICS’s environment is another critical challenge that needs to be addressed.
To that end, mmWave ICS systems have been demanding effective and flexible solutions that can not only jointly optimize communication and sensing functions but also adaptively handle the highly dynamic and uncertainty of the surrounding environment to best sustain high data rate communication links (given the highly directional mmWave communications) and sensing accuracy, e.g., low target miss-detection probability and estimation error of target’s range and velocity.

A few works in the literature have recently studied communication mmWave waveforms for ICS systems  [ 11 ,  12 ,  13 ,  14 ,  15 ,  6 ] .
In  [ 11 ]  and  [ 12 ] , the authors exploit a