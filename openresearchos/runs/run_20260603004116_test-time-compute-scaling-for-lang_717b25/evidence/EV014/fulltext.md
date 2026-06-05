[2602.02725] Automated Dysphagia Screening Using Noninvasive Neck Acoustic Sensing

Automated Dysphagia Screening Using Noninvasive Neck Acoustic Sensing

Abstract

Pharyngeal health plays a vital role in essential human functions such as breathing, swallowing, and vocalization. Early detection of swallowing abnormalities, also known as dysphagia, is crucial for timely intervention. However, current diagnostic methods often rely on radiographic imaging or invasive procedures.
In this study, we propose an automated framework for detecting dysphagia using portable and noninvasive acoustic sensing coupled with applied machine learning. By capturing subtle acoustic signals from the neck during swallowing tasks, we aim to identify patterns associated with abnormal physiological conditions.
Our approach achieves promising test-time abnormality detection performance, with an AUC-ROC of 0.904 under 5 independent train–test splits.
This work demonstrates the feasibility of using noninvasive acoustic sensing as a practical and scalable tool for pharyngeal health monitoring.

Index Terms —

Applied Machine Learning, Signal Processing, Pharyngeal Health, Digital Health.

Fig. 1 :

Overview of our proposed automated system. (A) Data collected from participants during this study. (B) Demonstration of our data annotation process. (C) Modeling procedure to explore the relationship between acoustic signal and the swallow abnormalities. (D) Presentation of the empirical results.

1  Introduction

Dysphagia, or difficulty swallowing, continues to pose a significant public health concern, affecting 10-20% of adults over the age of 50 and up to two-thirds of patients with conditions such as Parkinson’s disease, stroke, or head and neck cancer  [ 12 ,  10 ] . It is estimated that around $4.3 to $7.1 billion go towards hospitalization costs associated with dysphagia annually  [ 9 ] . Currently, swallow evaluations are performed through video-based modalities such as flexible endoscopic evaluation of swallowing (FEES), videofluoroscopic swallowing studies (VFSS) or high-resolution manometry, which, while informative, are either invasive, require radiation exposure, necessitate a trained practitioner, or are limited in cost efficiency  [ 11 ,  16 ] . Given the limitations of current gold-standard swallow evaluations, hospitalized patients at risk for aspiration are typically screened using a clinical swallow evaluation. However, these assessments are performed without instrumentation, demonstrate limited diagnostic sensitivity and specificity, and often leave providers with uncertainty in clinical decision-making. This highlights the pressing need for an objective swallow assessment tool capable of efficiently and accurately identifying dysphagia risk.

To address this gap, we collected audiometric data during fiberoptic endoscopic evaluation of swallowing (FEES) to investigate associations between swallow dysfunction and acoustic features. We then used these data to develop a machine learning model for predicting swallow dysfunction. We present a system that achieved promising classification performance, suggesting that surface digital auscultation captures acoustic signatures with diagnostic value reflective of underlying oropharyngeal dysfunction.

2  Related Work

To address these challenges imposed by dysphagia assessments, a growing body of research has also explored the use of machine learning techniques to analyze swallow mechanisms and detect related abnormalities using a mix of accelometry, acoustic signals, and electromyography (EMG)  [ 7 ,  13 ,  14 ,  6 ] . While preliminary studies suggest the potential of machine learning for detecting swallow dysfunction, they remain significantly limited. First, and most notably none have conducted real-time assessments of swallow sounds during a gold-standard evaluation such as FEES, restricting their ability to characterize the severity and nature of dysfunction. Second, many studies validated their models using only a single bolus consistency (e.g., a sip of water)  [ 7 ,  13 ,  14 ,  6 ] , despite the fact that dysphagia and aspiration may manifest differently across consistencies- thereby limiting external validity. Finally, most prior work has been constrained to proof-of-concept designs with small sample sizes  [ 4 ] . To overcome these limitations, we developed a machine learning model trained on more than 600 swallow events, using acoustic data collected during comprehensive swallow evaluations.

3  Method

3.1  Preliminary Experiments

Initial preliminary experiments were conducted to explore the performance of established baseline audio embedding models: AST, CLAP and OPERA  [ 3 ,  2 ,  19 ]  (Figure

1

section D.1.). This was done with our existing dataset, both with and without demographic features across 5 randomized train–test splits.
“Sev.” and “Abn.” standing for severity (3-class) and abnormality (2-class) respectively.
It is observed that the OPERA model achieve the best performance, hence it was included in our main experiments as a baseline comparison.

Although age and gender did not have substantial improvement to the model’s performance, we decided to include it due to prior research indicating that age and gender may change the acoustic characteristics of the swallow [ 17 ] . Furthermore, the performance between RFC and SVM was comparable, but for consistency we chose to use RFC for our main evaluation.

Table 1 :

Background Distribution of Participants

Factors

Sample Size (n)

Percentage (%)

Gender

Female

25

51.02

Male

24

48.98

PAS

Normal

PAS[1-2]

22

44.90

Abnormal

PAS[3-5]

20

40.82

PAS[6-8]

7

14.29

Total

27

55.10

BMI

&lt;  18.5

&lt;18.5

(underweight)

2

4.08

≤  24.9

\leq 24.9

(normal)

17

34.69

&gt;  24.9

&gt;24.9

(overweight)

30

61.22

Age

30  30

–

60  60

13

26.53

62  62

–

69  69

16

32.65

70  70

–

96  96

20

0.41

Chronic Obstructive Pulmonary Disease (COPD)

No

41

83.67

Yes

8

16.33

History of Pneumonia caused by Aspiration (PNA)

No

38

77.55

Yes

11

22.45

3.2  Data Annotation

This study was approved by University of California San Diego, where we recruited a total of 49 participants from Center for Airway, Voice and Swallowing, who self-reported symptoms of dysphagia. These participants underwent swallow evaluations using FEES. A standard FEES includes 8-10 trials of oral intake with different consistencies and bolus sizes. During this, a flexible
laryngoscope is placed through the nose and each swallow trial is observed. Speech and language pathologists and fellowship-trained laryngologists rated each FEES using the penetration-aspiration scale (PAS) to assess the severity of swallow dysfunction. The PAS is a categorical assessment rating, ranging from 1–8, assessing the degree of dysfunction and evidence of oral
intake invading the airway with higher scores indicating higher levels of swallow dysfunction. Based on this scale, PAS of 1-2 are
relatively normal swallows, PAS of 3-5 indicate evidence of penetration, and PAS of 6 and above refers to evidence of aspiration. During the FEES, each participant also had a digital stethoscope (3M TM Littmann Core Digital Stethoscope) placed lateral to the thyroid cartilage to collect audiometric data in real time during the FEES.

In total, 392 audio recordings were collected for this study. Corrupt audio files that contained talking or non-swallow sounds that our parameters could not filter out were removed. The swallows in each recording were then cleaned in preparation for feature extraction with an average duration of 0.64 seconds. After processing, there were 617 individual swallow events (24 participants contributed to 10-15
swallow events each, 10 participants contributed to 15-20 events each, 8 participants contributed

≤  \leq

10 events each and 3 participants contributed

≥  \geq

20 events each).
We aimed to use our domain-informed features compared with features extracted f