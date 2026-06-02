[2504.08817] Exploring utilization of generative AI for research and education in data-driven materials science

Exploring utilization of generative AI for research and education in data-driven materials science

\name Takahiro Misawa a , Ai Koizumi b , Ryo Tamura b,c , and Kazuyoshi Yoshimi a

a Institute for Solid State Physics, University of Tokyo, 5-1-5 Kashiwanoha, Kashiwa, Chiba 277-8581, Japan;  b National Institute for Materials Science, Namiki, Tsukuba-shi, Ibaraki, 305-0044, Japan;
 c Graduate School of Frontier Sciences, University of Tokyo, 5-1-5 Kashiwanoha, Kashiwa, Chiba 277-8581, Japan

Abstract

Generative AI has recently had a profound impact on various fields, including daily life, research, and education.
To explore its efficient utilization in data-driven materials science,
we organized a hackathon—AIMHack2024—in July 2024. In this hackathon, researchers from fields such as materials science, information science, bioinformatics, and condensed matter physics worked together to explore how generative AI can facilitate research and education.
Based on the results of the hackathon, this paper presents topics related to (1) conducting AI-assisted software trials, (2) building AI tutors for software, and (3) developing GUI applications for software.
While generative AI continues to evolve rapidly, this paper provides an early record of its application in data-driven materials science and highlights strategies for integrating AI into research and education.

keywords:

Generative AI; Data-driven materials science; visualization; GUI; development of software packages

1  Introduction

The emergence of generative AI, such as ChatGPT  [ 1 ] , has had a significant impact on various fields, including materials science, where it has been used for tasks such as data analysis, automated literature review, and computational modeling. In July 2024, to explore efficient ways to integrate generative AI into research and education in materials science, specifically in automated data analysis, literature review, and software development, we organized a hackathon, AIMHack2024  [ 2 ] .
Approximately 30 participants, ranging from students to industry researchers, took part in the hackathon. The participants came from diverse backgrounds, including materials science, computational science, bioinformatics, and condensed matter physics, which fostered interdisciplinary collaboration.

The workshop was designed to accommodate both beginners and advanced users. For beginners, the guided demonstrations introduced fundamental steps for using selected software from a curated list on a portal site such as MatDaCs (MaterialsDataCommons)  [ 3 ] .
Through MatDaCs, participants could access and use various computational materials science applications, including machine learning frameworks. Participants were encouraged to perform basic calculations with the selected software, gaining practical experience under the supervision of generative AI.
Advanced users, on the other hand, were given the flexibility to select and independently explore software of their choice, following a structured workflow similar to that used by beginners. This approach allowed them to investigate more advanced features and customize their usage according to their research interests.

The primary emphasis of the hackathon was to enhance participants’ ability to critically evaluate AI-generated content. Participants were encouraged to systematically verify AI-generated responses by cross-referencing them with established literature and their domain knowledge. They also tested the AI-generated code and tutorials by running the recommended software, assessing both functionality and accuracy. This exercise was designed not only to enhance technical proficiency but also to foster critical thinking about the reliability and applicability of AI-generated content.

The collaborative efforts in the hackathon aimed to refine participants’ strategies for using AI tools in their research activities, facilitating AI integration into research workflows and practical scientific applications.
The key results of the hackathon included
the implementation of AI-assisted software trials using ImageJ  [ 4 ]  for automated image analysis, the development of an AI tutor for HYSBO  [ 5 ] , a Bayesian optimization tool, using MyGPTs, and the creation of a graphical user interface (GUI) to enhance the usability of PHYSBO. These results highlight the increasing impact of generative AI in advancing materials science research and education.

This paper is organized as follows. Section

2

presents AI-assisted software trials using ImageJ. Section

3

discusses the development of an AI tutor using MyGPTs for PHYSBO. Section

4

describes the creation of a GUI for PHYSBO with AI support. Finally, Section

5

provides a summary of the findings and discusses future directions for AI in materials science.

2  AI-assisted trial use of software

In this section, we present an example of how ChatGPT can be used to assist in the trial use of software for materials science. During the hackathon, we chose the image analysis software ImageJ  [ 4 ]  from the list of software available on MatDaCs  [ 3 ]  as an example. Since ImageJ can count objects in an image, we decided to explore this feature with it.

Figure 1:  Sample images: (a) Circles without overlap and (c) Circles with overlap. ImageJ analysis results: (b) Counting result for non-overlapping circles and (d) Counting result for overlapping circles.

As an example, we prepared an image with several circles, as shown in Fig. 1(a). We then uploaded this image to ChatGPT and entered the following prompt:

”How can I count the number of circles in this image using ImageJ?”.

ChatGPT generated the following step-by-step instructions

1

1  1 We used ChatGPT 4o and ImageJ 1.54g in this tutorial.

:

1.

Open the Image

•

Launch ImageJ.

•

Select  File → Open...  from the menu and open the target image.

2.

Convert to Grayscale

•

Select  Image → Type → 8-bit  from the menu to convert the image to grayscale.

3.

Adjust the Threshold

•

Select  Image → Adjust → Threshold... .

•

Use the sliders to select only the circles (they should appear in red).

•

Click ”Apply.”

4.

Convert to Binary Image

•

Select  Process → Binary → Make Binary  to create a black-and-white image.

5.

Perform Particle Analysis

•

Select  Analyze → Analyze Particles... .

•

Set an appropriate ”Size” range (e.g.,  100-Infinity  to filter out noise).

•

Check  "Display results" ,  "Show outlines" , and  "Summarize" .

•

Click ”OK” to perform the analysis.

6.

View the Results

•

The results window will display the count of detected circles.

By following these instructions and operating ImageJ, we were able to count the circles, as shown in Fig. 1(b).
Typically, when using such software, one begins by reading manuals and learning how to use it, which can take a considerable amount of time. However, using generative AI in this way, we can significantly reduce the effort required for learning the software, which presents a major advantage. In fact, when we demonstrated this result, many participants were greatly amazed by how convenient generative AI is.

Furthermore, as a more advanced test, we examined whether ImageJ could distinguish overlapping objects. For this test, we prepared an image with several overlapping circles, as shown in Fig. 1(c). We then uploaded this image to ChatGPT and entered the following prompt:

“How can I count individual particles when they overlap, as shown in the attached image?”.

ChatGPT generated the following step-by-step instructions:

1.

Open the Image

•

Launch ImageJ.

•

Select  File → Open...  from the menu and open the target image.

2.

Convert to Grayscale

•

Select  Image → Type → 8-bit  from the menu to convert the image to grayscale.

3.

Adjust the Threshold

•

Select  Image → Adjust → Threshold... .

•

Use the sliders to highlight only the c