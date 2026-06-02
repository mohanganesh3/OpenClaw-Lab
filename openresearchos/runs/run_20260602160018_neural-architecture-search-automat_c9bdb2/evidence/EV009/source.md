# EV009: An M-Nary SAR Image Change Detection Based on GAN Architecture Search

URL: https://www.semanticscholar.org/paper/0142fd05b5b128adcef806ef9351af35f076e06e
Year: 2023
Source: semantic_scholar
Arxiv: n/a

## Abstract

Change detection (CD) in synthetic aperture radar (SAR) images aims to detect changed areas by considering the changes in backscattering coefficients. However, the changes can be further divided into positive and negative changes in terms of the increase or decrease of backscattering coefficient, so the CD task can be divided into binary and ternary according to the number of existent categories. This article introduces an M-nary (binary or ternary) SAR CD procedure based on the generative adversarial network (GAN) and neural architecture search (NAS) strategy to detect which changes exist in the SAR image pair and design specialized classifiers for both binary CD (BCD) and ternary CD (TCD). First, a difference image (DI) generation approach based on the salient changed region extraction and neighborhood information is designed for a robust difference representation on the M-nary CD. Due to the further subdivision of changes, the insufficiency of labeled data presents the M-nary CD with a dilemma. Concerning the lack of labeled information, this article presents a labeled sample generation strategy based on the GAN architecture search to supplement sample data. Since GAN training is inherently unstable, NAS provides an effective means of searching GAN architecture automatically and ameliorates the reliability of generated samples. During the architecture search procedure, a double-phase evolutionary search strategy is introduced to further improve the stability of GAN training. The experimental results with theoretical analysis prove the validity, robustness, and potential of our method in synthetic as well as real SAR datasets.
