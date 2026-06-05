[2512.17975] CoBiTS: Single-detector discrimination of binary black hole signals from glitches using deep learning

CoBiTS : Single-detector discrimination of binary black hole signals from glitches using deep learning

Matthew VanDyke, Kexuan Wu, and Sukanta Bose

Department of Physics &amp; Astronomy, Washington State University, 1245 Webster, Pullman, WA 99164-2814, USA

Abstract

We develop a Conformer neural network, called Conformer Binary
neTwork Search, or  CoBiTS , for distinguishing binary black
hole gravitational wave (GW) signals from non-Gaussian and
non-stationary noise artifacts in the data from current generation
LIGO-Virgo-KAGRA detectors. A large subset of these transient noise
artifacts, termed as “glitches” for short, trigger BBH search
templates. Some of them go on to produce detection candidates
and require human vetting, supported by data quality tools, to be
correctly identified and vetoed. In its current version,  CoBiTS  takes
as inputs single-detector strain timeseries snippets, claimed by other search
pipelines to be containing GW candidates, and outputs the
significance of each snippet to contain a BBH signal and a glitch.
 CoBiTS  is shown to be particularly effective in discriminating
high-mass BBH signals from blips and scattered light glitches, even
when a signal is near concurrent or overlapping with a glitch.
The performance of  CoBiTS  gains from employing Conformer, which is a
specialized model that combines convolutional layers and Transformer
architecture for sequence modeling tasks. Conformer is especially good at
leveraging the strengths of both convolutional layers – for local
feature extraction – and self-attention layers – for capturing
long-range dependencies.

I  Introduction

The launch of gravitational wave (GW) astronomy has had a stellar
start

GW150914 ;  TheLIGOScientific:2017qsa ;  LIGOScientific:2025slb  ,
with all sources detected so far as binaries involving stellar-mass or intermediate-mass black holes and neutron stars. This work develops solutions that will provide better
quality alerts to external observatories and detectors – even when
only one GW interferometer registers such a binary as a detection candidate.
Since we expect the CBC event rate to be considerably higher in the fifth observation run (O5) than in past runs of current ground based detectors, it assumes greater significance that we step up the GW alert “quality control” or even automate it (see Ref.

Essick:2020qpo

as an example of such an exercise).
This is why we propose here a Machine Learning (ML) algorithm for better and faster discrimination of binary black hole (BBH) signals from noise transients

LIGO:2024kkz

than traditional methods, in data from a single detector. Specifically, we
develop a high-mass
BBH candidate vetting pipeline that:
(a) is computationally faster than matched filtering;
(b) is at least as good as matched filtering, in its volume sensitivity, to BBHs across luminosity distance;
(c) performs better than
matched filtering, especially, when non-stationary or non-Gaussian noise transients, or “glitches”, are overlapping or present within a chirp-time of a BBH signal. This is observed for a
large subset of the non-spinning BBH parameter space.

The reason we focus on improving the vetting of single-detector search candidates is multifold. First, it is the basic unit of a transient search with a network of detectors. Second, since the duty factor is not going to be 100% in the current generation kilometer-scale detectors, improving the ability to detect with single detectors will improve the overall BBH detection rates of an observation run with any network. Third, owing to varied sensitivities and the different orientations of the global detectors,

1

1  1 LIGO Hanford and Livingston detectors have very similar but not exactly the same antenna patterns.

occasionally, signals from the same binary can turn out to be above the detection threshold in one detector but not the others.
Further, even a moderately significant signal in one detector can rise in significance when analyzed in conjunction with data from other detector(s) if they were in observation mode but did not on their own find an above-threshold signal around that time. A fast vetting of such a single-detector candidate can be exploited to launch follow-up multi-detector analyses.
Future versions of search pipelines can integrate our network model’s functions in order to reduce their false-alarm rates and event retractions.
Traditional BBH search pipelines

2016_pycbc_usman ;  aubin_mbta_2021 ;  Sachdev:2019vvd ;  chu2021spiir

have mechanisms for discriminating against the primary glitches

Nitz_2018 ;  Sachdev:2019vvd

with varying degrees of effectiveness. However, since their optimal statistics make simplifying assumptions about the detector noise, which may not be an accurate characterization of the real data, the ML models hold promise for better performance. They can be trained to construct better empirical representations of that data.

Almost all BBH search pipelines in ML to-date that satisfy the characteristics (a) and (b) above, in real strain data, have been shown to do so when employing multi-detector coincidence. Our single-detector improvements are promising and hold out hope for further performance upticks that can contribute to increasing the BBH detection rate. This will help in forming a more accurate picture of BBH demographics, which informs our understanding of their formation channels, population synthesis models, and aids broader pursuits like the measurement of the Hubble constant.
The reason it helps to achieve computational speed-up for transient GW searches is that it enables followup campaigns by other observatories for catching any electromagnetic or particle counterpart signals that may get triggered by their astrophysical sources.
As noted above, an MMA observation of this kind can significantly boost the probability of finding the progenitor’s cosmic coordinates or the true host galaxy. This in turn can reveal any peculiarities that maybe shared by those hosts or stellar-mass BBH environments and, thereby, shed light on their formation scenarios. A host galaxy can also yield redshift information, which along with the GW luminosity distance can provide a measurement of the Hubble constant.
On the other hand, any reductions in the
computational costs of signal searches can help in allocating the freed-up resources for other scientific tasks.
The only
GW signal that is unequivocally accepted to have had electromagnetic counterparts is the binary neutron start (BNS) merger GW170817

TheLIGOScientific:2017qsa  . Arguably, BNSs should be the top target of such ML models. This is what we will pursue in the future but restrict ourselves here to BBHs, which have shorter signals. BBHs have also been proposed as progenitors of a few EM signals and, if true, offer intriguing prospects for probing their host environments

Graham:2020gwr  .

We will target the era of the fifth observation run (O5) of the ground-based GW detectors LIGO

advligo ;  Saleem_2022  , Virgo

Acernese_2015

and KAGRA

kagra  .
We expect several GRB, X-ray, optical and radio observatories (not to mention neutrino detectors) to be active then that can contribute to this MMA campaign. New campaigns, such as the Legacy Survey of Space and Time (LSST) of the Rubin Observatory, may also capitalize on MMA opportunities

LSST:2008ijt  .

In recent years, multiple machine learning (ML) approaches have been proposed
for gravitational-wave (GW) data analysis (see

Cuoco_2021

for a review).
A key motivation is that neural networks can exploit non-linear feature
relationships to distinguish patterns in time-transient data and images.
They are particularly effective in learning complex, non-linear structures and
in separating clusters that are not linearly separable. By contrast,
traditional matched filtering and signal-based consistency tests, such as