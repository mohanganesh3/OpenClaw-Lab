# EV001: FlexiCurve: Flexible Piecewise Curves Estimation for Photo Retouching

URL: https://www.semanticscholar.org/paper/00031ea9ba2be7b1e6e20f7aba288e35c1313ec2
Year: 2023
Source: semantic_scholar
Arxiv: n/a

## Abstract

This paper presents a new method, called FlexiCurve, for photo retouching. Unlike most existing methods that perform image-to-image mapping, which requires expensive pixel-wise reconstruction, FlexiCurve takes an input image and estimates global curves to adjust the image. The adjustment curves are specially designed for performing piecewise mapping, taking nonlinear adjustment and differentiability into account. To cope with challenging and diverse properties in real-world photos, FlexiCurve is formulated to produce diverse estimations. The spatial dependencies among these estimations are implicitly modeled by a Transformer structure to improve local retouching of different regions. Thanks to the image-to-curve formulation, FlexiCurve only needs a lightweight network. Our method improves efficiency without compromising the retouching quality and losing details in the original image. The method is also appealing as it is not limited to paired training data, thus it can flexibly learn rich retouching styles from unpaired data. Extensive experiments demonstrate the efficiency, retouching performance, and flexibility of our method quantitatively and qualitatively.
