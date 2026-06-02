# EV001: A novel FOD detection framework based on attention mechanism in real-time airport environment

URL: https://www.semanticscholar.org/paper/0008e4f527e96641911ec8bc855632950a358828
Year: 2025
Source: semantic_scholar
Arxiv: n/a

## Abstract

Foreign Object Debris (FOD) on airport runways may cause aircraft damage and endanger the passengers' lives. The current research methods for FOD detection have the following shortcomings: (1) the FOD dataset used is relatively outdated and not collected in the real-time airport environment. (2)The algorithm adopted cannot simultaneously meet the requirements of speed and accuracy, resulting in the inability to prevent aviation disasters efficiently in a timely manner. The novel solution proposed in this paper includes three appealing properties:(1)Use the unmanned aerial system to regularly fly on the airport runway and capture FOD videos.(2) Apply video processing tools and python scripts to extract FOD images from videos and convert them into the FOD standard dataset format, constitute the real-time FOD dataset.(3)Employ the RetinaNet network that can balance the speed and accuracy of FOD recognition, and integrate the Convolutional Block Attention Module(CBAM) into RetinaNet to train and detect the real-time FOD dataset. Compare the two schemes between CBAM-RetinaNet and RetinaNet without CBAM when training the FOD dataset. The experimental results show that the application of attention mechanism enhances the mAP of FOD detection by 2.1% on FOD-A dataset and raises the mAP of identifying FOD by 4.8% on real-time FOD dataset we created. Some other network models are compared with CBAM-RetinaNet. The experiment results demonstrate CBAM-RetinaNet can outperform other network models. Finally, the author discusses the issues involved in the research process and puts forward new research prospects for the future. We provide the github linkage for relevant python script files and part private dataset.
