# EV003: Spatio-temporal Cokriging method for assimilating and downscaling multi-scale remote sensing data

URL: https://www.semanticscholar.org/paper/0013193315c7ffdbd43f7cd1dcea1f79059aaa9c
Year: 2021
Source: semantic_scholar
Arxiv: n/a

## Abstract

Abstract No single satellite remote sensing system is able to provide the observations on the Earth's surface at both high spatial and high temporal resolution due to the general trade-off between orbit revisit frequency and satellite sensor's spatial resolution. This paper presents a spatio-temporal Cokriging (ST-Cokriging) method for assimilating remote sensing data sets acquired by multiple remote sensing systems with different temporal sampling frequencies and different spatial resolutions. By extending the traditional Cokriging technique from a sole spatial domain to a spatio-temporal domain, we derived and implemented ST-Cokriging algorithm that explicitly takes the spatial covariance, temporal covariance and spatio-temporal covariance structures within and between different data sets into account. Compared with previous downscaling methods, such as, STARFM and FSDAF, our ST-Cokriging method produces more accurate and reliable assimilation results for the heterogeneous region, with associated uncertainty estimates. This method has been implemented into a software package using Python language within ArcGIS environment. The advantages and effectiveness of our ST-Cokriging method have been demonstrated through an application example, in which MODIS images (daily, 250 m and 500 m spatial resolution) and Landsat TM/ETM+ images (16 days revisit cycle, 30 m) are assimilated to generate daily spectral bands and NDVI images at 30 m spatial resolution. Our validation and accuracy assessments indicate that our ST-Cokriging method can effectively fill in data gaps due to clouds and generate reliable assimilation results and uncertainty estimates at both high spatial resolution and high temporal frequency
