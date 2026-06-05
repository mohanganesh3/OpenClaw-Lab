# EV009: Optimization of Data Search Process Using a Combination of Merge Sort and Binary Search

URL: https://www.semanticscholar.org/paper/00104786c11ba281010ccf6671df0d6812472edc
Year: 2025
Source: semantic_scholar
Arxiv: n/a

## Abstract

The growth in data volume in the digital age makes speed in processing and searching for information an increasingly important need. This research focuses on optimizing the data search process by comparing two algorithm combination strategies. The first strategy combines recursive Merge Sort with manual Binary Search, while the second strategy uses iterative (bottom-up) Merge Sort paired with the lower bound function from the C++ library. This research employs a quantitative experimental approach by testing execution times on random datasets of 1,000, 10,000, and 100,000 elements. The test results show that bottom-up Merge Sort has faster execution times compared to the recursive version, for example, 71.2 ms versus 107.2 ms for a dataset of 100,000 elements. This speed is achieved because the iterative approach does not require calling recursive functions, which can add to the memory load. For the search process, manual Binary Search proved to be very fast for a one-time search. However, in a repeated search scenario, the lower bound function is more efficient and consistent in execution time. The combination of bottom-up Merge Sort and lower bound becomes the optimal choice for large-scale systems that require a one-time sorting process and repeated searches, such as search engines and massive data filters. This research concludes that selecting an algorithm strategy appropriate for the frequency and needs of the search can improve system efficiency in processing large-scale data.
