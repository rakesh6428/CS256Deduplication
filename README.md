# CS256Deduplication
Unsupervised Smarter Deduplication

Deduplication is a concept of logically storing a single instance of data by eliminating the redundant copies thereby optimizing the storage capacity. 
It has gained popularity in large-scale storage systems where digital data growth is exponential. 
Data integration from multiple platforms of an enterprise is a challenging task due to difficulties in record linkage and duplicate detection. 
The duplicates are identified by secure hash signatures called fingerprints that are proven to be efficient for computation compared to traditional data compression techniques. 
It is a new area of research and there are only a few solutions to effectively handle duplicates in data integration. 
This paper focuses on source-level deduplication that identifies duplicates before storing on to a storage device. 
Each record containing various attributes are hashed using the rolling hash. 
The hash values are later clustered using KMeans and Expectation Maximization(EM) clustering technique. 
The duplicates and the similar data end up in the same cluster thereby reducing the number of line items that are to be compared for finding a duplicate. 
To further improve the performance, the divide and conquer approach is used for intra-cluster comparison. 
The experimental results show that the proposed approach is capable of detecting duplicates with higher accuracy than existing solutions. 
The outcome of this solution will serve as a guideline for designing large-scale data integration and indicates that deduplication will benefit the enterprise and storage providers. 
This paper discusses the current trends in data deduplication and provides a synopsis of the new solution and future direction in handling duplicates in large-scale data integration. 

Keywords— Data deduplication, large-scale data integration, Clustering, KMeans, Expectation Maximization, Hashing, space efficiency
