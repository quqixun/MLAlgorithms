# DBSCAN

This code implements DBSCAN method to cluster data.  
The code is tested under Python 3.6.5 in Ubuntu 18.04.  
It should work in Windows and macOS if required libraries are installed.

# Algorithm

Paper [A density-based algorithm for discovering clusters in large spatial databases with noise](http://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf) proposed DBSCAN.  
In its [Wikipedia page](https://en.wikipedia.org/wiki/DBSCAN), a good pseudocode is availabel.

# Dataset

### Simulation Dataset
Use scikit-learn sample generator to create random examples  
in three types of clusters (**blobs**, **circles**, **moons**).  
See [Sample generators](http://scikit-learn.org/stable/datasets/index.html#sample-generators) for more information.  
In this case, 600 samples are generated, and each sample has 2 features.

# Performance

Change working directory to this folder.  
And make sure that you have activated environment "algo".
```
cd DBSCAN\src
```

### Simulation Dataset

Run the test script on **simulation dataset**.  
```
python test.py
```

Plot original points and the final clustering results.  

For **blobs**,  
<img src="https://github.com/quqixun/MLAlgorithms/blob/master/DBSCAN/images/blobs.png" alt="train" width="700">  
For **circles**,  
<img src="https://github.com/quqixun/MLAlgorithms/blob/master/DBSCAN/images/circles.png" alt="test" width="700">  
For **moons**,  
<img src="https://github.com/quqixun/MLAlgorithms/blob/master/DBSCAN/images/moons.png" alt="test" width="700">
