## 0. Create Virtual Environment

- [Download Anaconda](https://www.anaconda.com/download/) and [install](https://conda.io/docs/user-guide/install/index.html#regular-installation) it in you computer.
- Download this repo and extract it.
- In terminal (or Anaconda prompt if you use Windows), change the working directory to the repo folder.
```
cd $<repo_folder_path>
```
- Create a virtual environment named "algo" and install all required libraries.
```
conda env create -f environment.yml
```
- Activate the environment in Linux and macOS (remove "source" if you use Windows)  
befor you run the test of this repo.
```
source activate algo
```  

- Quit the environment in Linux and macOS (remove "source" if you use Windows)..
```
source deactivate
```
