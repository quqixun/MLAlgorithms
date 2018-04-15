## 0. Create Virtual Environment

You can build a virtual python environment in **Linux**, **macOS** or **Windows** after these steps.

**0.1** [Download Anaconda](https://www.anaconda.com/download/) and [install](https://conda.io/docs/user-guide/install/index.html#regular-installation) it in you computer.  
**NOTE: In Windows, you need to add a new folder path into the system PATH.**  
This folder contains all excutive files.
```
$<Anaconda_folder_path>/Scripts
```

**0.2** Download this repo and extract it.  

**0.3** In terminal (or Anaconda prompt if you use **Windows**), change the working directory to the repo folder.
```
cd $<repo_folder_path>
```

**0.4** Create a virtual environment named "algo" and install all required libraries.  
**NOTE: This may not work in Windows.**
```
conda env create -f environment.yml
```

**0.5** Activate the environment in **Linux** and **macOS** (remove "source" if you use Windows)  
befor you run the test of this repo.
```
source activate algo
```

**0.6** Quit the environment in **Linux** and **macOS** (remove "source" if you use Windows)..
```
source deactivate
```

**0.7** Create environment in **Windows** if the above command (in **0.4**) did not work.
```
conda create --name algo python=3
pip install -r requirement.txt
```
