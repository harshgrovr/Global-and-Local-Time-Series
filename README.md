

# Traffic Prediction using Global, local and global + local time series on PEMS traffic dataset

[Using Local Time Series](https://github.com/harshgrovr/Global-and-Local-Time-Series/tree/main/DeepGLO/Task1)

### Python Scripts
`cd DeepGLO/Task1`

`python local.py`


## Using Just global time series and using Both Local & Global Time Series.

### Requirements

1. The repository assumes that you have Pytorch installed with CUDA support. Please follow the instructions at https://pytorch.org/ to install the correct version for your system. 
2. The other required packages are numpy, scikit-learn, scipy, pandas and matplotlib. Please install these packages before using this package. 

### Python Scripts

`python run_scripts/run_pems.py --normalize False`

## Results

 - Task 1: 'RMSE_local': 15.84

 - Task 2: 'RMSE_global': 7.99688920619827

 - Task 3: 'RMSE-global + local': 5.84417049619019

-  Local + Global Time Series Performed best of all the hypothesis.

