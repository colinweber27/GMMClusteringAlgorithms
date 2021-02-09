# GMMClusteringAlgorithms

A package for implementing Gaussian Mixture Models as a data 
analysis tool in PI-ICR Mass Spectroscopy experiments. It was
first developed in the Fall of 2020 to be used in PI-ICR 
experiments at Argonne National Laboratory (Lemont, IL, U.S.).
At its core is a modified version of the ['mixture' module 
from the package scikit-learn.](https://scikit-learn.org/stable/modules/mixture.html)
The modified version, *sklearn_mixture_piicr*, retains all 
the same components as the 
original version. In addition, it contains two classes with 
restricted fitting algorithms: a GMM fit where the phase 
dimension of the component means is _not_ a parameter, and a
BGM fit where the number of components is _not_ a parameter.
The rest of the package facilitates
quick, intuitive use of the GMM algorithms through the use 
of 4 classes, and visualization methods for debugging. 

#### 1. DataFrame
* This class is responsible for processing the .lmf 
  file and phase shifts. As attributes, it holds the 
  processed data for easy access, as well as any data 
  cuts.
  
#### 2. GaussianMixtureModel
* This class fits Gaussian Mixture Models to the 
  DataFrame object. As parameters, it takes:
  1. Cartesian/Polar coordinates
  2. Number of components to use
  3. Covariance matrix type
  4. Information criterion
* Allows for 'strict' fits, i.e. fits where the number 
  of components is specified.
  
#### 3. BayesianGaussianModel
* Exact same as the GaussianMixtureModel class, but 
  uses the BayesianGaussianModel class from scikit-learn
  instead of the GaussianMixtureModel class.
  
#### 4. PhaseFirstGaussianModel
* Implements a fit where the phase dimension is fit to
    first, followed by a GMM fit to both spatial dimensions
    in which the phase dimension of the component means is
    fixed. This type of fit was found to work especially 
    well with data sets in which there were many species, 
    like the 168Ho data.
    
* Only works with Polar coordinates

Each model class also includes the ability to visualize
results in several ways (clustering results, One-dimensional
histograms, Probability density function) and the ability to
copy fit results to the clipboard for pasting into an Excel
spreadsheet.

### Installation
####  Dependencies
GMMClusteringAlgorithms requires:
* Python (>=3.6)
* scikit-learn (>=0.23.2)
* pandas (>=1.2.0)
* matplotlib (>=3.3.0)
* lmfit (>=1.0.0)
* joblib (>=1.0.0)
* tqdm (>=4.56.0)

#### User Installation
Assuming Python and `pip` have already been installed, decide
whether you want a system-wide or local installation, and 
which Python distribution (e.g. Anaconda) you want to 
install under. Then, open the Command Prompt (for regular 
Python distribution) or the Prompt for another distribution 
(e.g. Anaconda Prompt for Anaconda), and run either:
* `pip install GMMClusteringAlgorithms` for a system-wide 
installation (works for regular Python distributions only),
  **OR**
* `pip install -U GMMClusteringAlgorithms` for a local 
    installation.
  
If you want to install in a virtual environment instead, 
then navigate to the virtual environment's directory, activate 
the virtual environment, and install with the commands above.

### Source code
You can check the latest source code with the command 
`git clone https://github.com/colinweber27/GMMClusteringAlgorithms`

  

