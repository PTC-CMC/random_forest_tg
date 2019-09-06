## Random forest models for predicting the coefficient of friction and adhesion for systems of two contacting functionalized monolayers.

This work is associated with the manuscript 
"Examining Chemistry-Property Relationships in 
Lubricating Monolayer Films through Molecular Dynamics Screening" currently under review,
authored by Andrew Z. Summers, Justin B. Gilmer, Christopher R. Iacovella, 
Peter T. Cummings, and Clare McCabe at Vanderbilt University.

### Installation

Use of this model requires several Python packages to be installed, as well
as data obtained from molecular dynamics screening. Most of the required
packages are located in the `req.txt` file. It is recommended to
use the Anaconda package manager to create a new environment, as the
packages can be pulled from this file directly.

The recommended installation instructions are as follows:

#### Clone this repository

```
>> git clone https://github.com/PTC-CMC/random_forest_tg.git
```

#### Create a new Anaconda environment

**NOTE: This environment is only meant for MacOS, as certain packages like `appnope` are MacOS specific**

`>> conda env create -f environment-macos.yml`

`>> conda activate screening35`

`>> pip install -r requirements-macos.txt`


#### Download data from MD screening

```
git clone https://github.com/PTC-CMC/terminal_group_screening.git
git clone https://github.com/PTC-CMC/terminal_groups_mixed.git
```

#### Install atools-ml package

```
git clone https://github.com/PTC-CMC/atools_ml.git
cd atools_ml
pip install .
cd ..
```

### Using the models
The random forest models can be regenerated in a few seconds.
Thus, rather than providing these models in a form already generated (such
as a serialized form like pickle), the script herein re-creates the models
on the spot.
The script `rf.py` is used to regenerate the models and generate
predictions for user-specified terminal group chemistries. These can be
changed by opening the file and altering the `"SMILES1"` and `"SMILES2"`
variables.
Further instructions can be found inside the `rf.py` file.
