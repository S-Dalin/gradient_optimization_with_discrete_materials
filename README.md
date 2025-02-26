# Notebook on Gradient-based Inverse Design
Jupyter notebook guide for gradient-based inverse design techniques, focusing on applications in nanophotonics using Python and TensorFlow. 
This notebook includes step-by-step tutorials on setting up the computational environment, implementing gradient-based optimization algorithms, and applying these techniques to design photonic structures with desired optical properties.

## Table of Contents
- [About the Project](#about-the-project)
- [Installation](#installation)
- [Usage](#usage)
- [Files and Directory Structure](#files-and-directory-structure)


## About the Project
This project uses a gradient-based optimization approach for inverse design with discrete materials, allowing to achieve target optical properties by adjusting latent space. This process integrates forward model and WGAN-GP to allow the gradient-based optimization. 

The demonstration includes the data generation, data preprocessing, forward modeling, Re-parametrization of discrete material using WGAN-GP. And combined those pre-trained model to enabled the gradient optimization. 

The optimization process is to maximize the Qfwd while minimize the Qback at desired wavelength. 

## Installation
1. Clone the repository:
   ```bash
   $ git clone https://github.com/S-Dalin/guide_gradient-based_inversedesign.git
   $ cd guide_gradient-based_inversedesign
   ```

2. Create virtual environment in the folder `guide_gradient-based_inversedesign`: 
   ```bash
   python -m venv projectname
   ```

3. Activate the virtual environment: 
   ```bash 
   source projectname/bin/activate
   ```

4. Install ipykernel using `pip`:
   ```bash 
   pip install ipykernel
   ipython kernel install --user --name=projectname #install new kernel
   pip install -r requirements.txt                  #install independencies 
   ```


## Usage
1. **Data generation**:
   Run the data generation script to create the core-shell geometries and their optical properties (Qfwd and Qback). 
   It will then save the data in folder `datasets/`
   ```bash
   03a_dataset_generation_modPW.ipynb
   ```

2. **Data Preprocessing**:
   Run the data preprocessing script to prepare input data. 
   ```bash
   03b_preprocessing_mod_PW.ipynb
   ```

3. **Forward Modeling (Resnet model) and Model Evaluation**:
   Train the forward model 
   ```bash
   03c_forward_resnet_model_modPW.ipynb
   03d_evaluate_forward_modPW.ipynb
   ```

4. **Reparemetrized Discrete Material (WGAN-GP model) and Model Evalaution**:
   Train the WGAN-GP model 
   ```bash
   04a_wgangp_wideDataset.ipynb
   04b_evaluate_wgangp.ipynb
   ```

5. **Gradient enabled Inverse Design - On test-set Sample**:
   Run the below script to find the inverse design of the entire spectrum on test-set. It is also give the inverse design statistics.
   ```bash
   05a_inverse_gradient.ipynb
   05b_statistics_inverse_gradient.ipynb
   ```

6. **Gradient enabled Inverse Design**:
   Run the below script to maximize the Qfwd while minimize the Qback (or zero back-scattering) at the desired wavelength.
   ```bash
   05c_inverse_gradient_maximize_minimize.ipynb
   ```

7. **Global enabled Inverse Design**:
   Run the below script to find the inverse design of the entire spectrum on test-set using global opt approach.
   ```bash
   06a_inverse_global.ipynb
   06b_statistics_inverse_global.ipynb
   ```

8. **Other py file**:
   Check performance metrics, visualize the results, and WGAN-GP model architecture
   ```bash
   python evaluation_protocol.py 
   python utils_plot.py
   python wgangp_model.py

## Directory Structure
- `datasets/` - Directory where generateds datasets are stored.
- `materials/` - Contains material property files for simulation.
- `save_figures/` - Directory where output figures and plots are saved.
- `best_geometries/` - folder that save the best geometries from optimization (gradient-based optimization).  
- `fwd_predicted/` - folder that save the predicted value into dataframe from forward neural network model.
- `models/` - folder that save the models from (F-NN, G-NNs)  
- `result_dict/` - folder that save the runtime from the inverse design optimization on the test-set.



