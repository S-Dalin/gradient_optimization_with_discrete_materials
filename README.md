# Notebook on Gradient-based Inverse Design
Jupyter notebook guide for gradient-based inverse design techniques, focusing on applications in nanophotonics using Python and TensorFlow. 
This notebook includes step-by-step tutorials on setting up the computational environment, implementing gradient-based optimization algorithms, and applying these techniques to design photonic structures with desired optical properties.

## Table of Contents
- [About the Project](#about-the-project)
- [Installation](#installation)
- [Usage](#usage)
- [Files and Directory Structure](#files-and-directory-structure)


## About the Project
This project uses a gradient-based optimization approach for inverse design with discrete materials, allowing to achieve target optical properties by adjusting latent space `z`. This process integrates forward model and WGAN-GP to allow the gradient-based optimization. 

The demonstration includes the data generation, data preprocessing, forward modeling, Re-parametrization of discrete material using WGAN-GP. And combined those pre-trained model to enabled the gradient-based optimization. 

The optimization include the inverse design on test data and other problem to maximize the Qfwd while minimize the Qback at desired wavelength. 
We also compare gradient-based apporach to the global optimization.  

## Installation
1. Clone the repository:
   ```bash
   $ git clone https://github.com/S-Dalin/gradient_optimization_with_discrete_materials.git
   $ cd gradient_optimization_with_discrete_materials
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
   01a_dataset_generation_modPW.ipynb
   ```

2. **Data Preprocessing**:
   Run the data preprocessing script to prepare input data. 
   ```bash
   01b_preprocessing_mod_PW.ipynb
   ```

3. **Forward Modeling (Resnet model) and Model Evaluation**:
   Train the forward model. 
   ```bash
   02a_forward_resnet_model_modPW.ipynb
   02b_evaluate_forward_modPW.ipynb
   ```

4. **Reparemetrized Discrete Material (WGAN-GP model) and Model Evalaution**:
   Train the WGAN-GP model 
   ```bash
   03a_wgangp_wideDataset.ipynb
   03b_evaluate_wgangp.ipynb
   ```

5. **Gradient enabled Inverse Design - On test-set Sample**:
   Run the below script to find the inverse design of the entire spectrum on test-set. It is also give the inverse design statistics.
   ```bash
   04a_inverse_gradient.ipynb
   04b_statistics_inverse_gradient.ipynb
   ```

6. **Gradient enabled Inverse Design**:
   Run the below script to maximize the Qfwd while minimize the Qback (or zero back-scattering) at the desired wavelength.
   ```bash
   04c_inverse_gradient_maximize_minimize.ipynb
   04d_evaluate_inverse_gradient_maximize_minimize.ipynb
   ```

7. **Global enabled Inverse Design**:
   Run the below script to find the inverse design of the entire spectrum on test-set using global opt approach.
   ```bash
   05a_inverse_global_opt.ipynb
   05b_statistics_inverse_global.ipynb
   ```

8. **Other py file**:
   visualize the results, and WGAN-GP model architecture
   ```bash
   python utils_plot.py
   python wgangp_model.py

## Directory Structure
- `datasets/` - Directory where generated datasets are stored.
- `models/` - folder that save the models from (Forward model and Generator Model)
- `runtime/` - folder that save the optimization runtime (gradient-based optimization and global optimization).
- `best_geometries/` - folder that save the best geometries from optimization (gradient-based optimization and global optimization).



