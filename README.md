# Nazgul

This work-in-progress pipeline aims to take hydrodynamical simulation outputs, i.e. galaxy catalogues and particle datasets, locate gravitational lenses, and simulate lensed images by computing the lensing effects of the single particles. The lensing code relies on the lenstronomy framework. 
The name references the fact that this code was first developed on the EAGLE simulation, following a similar approach to the SEAGLE project, and was further adapted to the COLIBRE simulation, while being part of the ELROND project.

## Quickstart

```bash
git clone git@github.com:GiacomoQueirolo/EAGLE_Lensing
cd EAGLE_Lensing
echo "for the following 2 commands you can alternatively use conda instead of mamba"
mamba env create -f nazgul_env.yaml
mamba activate nazgul_env
python -m pip install -e .
cd src/nazgul
echo "If you will be using EAGLE simulated lenses, follow the following script: setup_eagle_data.py"
jupyter notebook Tutorial.ipynb 
```
