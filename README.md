# PyTorch implementation of “Monge-Ampère Flow for Generative Modeling” [arXiv1809.10188](https://arxiv.org/abs/1809.10188)


## How to run the code 

### Density estimation of MNIST
```python
python density_estimation.py -dataset MNIST -hdim 1024 -Nsteps 100 -train -cuda 7
```

### Variational free energy of Ising
```python 
python variational_free_energy.py -L 16 -fe_exact -2.3159198563359373 -train -cuda 7 -hdim 512 -Nsteps 50 -Batchsize 64 -symmetrize
```

### Plots in the paper

- MNIST NLL

```python 
python paper/plot_nll.py -outname nll.pdf 
```

- Gaussianization MNIST

```python
python density_estimation.py -hdim 1024 -Nsteps 100 -epsilon 0.1 -checkpoint data/learn_mnist/Simple_MLP_hdim1024_Batchsize100_lr0.001_Nsteps100_epsilon0.1/epoch-1.chkp -show 
```

- Direct sample Ising
```python
python variational_free_energy.py -hdim 512 -Nsteps 50 -checkpoint data/learn_ot/ising_L16_d2_T2.269185314213022_symmetrize_Simple_MLP_hdim512_Batchsize64_lr0.001_delta0.0_Nsteps50_epsilon0.1/epoch-1.chkp -show  -L 16  -symmetrize 
```

## Exact Ising free energy at critical temperature on $L\times L$ periodic lattices (For details see https://arxiv.org/abs/1802.02840)

| $L$  | Free energy per site |
| :--: | ------------------: |
| 4    | -2.33604476445      |
| 8    | -2.3227349295609376 |
| 16   | -2.3159198563359373 |
| 32   | -2.3140498159960936 |
| 64   |  -2.3135805785878905 |
