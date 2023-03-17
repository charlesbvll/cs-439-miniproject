# Exploring optimization and regularization methods in Federated Learning
Project for the CS-439 course (Optimization for Machine Learning).

## Plan of action
- [x] Get familiar with [Flower](https://github.com/adap/flower) 

- [x] Centralized setting

  - [x] Implement PyTorch data loading
  
  - [x] Implement PyTorch training
  
  - [x] Implement PyTorch testing

- [x] Distributed setting

  - [x] Implement Flower Client 

  - [x] Write data partitionning methods
  
  - [x] Distribute data amongst clients
  
- [x] Write plotting functions
 
  - [x] Distributed setting
  
  - [x] Centralized setting
  
- [x] Use parser to set parameters

- [x] Use config manager

- [x] Make optimizer modular

- [ ] Write custom optimizer

- [ ] Play around with different methods

## Install

`Poetry` is recommended for installing the dependencies: `poetry install`.

But a `requirements.txt` file is also present in order to install the dependencies with `pip`: `pip install -r requirements.txt`.

## Centralized setting

To run the MNIST digit recognition task in a centralized setting, the following command can be used:

```sh
python centralized.py
```

## Distributed setting

To run MNIST digit recognition task in a distributed setting, the following command can be used:

```sh
python distributed.py
```

## Changing parameters

The `--help` flag can be used to display the different parameters that can be changed.

