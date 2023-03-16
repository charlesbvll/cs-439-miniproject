# Exploring optimization and regularization methods in Federated Learning
Project for the CS-439 course (Optimization for Machine Learning).

## Plan of action
- [ ] Get familiar with [Flower](https://github.com/adap/flower) 

- [x] Centralized setting

  - [x] Implement PyTorch data loading
  
  - [x] Implement PyTorch training
  
  - [x] Implement PyTorch testing

- [ ] Distributed setting

  - [ ] Implement Flower Client 

  - [ ] Write data partitionning methods
  
  - [ ] Distribute data amongst clients
  
- [ ] Write plotting functions

- [ ] Play around with different methods

## Install

`Poetry` is recommended for installing the dependencies: `poetry install`.

But a `requirements.txt` file is also present in order to install the dependencies with `pip`: `pip install -r requirements.txt`.

## Centralized setting

To run the MNIST digit recognition task in a centralized setting, the following command can be used:

```sh
python centralized.py
```

Many of the parameters can be changed inside the `parameters.py` file.



