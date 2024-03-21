# Communication Efficient Federated Learning


## Requirements
- python     = 3.7
- tensorflow = 2.1.0
- numpy      = 1.17
- bitarray   = 1.2.1

## Running
To run a non-IID MNIST CE-FedAvg simulation with 10 workers, C = 0.5, E = 1, sparsity = 0.6 using the MNIST-2NN model, run 
> python main.py

The main.py file is easily edited to run experiments with the CIFAR dataset and different FL parameters.
