# Training quantum Boltzmann machines using extreme rates of unit dropout 

Ilmo Salmenpera,
University of Helsinki



Repository for my online poster for FCAI Days 2021 about using Unit dropout method to train Quantum Boltzmann Machines.

## Abstract

Quantum annealing is a form of quantum computing that has wide applicability in many realms, like quantum chemistry, logistics or machine learning. One of these applications is to use the quantum annealing device as a quantum sampler for sampling from the model distribution of a common machine learning model called Boltzmann Machines. This has been shown to be a quite promising way of applying quantum computing to machine learning in practice, outperforming the current classical algorithms for performing these sampling tasks.
While these devices tend to be large in comparison to universal quantum computers, they still lacking in size to be used in practical machine learning tasks. This calls for clever strategies to mitigate these issues, as most actual machine learning tasks require large layer sizes to perform well. Unit dropout method is one candidate for alleviating these issues. This model agnostic technique was originally developed for regulating weights of machine learning models, but it can also be used to reduce the effective overall size of the layers during training.
We tested the effects of extreme rates of unit dropout in the process of pretraining multiple restricted Boltzmann machines to form deep belief network and determined what sort of constraints do the results infer to quantum hardware they would be computed on. While the optimal dropout rate seems to be around 50%, which is supported by existing research, more extreme rates of dropout can give further benefits for quantum machine learning, as they allow for larger layer sizes to be used during training. Even the model with dropout rate of 92% managed to learn some representation of the underlying model distribution, which is important, as this is the model that could be feasibly computed using existing quantum annealing devices. 

## Introduction

Text

## Restricted Boltzmann machines (RBM)

RBMs are a type of stochastic neural networks, characterized by visible and hidden layers of units that are connected to each other symmetrically [1]. While they aren’t that useful by themselves, they are integral to many widely used machine learning models like deep belief networks [2]. Training RBMs require evaluation of a model distribution of the system, which is computationally intractable when done analytically, but feasible using sampling methods.

<p align="center">
  <figure>
    <img src="https://github.com/Ilmosal/FCAIDays2021Poster/blob/main/pictures/bm.png" alt="Restricted Boltzmann Machine"/>
    <figcaption><b>1.1 This is my caption text.</b></figcaption>
  </figure>
</p>

## Quantum annealing and sampling

Quantum annealing is a special type of quantum computation, where qubits are connected to each other by tweakable couplings, as opposed to the gate model for quantum computation [3]. These devices are capable of solving the QUBO problems by finding the minimum of the hamiltonian energy function of the system. They can also be used for efficiently sampling from the boltzmann distribution of the characterized problem, allowing them to be used to estimate the model distribution of a RBM [4]. While these devices generally have larger qubit counts than universal quantum computers, embedding these problems still restrict the layer sizes of RBMs to unconventionally small widths.

## Unit Dropout Method

Unit Dropout method is a common weight regularization method, where units from the model are “dropped” stochastically with a probability of p, for the duration of a single batch [5]. While the most optimal choice for this p is around 0.5, this value can be pushed to be even larger, causing the computed layer sizes to be scaled to (1 – p) of their original size. This is very convenient when using quantum annealing to estimate the model distribution of the RBM, as it allows the layer sizes to be increased without needing to increase the device size. It also allows the problem to be parallelized for multiple quantum annealers.

## Results

The effect of extreme values of p were tested on the MNIST dataset [6] with a custom RBM implementation written in python [7]. The results show that while the most optimal value for p is around 0.5 as stated in the litiature, the value can be pushed further without large performance issues on the prediction rate, allowing for layer sizes eight times larger than would normally be possible. The model can with p = 0.92, allowing contemporary quantum annealing devices to be used for training a fully connected RBM on the full MNIST dataset, even if the prediction rate suffers greatly as result.

## Implicit Labeling for RBMs

Attaching to labels to RBMs usually requires an additional layer of labeling units to be attached to the hidden layer of the RBM. While it is possible to do this with quantum annealing, it can be quite unefficient, as there are very few qubits to work with. More efficient is to add implicit label influence to the biases of the hidden units for the duration of sampling and then sample the states of the labeling units afterwards from the states of the hidden units. This does cause only a minor performance hit to the model and it can be very useful for the sake of developing new sampling techniques for the QRBMs, as most of the time labeling is used to only evaluate the general performance of the model.

### References
