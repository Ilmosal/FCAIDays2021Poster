# Training quantum Boltzmann machines using extreme rates of unit dropout 

Ilmo Salmenpera,
University of Helsinki

## Abstract

Quantum annealing is a form of quantum computing that has wide applicability in many realms, like quantum chemistry, logistics or machine learning. One of these applications is to use the quantum annealing device as a quantum sampler for sampling from the model distribution of a common machine learning model called Boltzmann Machines. This has been shown to be a quite promising way of applying quantum computing to machine learning in practice, outperforming the current classical algorithms for performing these sampling tasks.

While these devices tend to be large in comparison to universal quantum computers, they still lacking in size to be used in practical machine learning tasks. This calls for clever strategies to mitigate these issues, as most actual machine learning tasks require large layer sizes to perform well. Unit dropout method is one candidate for alleviating these issues. This model agnostic technique was originally developed for regulating weights of machine learning models, but it can also be used to reduce the effective overall size of the layers during training.

We tested the effects of extreme rates of unit dropout in the process of pretraining multiple restricted Boltzmann machines to form deep belief network and determined what sort of constraints do the results infer to quantum hardware they would be computed on. While the optimal dropout rate seems to be around 50%, which is supported by existing research, more extreme rates of dropout can give further benefits for quantum machine learning, as they allow for larger layer sizes to be used during training. Even the model with dropout rate of 92% managed to learn some representation of the underlying model distribution, which is important, as this is the model that could be feasibly computed using existing quantum annealing devices. 

## Introduction

Text

## Restricted Boltzmann machines (RBM)

RBMs are a type of stochastic neural networks, characterized by visible and hidden layers of units that are connected to each other symmetrically [1]. While they aren’t that useful by themselves, they are integral to many widely used machine learning models like deep belief networks [2]. Training RBMs require evaluation of a model distribution of the system, which is computationally intractable when done analytically, but feasible using sampling methods.

<figure>
  <img src="https://github.com/Ilmosal/FCAIDays2021Poster/blob/main/pictures/bm.png" alt="Restricted Boltzmann Machine"/>
  <figcaption><b>Figure 1:</b> Picture of the two layers of a RBM. Necessary information will be encoded in the visible layer, from which the states of the units in the hidden layer will be inferred from using a sigmoid fuction. While this image doesn't show labeling units, they can be added separately by including a softmax group of units attached to the hidden layer.</figcaption>
</figure>

<figure>
  <img src="https://github.com/Ilmosal/FCAIDays2021Poster/blob/main/pictures/rbm_kaavat.png" alt="Learning rules for Restricted Boltzmann Machines"/>
  <figcaption><b>Figure 2:</b> Learning rule for the parameters of a RBM. The first equation shows how a single parameter will be updated according to two distinct distributions: the expected data distribution and the expected model distribution. Computing the data distribution is very simple, while computing the model distribution is intractable. This means that the model distribution needs to be estimated using sampling methods.</figcaption>
</figure>

## Quantum annealing and sampling

Quantum annealing is a special type of quantum computation, where qubits are connected to each other by tweakable couplings, as opposed to the gate model for quantum computation [3]. These devices are capable of solving the QUBO problems by finding the minimum of the hamiltonian energy function of the system. They can also be used for efficiently sampling from the boltzmann distribution of the characterized problem, allowing them to be used to estimate the model distribution of a RBM [4]. While these devices generally have larger qubit counts than universal quantum computers, embedding these problems still restrict the layer sizes of RBMs to unconventionally small widths.

<figure>
  <img src="https://github.com/Ilmosal/FCAIDays2021Poster/blob/main/pictures/chimera.png" alt="Chimera Graph"/>
  <figcaption><b>Figure 3:</b> Connectivity graph for a Quantum annealing device with Chimera topology. Each dot represents a qubit and the lines represent the couplings between them. While embedding a RBM into this graph naively is impossible, it can be done by chaining qubits together with strong couplings, allowing effective qubit size of the system to be sacrificed for more connections.</figcaption>
</figure>

<figure>
  <img src="https://github.com/Ilmosal/FCAIDays2021Poster/blob/main/pictures/qubits_to_nmax.png" alt="Qubits to Nmax"/>
  <figcaption><b>Figure 5:</b> Relation between the amount of available qubits in a Chimera topology and the allowed maximum layer size. Two red lines show the theoretical maximum capacity of two existing quantum annealing devices: DWave 2000Q and DWave Advantage. Note that the DWave Advantage uses a different connectivity graph compared to Chimera topology, and it could have more efficient embedding schemes available.</figcaption>
</figure>

## Unit Dropout Method

Unit Dropout method is a common weight regularization method, where units from the model are “dropped” stochastically with a probability of p, for the duration of a single batch [5]. While the most optimal choice for this p is around 0.5, this value can be pushed to be even larger, causing the computed layer sizes to be scaled to (1 – p) of their original size. This is very convenient when using quantum annealing to estimate the model distribution of the RBM, as it allows the layer sizes to be increased without needing to increase the device size. It also allows the problem to be parallelized for multiple quantum annealers.

<figure>
  <img src="https://github.com/Ilmosal/FCAIDays2021Poster/blob/main/pictures/dropout.png" alt="Unit dropout method"/>
  <figcaption><b>Figure 4:</b> Unit Dropout Method. Units will be dropped off from the graph with a probability of p. This value can also be fixed so that the amount of dropped out units remains constant during the dropout process.</figcaption>
</figure>

## Implicit Labeling for RBMs

Attaching to labels to RBMs usually requires an additional layer of labeling units to be attached to the hidden layer of the RBM. While it is possible to do this with quantum annealing, it can be quite unefficient, as there are very few qubits to work with. More efficient is to add implicit label influence to the biases of the hidden units for the duration of sampling and then sample the states of the labeling units afterwards from the states of the hidden units. This does cause only a minor performance hit to the model and it can be very useful for the sake of developing new sampling techniques for the QRBMs, as most of the time labeling is used to only evaluate the general performance of the model.

<figure>
  <img src="https://github.com/Ilmosal/FCAIDays2021Poster/blob/main/pictures/implicit_labeling.png" alt="Implicit Labeling"/>
  <figcaption><b>Figure 5:</b> Three phases of Implicit Labeling: first the an "average" influence of the labels to the hidden units is computed by setting all the labels to 0.5, and inferring their influence to the hidden layer using the sigmoid activation rule and adding these values to the biases of the hidden layer. Then during the sampling phase, the sampling is done without any active influence from the labels. After the sampling the states of the labeling units will be inferred from the hidden units as normal, and these values will be used for the gradient descent process.</figcaption>
</figure>

## Results

The effect of extreme values of p were tested on the MNIST dataset [6] with a custom RBM implementation written in python [7]. The results show that while the most optimal value for p is around 0.5 as stated in the litiature, the value can be pushed further without large performance issues on the prediction rate, allowing for layer sizes eight times larger than would normally be possible. The model can with p = 0.92, allowing contemporary quantum annealing devices to be used for training a fully connected RBM on the full MNIST dataset, even if the prediction rate suffers greatly as result.

### References

[1] Hinton, G.E., & Sejnowski, J. (1983). OPTIMAL PERCEPTUAL INFERENCE.

[2] Hinton, G. E., Osindero, S., & Teh, Y.-W. (07 2006). A Fast Learning Algorithm for Deep Belief Nets. Neural Computation, 18(7), 1527–1554. doi:10.1162/neco.2006.18.7.1527

[3] Hauke, P., Katzgraber, H. G., Lechner, W., Nishimori, H., & Oliver, W. D. (2020). Perspectives of quantum annealing: methods and implementations. Reports on Progress in Physics, 83(5), 054401. doi:10.1088/1361-6633/ab85b8

[4] Benedetti, M., Realpe-Gómez, J., Biswas, R., & Perdomo-Ortiz, A. (2016). Estimation of effective temperatures in quantum annealers for sampling applications: A case study with possible applications in deep learning. Physical Review A, 94(2). doi:10.1103/physreva.94.022308

[5] Hinton, G. E., Srivastava, N., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. R. (2012). Improving neural networks by preventing co-adaptation of feature detectors. arXiv [cs.NE]. Opgehaal van http://arxiv.org/abs/1207.0580

[6] Salmenperä, I. E., (2021). Training Quantum Restricted Boltzmann Machines Using Dropout Method. http://urn.fi/URN:NBN:fi:hulib-202105112138

[7] QDBN project github - https://github.com/Ilmosal/QDBN

