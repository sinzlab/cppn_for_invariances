# Learning Invariance Manifold via CPPNs

We present a systematic data-driven approach based on implicit image representations and contrastive learning, that allows the identification and parameterization of the manifold of highly activating stimuli, for visual sensory neurons.

<p align="center">
  <img src="figures/concept.png" />
</p>

You can read the full paper [here]().

## Requirements

This project requires that you have the following installed:

- [docker](https://docs.docker.com/get-docker/)
- [docker-compose](https://docs.docker.com/compose/install/)

## Model

We tested our method on simple Gabor-based model neurons with known (and exact) invariances as well as neural network models predicting the responses of macaque V1 neurons. The Gabor-based models can be readily created and be used to test the method (see [`notebook_name.ipynb`]()). For the ANN first the model needs to be trained on the responses of V1 neurons to natural stimuli. You can train the model from scratch (see [`notebook_name.ipynb`]()) or download the weights for an already trained model from [here]() and skip the training.

## Moneky V1 Data

You can download the data from [here]().

## Example notebooks

- `notebook_name.ipynb`: Learning known invaraince manifolds of model neurons
- `notebook_name.ipynb`: Training the ANN on the nueral responses of Monkey V1 to natural images
- `notebook_name.ipynb`: Identifying complex cells
- `notebook_name.ipynb`: Learning the invariance manifold of Monkey V1 neurons

## Issues

If you encounter any problems or have suggestions, please open an [issue]().

## Citing our work
