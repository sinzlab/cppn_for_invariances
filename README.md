# Learning Invariance Manifold via CPPNs

We present a systematic data-driven approach based on implicit image representations and contrastive learning, that allows the identification and parameterization of the manifold of highly activating stimuli, for visual sensory neurons.

<p align="center">
  <img src="figures/concept.png" />
</p>

We tested our method on simple [Gabor-based model neurons](https://github.com/sinzlab/cppn_for_invariances/tree/main/notebooks/simulated_data) with known (and exact) invariances as well as neural network models predicting the responses of [macaque V1 complex cell neurons](https://github.com/sinzlab/cppn_for_invariances/blob/main/notebooks/macaqueV1/find_invariance.ipynb).


You can read the full paper [here](https://openreview.net/forum?id=2dQyENiU330).


## Requirements

This project requires that you have the following installed:

- [docker](https://docs.docker.com/get-docker/)
- [docker-compose](https://docs.docker.com/compose/install/)

Before building the docker image from the Dockerfile is necessary to edit the [.env_template](https://github.com/sinzlab/cppn_for_invariances/blob/main/.env_template) file and as suggested

