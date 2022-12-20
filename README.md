# Learning Invariance Manifold via CPPNs

We present a systematic data-driven approach based on implicit image representations and contrastive learning, that allows the identification and parameterization of the manifold of highly activating stimuli, for visual sensory neurons.

<p align="center">
  <img src="figures/concept.png" />
</p>

We tested our method on simple [Gabor-based model neurons](https://github.com/sinzlab/cppn_for_invariances/tree/main/notebooks/simulated_data) with known (and exact) invariances as well as neural network models predicting the responses of [macaque V1 complex cell neurons](https://github.com/sinzlab/cppn_for_invariances/blob/main/notebooks/macaqueV1/find_invariance.ipynb). Below are the invraince manifold of two example V1 neurons, learned by our method:

<br>

<p align="center">
	    <be>
	    <img src="https://media.giphy.com/media/MWCvCJdr1g4C1AFa6S/giphy.gif" height="200" style="border:5px solid black;margin:0px 150px">
      <img src="https://media.giphy.com/media/1FGpvrMs7ZmrGYCuWy/giphy.gif" height="200" style="border:5px solid black;margin:0px 150px">
    </p>

You can read the full paper [here](https://openreview.net/forum?id=2dQyENiU330).


## Requirements

This project requires that you have the following installed:

- [docker](https://docs.docker.com/get-docker/)
- [docker-compose](https://docs.docker.com/compose/install/)


## Instructions to run the code

1. Clone the repository: `git clone https://github.com/sinzlab/cppn_for_invariances.git`
2. Navigate to the project directory: `cd cppn_for_invariances`
3. Run the following command inside the directory

    ```bash
    docker-compose run -d -p 10101:8888 jupyterlab
    ```
    This will create a docker image followed by a docker container from that image in which we can run the code. 

3. You can now open the [jupyter lab evironment](https://jupyterlab.readthedocs.io/en/stable/#) in your browser via `localhost:10101`


## Issues

If you encounter any problems or have suggestions, please open an [issue]().

## Citing our work
```
@inproceedings{baroni2022learning,
  title={Learning Invariance Manifolds of Visual Sensory Neurons},
  author={Luca Baroni and Mohammad Bashiri and Konstantin F Willeke and Jan Antolik and Fabian H Sinz},
  booktitle={NeurIPS 2022 Workshop on Symmetry and Geometry in Neural Representations},
  year={2022}
}
```