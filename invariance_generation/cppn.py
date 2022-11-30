import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np


class CPPNGenerator(nn.Module):
    def __init__(
        self,
        ndim=2,
        layers=8,
        width=15,
        aux_dim=1,
        out_channels=1,
        positional_encoding_dim=None,
        positional_encoding_projection_scale=1.0,
        nonlinearity=nn.LeakyReLU,
        final_nonlinearity=nn.Tanh,
        bias=False,
        batchnorm=True,
        weights_scale=0.1,
        # final_batchnorm=False,
        # track_running_stats=False,
    ):
        """
        CPPN class producing images from coordinates (and some function of coordinates) + auxiliary inputs
        Args:
            ndim (int, optinal): Defaults to 2.
            layers (int, optional): [description]. Defaults to 8.
            width (int, optional): [description]. Defaults to 20.
            aux_dim (int, optional): [description]. Defaults to 1.
            out_channels (int, optional): [description]. Defaults to 1.
            nonlinearity ([type], optional): [description]. Defaults to Cos.
            final_nonlinearity ([type], optional): [description]. Defaults to nn.Sigmoid.
            bias (bool, optional): whether to use bias term for the linear layers. Defaults to False.
        """

        super().__init__()

        # TODO: make this flexible
        self.ndim = ndim  # number of deterministic input dimensions
        self.aux_dim = aux_dim  # number of auxiliary dimensions

        self.positional_encoding_dim = positional_encoding_dim
        if positional_encoding_dim is not None:
            self.positional_encoding_projection_scale = (
                positional_encoding_projection_scale
            )
            self.register_buffer(
                "B",
                torch.randn(ndim, positional_encoding_dim)
                * positional_encoding_projection_scale,
            )
            self.in_dim = 2 * self.positional_encoding_dim + self.aux_dim
        else:
            self.in_dim = ndim + self.aux_dim
        self.out_channels = out_channels
        self.batchnorm = batchnorm
        self.weights_scale = weights_scale

        # add layers
        n_input = self.in_dim
        elements = []
        for i in range(layers - 1):
            elements.append((f"layer{i}", nn.Linear(n_input, width, bias=bias)))
            if self.batchnorm == True:
                elements.append(
                    (f"batchnorm{i}", nn.BatchNorm1d(width, track_running_stats=False))
                )
            if nonlinearity is not None:
                elements.append((f"nonlinearity{i}", nonlinearity()))
            n_input = width

        # last layer
        elements.append(
            (f"layer{layers-1}", nn.Linear(n_input, out_channels, bias=bias))
        )
        elements.append((f"nonlinearity{layers-1}", final_nonlinearity()))

        self.func = nn.Sequential(OrderedDict(elements))

        self.apply(self.weights_init)

    def weights_init(self, m):
        if "Linear" in m.__class__.__name__:
            m.weight.data.normal_(0.0, self.weights_scale)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def forward(self, aux, out_shape):
        """
        Generates images.
        Args:
            aux (torch.tensor): auxiliary inputs which has a shape of (images_n, aux_dim).
                Therefore, if you want 10 images, for instance, pass a tensor with dimensions (10, aux_dim).
            out_shape (tuple): specify the size (height and width) of the output image(s)
        Returns:
            torch.tensor: images with shape (images_n, channels, height, width)
        """

        device = aux.device

        # number of images to be produced
        n = aux.shape[0]

        # get the coordinate values
        coords = torch.meshgrid(
            [torch.linspace(-1, 1, shape, device=device) for shape in out_shape]
        )

        fixed_inputs = torch.stack(coords, dim=-1)
        fixed_inputs = fixed_inputs.unsqueeze(0).expand(n, *fixed_inputs.shape)

        # positional encoding
        if self.positional_encoding_dim is not None:
            fixed_inputs = torch.cat(
                [torch.sin(fixed_inputs @ self.B), torch.cos(fixed_inputs @ self.B)],
                axis=3,
            )

        aux_inputs = aux.view(n, *([1] * len(out_shape)), self.aux_dim).expand(
            -1, *out_shape, -1
        )  # n_images x h x w x aux_dim

        # concatentate the inputs and pass through the network
        x = torch.cat(
            (fixed_inputs, aux_inputs), dim=-1
        )  # n_images x h x w x (fixed_dim + aux_dim)
        s = x.shape
        x = x.reshape(-1, s[-1])
        y = self.func(x)
        y = y.reshape(*s[:-1], 1)
        y = y.permute(0, -1, *tuple(range(1, self.ndim + 1)))

        return y


class CPPNForInvariances(nn.Module):
    """
    CPPN images generator

    requires inputs of shape: [x, num_invariances]  where x is the number of images per neuron
    creates x * n_neurons images returning an output of shape [x*num_neurons, channels, *img_res]

    """

    def __init__(
        self,
        img_res,
        channels,
        num_neurons=1,
        embedding_dim=0,
        num_invariances=1,
        with_periodic_invariances=True,
        # cppn_generator arguments
        layers=8,
        layers_width=15,
        positional_encoding_dim=None,
        positional_encoding_projection_scale=1.0,
        nonlinearity=nn.LeakyReLU,
        final_nonlinearity=nn.Tanh,
        bias=False,
        batchnorm=False,
        weights_scale=0.1,
        aux_dim_scale=1.0,
        **args,
    ):

        super().__init__()
        self.num_neurons = num_neurons
        self.num_invariances = num_invariances
        self.img_res = img_res
        self.channels = channels
        self.embedding_dim = embedding_dim
        if type(with_periodic_invariances) == bool:
            self.with_periodic_invariances = [
                with_periodic_invariances
            ] * num_invariances
        else:
            if len(with_periodic_invariances) != num_invariances:
                raise ValueError(
                    "Please specify the periodicity for each invariance dim -> len(with_periodic_invariances) should be equal to num_invariances."
                )
            self.with_periodic_invariances = with_periodic_invariances
        num_periodic_invariances = sum(self.with_periodic_invariances)
        num_nonperiodic_invariances = (
            len(self.with_periodic_invariances) - num_periodic_invariances
        )
        self.aux_dim = (
            self.embedding_dim
            + num_nonperiodic_invariances
            + num_periodic_invariances * 2
        )
        self._neuron_embeddings = nn.Parameter(
            torch.rand(self.num_neurons, self.embedding_dim) - 0.5
        )
        self.aux_dim_scale = aux_dim_scale
        self.cppn_generator = CPPNGenerator(
            aux_dim=self.aux_dim,
            layers=layers,
            width=layers_width,
            positional_encoding_dim=positional_encoding_dim,
            positional_encoding_projection_scale=positional_encoding_projection_scale,
            out_channels=channels,
            nonlinearity=nonlinearity,
            final_nonlinearity=final_nonlinearity,
            bias=bias,
            batchnorm=batchnorm,
            weights_scale=weights_scale,
        )

    @property
    def neuron_embeddings(self):
        return self._neuron_embeddings

    def __repr__(self):
        x = (
            f"\nNumber of neurons = {self.num_neurons}"
            + f"\nEmbedding dims = {self.embedding_dim}"
            + f"\nInvariance dims = {self.num_invariances}"
            + f"\nPeriodicity = {self.with_periodic_invariances}"
            + f"\nsingle image shape = {[self.channels, *self.img_res]}\n"
            + super().__repr__()
        )
        return x

    def forward(
        self,
        inv_aux_input,
        img_res=None,
        neuron_embeddings=None,
    ):
        """
        inv_aux_input : [num_imgs_per_neuron, inv_dim]
        neuron_embeddings: [num_neurons, embedding_dim]
        expand [num_neurons, num_images_per_neuron, embedding_dim, inv_dim]
        """
        img_res = img_res if img_res is not None else self.img_res
        neuron_embeddings = (
            neuron_embeddings
            if neuron_embeddings is not None
            else self.neuron_embeddings
        )
        # TODO decide whether it should go here
        inv_aux_input_transformed = []
        for i, periodicity in enumerate(self.with_periodic_invariances):
            if periodicity == True:
                aux_transformed = torch.stack(
                    [torch.sin(inv_aux_input[:, i]), torch.cos(inv_aux_input[:, i])],
                    dim=1,
                )
            else:
                aux_transformed = (inv_aux_input[:, i] / np.pi - 1).unsqueeze(
                    1
                )  # this is assuming that the inv_aux_input values are between 0 and 2*pi
                # aux_transformed = torch.sigmoid(aux_transformed * 6) * 2 - 1.0
            inv_aux_input_transformed.append(aux_transformed)

        inv_aux_input = torch.cat(inv_aux_input_transformed, dim=1)
        aux = torch.einsum(
            "si,ne->nsie", inv_aux_input, neuron_embeddings
        )  # TODO check if opt_einsum.contract is faster

        neuron_embeddings = neuron_embeddings.unsqueeze(1).expand(
            -1,
            inv_aux_input.shape[0],  # neuron embeddings are repeated per each image
            -1,
        )

        inv_aux_input = inv_aux_input.unsqueeze(0).expand(
            neuron_embeddings.shape[0],  # neuron inv_aux_input are repeated per neuron
            -1,
            -1,
        )

        aux = torch.cat([inv_aux_input, neuron_embeddings], -1)
        aux = aux.flatten(start_dim=0, end_dim=1) * self.aux_dim_scale
        images = self.cppn_generator(aux, img_res)
        return images
