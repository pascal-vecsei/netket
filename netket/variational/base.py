import abc
from functools import partial
from typing import Any, Optional, Tuple, Callable

import jax
import flax
from flax.core.frozen_dict import FrozenDict

import jax.numpy as jnp

import netket.jax as nkjax
import netket.nn as nknn
from netket.operator import AbstractOperator, LocalLiouvillian
from netket.hilbert import AbstractHilbert, DoubledHilbert
from netket.stats import Stats

PyTree = Any
ShapeT = Tuple[int, ...]
DTypeT = Any
PRNGKey = jnp.ndarray
InitFunType = Callable[[PRNGKey, ShapeT, DTypeT], jnp.ndarray]


class VariationalState(abc.ABC):
    """Abstract class for variational states representing either pure states
    or mixed quantum states.

    A variational state is a quantum state depending on a set of
    parameters, and that supports operations such
    as computing quantum expectation values and their gradients.

    A Variational stat can be serialized using flax's msgpack machinery.
    See `their docs <https://flax.readthedocs.io/en/latest/flax.serialization.html>`_.

    """

    def __init__(self, hilbert: AbstractHilbert):
        self._hilbert = hilbert  # type: AbstractHilbert

        self._model_state = {}  # type: PyTree
        self._parameters = {}  # type: PyTree

    @property
    def hilbert(self) -> AbstractHilbert:
        r"""The descriptor of the Hilbert space
        on which this variational state is defined.
        """
        return self._hilbert

    @property
    def parameters(self) -> PyTree:
        r"""The pytree of the parameters of the model."""
        return self._parameters

    @property
    def n_parameters(self) -> int:
        r"""The total number of parameters in the model."""
        return nkjax.tree_size(self.parameters)

    @parameters.setter
    def parameters(self, pars: PyTree):
        if not isinstance(pars, FrozenDict):
            if not isinstance(pars, list) and not isinstance(pars, tuple):
                pars = flax.core.freeze(pars)

        self._parameters = pars

    @property
    def model_state(self) -> Optional[PyTree]:
        r"""The optional pytree with the mutable state of the model."""
        return self._model_state

    @model_state.setter
    def model_state(self, state: PyTree):
        if not isinstance(state, FrozenDict):
            if not isinstance(state, list) and not isinstance(state, tuple):
                state = flax.core.freeze(state)

        self._model_state = state

    @property
    def variables(self) -> PyTree:
        r"""The PyTreee containing the paramters and state of the model,
        used when evaluating it.
        """
        return flax.core.freeze({"params": self.parameters, **self.model_state})

    @variables.setter
    def variables(self, vars: PyTree):
        if not isinstance(vars, FrozenDict):
            vars = flax.core.freeze(vars)

        self.model_state, self.parameters = vars.pop("params")

    def init_parameters(
        self, init_fun: Optional[InitFunType] = None, *, seed: Optional[PRNGKey] = None
    ):
        r"""
        Re-initializes all the parameters with the provided initialization function, defaulting to
        the normal distribution of standard deviation 0.01.

        Args:
            init_fun: a jax initializer such as :ref:`netket.nn.initializers.normal`. Must be a Callable
                taking 3 inputs, the jax PRNG key, the shape and the dtype, and outputting an array with
                the valid dtype and shape. If left unspecified, defaults to :code:`netket.nn.initializers.normal(stddev=0.01)`
            seed: Optional seed to be used. The seed is synced across all MPI processes. If unspecified, uses
                a random seed.
        """
        if init_fun is None:
            init_fun = nknn.initializers.normal(stddev=0.01)

        rng = nkjax.PRNGSeq(nkjax.PRNGKey(seed))

        def new_pars(par):
            return jnp.asarray(
                init_fun(rng.take(1)[0], shape=par.shape, dtype=par.dtype),
                dtype=par.dtype,
            )

        self.parameters = jax.tree_map(new_pars, self.parameters)

    def reset(self):
        r"""Resets the internal cache of th variational state.
        Called automatically when the parameters/state is updated.
        """
        pass

    @abc.abstractmethod
    def expect(self, Ô: AbstractOperator) -> Stats:
        r"""Estimates the quantum expectation value for a given operator O.
            In the case of a pure state $\psi$, this is $<O>= <Psi|O|Psi>/<Psi|Psi>$
            otherwise for a mixed state $\rho$, this is $<O> = \Tr[\rho \hat{O}/\Tr[\rho]$.

        Args:
            Ô: the operator O.

        Returns:
            An estimation of the quantum expectation value <O>.
        """
        raise NotImplementedError

    def grad(
        self, Ô, *, is_hermitian: Optional[bool] = None, mutable: Optional[Any] = None
    ) -> PyTree:
        r"""Estimates the gradient of the quantum expectation value of a given operator O.

        Args:
            op (netket.operator.AbstractOperator): the operator O.
            is_hermitian: optional override for whever to use or not the hermitian logic. By default
                it's automatically detected.

        Returns:
            array: An estimation of the average gradient of the quantum expectation value <O>.
        """
        return self.expect_and_grad(Ô, mutable=mutable)[1]

    def expect_and_grad(
        self,
        Ô: AbstractOperator,
        *,
        mutable: Optional[Any] = None,
        is_hermitian: Optional[bool] = None,
    ) -> Tuple[Stats, PyTree]:
        r"""Estimates both the gradient of the quantum expectation value of a given operator O.

        Args:
            Ô: the operator Ô for which we compute the expectation value and it's gradient
            mutable: Can be bool, str, or list. Specifies which collections in the model_state should
                     be treated as  mutable: bool: all/no collections are mutable. str: The name of a
                     single mutable  collection. list: A list of names of mutable collections.
                     This is used to mutate the state of the model while you train it (for example
                     to implement BatchNorm. Consult
                     `Flax's Module.apply documentation <https://flax.readthedocs.io/en/latest/_modules/flax/linen/module.html#Module.apply>`_
                     for a more in-depth exaplanation).
            is_hermitian: optional override for whever to use or not the hermitian logic. By default
                          it's automatically detected.

        Returns:
            An estimation of the quantum expectation value <O>.
            An estimation of the average gradient of the quantum expectation value <O>.
        """
        raise NotImplementedError

    # @abc.abstractmethod
    def quantum_geometric_tensor(self, sr):
        r"""Computes an estimate of the quantum geometric tensor G_ij.

        This function returns a linear operator that can be used to apply G_ij to a given vector
        or can be converted to a full matrix.

        Args:
            sr: The object containing the settings off the SR format to be used.

        Returns:
           A linear operator representing the quantum geometric tensor.
        """
        raise NotImplementedError

    def to_array(self, normalize: bool = True) -> jnp.ndarray:
        """
        Returns the dense-vector representation of this state.

        Args:
            normalize: If True, the vector is normalized to have L2-norm 1.

        Returns:
            An exponentially large vector representing the state in the computational
            basis.
        """
        return NotImplemented


class VariationalMixedState(VariationalState):
    def __init__(self, hilbert, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hilbert_physical = hilbert

    @property
    def hilbert_physical(self) -> AbstractHilbert:
        return self._hilbert_physical

    def expect(self, Ô: AbstractOperator) -> Stats:
        # If it is super-operator treat, they act on the same space so
        # the expectation value is standard.
        if self.hilbert == Ô.hilbert:
            return super().expect(Ô)
        elif self.hilbert_physical == Ô.hilbert:
            return self.expect_operator(Ô)
        else:
            return NotImplemented

    def expect_and_grad(
        self,
        Ô: AbstractOperator,
        mutable: bool = None,
    ) -> Tuple[Stats, PyTree]:
        # do the computation in super-operator space
        if self.hilbert == Ô.hilbert:
            return super().expect_and_grad(Ô, mutable=mutable)
        elif self.hilbert_physical == Ô.hilbert:
            return super().expect_and_grad(Ô, mutable=mutable)
        else:
            return NotImplemented

    @abc.abstractmethod
    def expect_operator(self, Ô: AbstractOperator) -> Stats:
        raise NotImplementedError

    def grad_operator(self, Ô: AbstractOperator) -> Stats:
        return self.expect_and_grad_operator(Ô)[1]

    # @abc.abstractmethod
    def expect_and_grad_operator(self, Ô: AbstractOperator) -> Stats:
        raise NotImplementedError

    def to_matrix(self, normalize: bool = True) -> jnp.ndarray:
        """
        Returns the dense-matrix representation of this operator.

        Args:
            normalize: If True, the matrix is normalized to have trace 1.

        Returns:
            An exponentially large matrix representing the state in the computational
            basis.
        """
        return NotImplemented