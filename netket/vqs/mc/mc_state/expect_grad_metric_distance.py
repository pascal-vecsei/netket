#Calculates the Mean Metric distance between two states, and its gradient.

#defined as (1/|{\vec \sigma}|) \sum_{\{ \sigma \} } d(Psi(sigma) , Phi(sigma)) (for regular supervised learning)
#defined as (1/|{\vec \sigma}|) \sum_{\{ \sigma \} } d(Psi(sigma), O Phi(sigma)) (for application of operator)



from functools import partial
from typing import Any, Callable, Tuple

import jax
from jax import numpy as jnp

from netket import jax as nkjax
from netket import config
from netket.stats import Stats, statistics
from netket.utils import mpi
from netket.utils.types import PyTree
from netket.utils.dispatch import dispatch, TrueT, FalseT

from netket.operator import (
    AbstractOperator,
    DiscreteOperator,
    Squared,
)

from netket.vqs.mc import (
    get_local_kernel_arguments,
    get_local_kernel,
)

from .state import MCState

#@dispatch
#def get_local_kernel_arguments(vstate: MCState, origState: MCState, Ô: DiscreteOperator):  # noqa: F811
    #check_hilbert(vstate.hilbert, Ô.hilbert)
    #check_hilbert(origState.hilbert, Ô.hilbert)
    
    #σ = vstate.samples
    #σp, mels = Ô.get_conn_padded(σ)
    #return σ, (σp, mels)


@partial(jax.jit, static_argnums=(0, 1, 2))
def grad_expect_hermitian_for_distance(
    local_value_kernel: Callable,
    model_apply_fun: Callable,
    model2_apply_fun: Callable,
    mutable: bool,
    parameters: PyTree,
    parameters2: PyTree,
    model_state: PyTree,
    model_state2: PyTree,
    σ: jnp.ndarray,
    local_value_args: PyTree,
) -> Tuple[PyTree, PyTree]:

    σ_shape = σ.shape
    if jnp.ndim(σ) != 2:
        σ = σ.reshape((-1, σ_shape[-1]))

    n_samples = σ.shape[0] * mpi.n_nodes

    O_loc = local_value_kernel(
        model_apply_fun,
        model_apply_fun2,
        {"params": parameters, **model_state},
        {"params": parameters2, **model_state2},
        σ,
        local_value_args,
    )

    Ō = statistics(O_loc.reshape(σ_shape[:-1]).T)

    O_loc -= Ō.mean

    # Then compute the vjp.
    # Code is a bit more complex than a standard one because we support
    # mutable state (if it's there)
    is_mutable = mutable is not False
    _, vjp_fun, *new_model_state = nkjax.vjp(
        lambda w: model_apply_fun({"params": w, **model_state}, σ, mutable=mutable),
        parameters,
        conjugate=True,
        has_aux=is_mutable,
    )
    Ō_grad = vjp_fun(jnp.conjugate(O_loc) / n_samples)[0]

    Ō_grad = jax.tree_multimap(
        lambda x, target: (x if jnp.iscomplexobj(target) else 2 * x.real).astype(
            target.dtype
        ),
        Ō_grad,
        parameters,
    )

    new_model_state = new_model_state[0] if is_mutable else None

    return Ō, jax.tree_map(lambda x: mpi.mpi_sum_jax(x)[0], Ō_grad), new_model_state



@batch_discrete_kernel
def local_distance_kernel(logpsi: Callable, logpsi2: Callable, pars: PyTree, pars2: PyTree, σ: Array, args: PyTree): #logpsi: function that is fitted.
    """
    local_value kernel for MCState and generic operators
    """
    σp, mel = args
    #return jnp.sum(mel * jnp.exp(logpsi(pars, σp) - logpsi(pars, σ)))

    return jnp.exp(logpsi(pars, σ)) - jnp.sum(jnp.conj(mel) * jnp.exp(logpsi2(pars2, σp)))


def local_distance_squared_kernel(logpsi: Callable, logpsi2: Callable, pars: PyTree, pars2: PyTree, σ: Array, args: PyTree):
    """
    local_value kernel for MCState and Squared (generic) operators
    """
    return jnp.abs(local_value_kernel(logpsi, logpsi2, pars, pars2, σ, args)) ** 2


@dispatch
def expect_and_grad_distance_impl(  
    vstate: MCState,
    origState: MCState,
    Ô: AbstractOperator,
    use_covariance: TrueT,
    *,
    mutable: Any,
) -> Tuple[Stats, PyTree]:
    σ, args = get_local_kernel_arguments(vstate, Ô)

    #local_estimator_fun = get_local_kernel(vstate, Ô) #change just the local estimator fun, and which vstate is used for getting the samples / getting the connection
    
    local_estimator_fun = local_distance_squared_kernel
    
    Ō, Ō_grad, new_model_state = grad_expect_hermitian_for_distance(
        local_estimator_fun,
        vstate._apply_fun,
        origState._apply_fun,
        mutable,
        vstate.parameters,
        origState.parameters,
        vstate.model_state,
        origState.model_state,
        σ,
        args,
    )

    if mutable is not False:
        vstate.model_state = new_model_state

    return Ō, Ō_grad








