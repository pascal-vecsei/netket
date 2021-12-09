# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial

import pytest
from pytest import raises

import numpy as np
import jax
import netket as nk
from jax.nn.initializers import normal

from netket.operator import AbstractSuperOperator

from .. import common

pytestmark = common.skipif_mpi

nk.config.update("NETKET_EXPERIMENTAL", True)

SEED = 2148364

machines = {}

standard_init = normal()
NDM = partial(nk.models.NDM, bias_init=standard_init, visible_bias_init=standard_init)

machines["model:(R->C)"] = NDM(
    alpha=1,
    beta=1,
    dtype=float,
    kernel_init=normal(stddev=0.1),
    bias_init=normal(stddev=0.1),
)
# machines["model:(C->C)"] = RBM(
#    alpha=1,
#    dtype=complex,
#    kernel_init=normal(stddev=0.1),
#    bias_init=normal(stddev=0.1),
# )

operators = {}

L = 2
g = nk.graph.Hypercube(length=L, n_dim=1)
hi = nk.hilbert.Spin(s=0.5, N=L)

ha = nk.operator.Ising(hi, graph=g, h=1.0, dtype=complex)
jump_ops = [nk.operator.spin.sigmam(hi, i) for i in range(L)]

liouv = nk.operator.LocalLiouvillian(ha.to_local_operator(), jump_ops)
LdagL = liouv.H @ liouv

# operators["operator:(Lind)"] = liouv
operators["operator:(Lind^2)"] = LdagL
operators["operator:H"] = ha
operators["operator:sigmam"] = jump_ops[0]


@pytest.fixture(params=[pytest.param(ma, id=name) for name, ma in machines.items()])
def vstate(request):
    ma = request.param

    sa = nk.sampler.ExactSampler(hilbert=nk.hilbert.DoubledHilbert(hi), n_chains=16)

    vs = nk.vqs.MCMixedState(sa, ma, n_samples=1000, seed=SEED)

    return vs


def test_n_samples_api(vstate):
    with raises(
        ValueError,
    ):
        vstate.n_samples = -1

    with raises(
        ValueError,
    ):
        vstate.chain_length = -2

    with raises(
        ValueError,
    ):
        vstate.n_discard_per_chain = -1

    vstate.n_samples = 2
    assert vstate.samples.shape[0:2] == (1, vstate.sampler.n_chains)

    vstate.chain_length = 2
    assert vstate.n_samples == 2 * vstate.sampler.n_chains
    assert vstate.samples.shape[0:2] == (2, vstate.sampler.n_chains)

    vstate.n_samples = 1000
    vstate.n_discard_per_chain = None
    assert vstate.n_discard_per_chain == 0

    vstate.sampler = nk.sampler.MetropolisLocal(
        hilbert=nk.hilbert.DoubledHilbert(hi), n_chains=16
    )
    vstate.n_discard_per_chain = None
    assert vstate.n_discard_per_chain == vstate.n_samples // 10


def test_n_samples_diag_api(vstate):
    with raises(
        ValueError,
    ):
        vstate.n_samples_diag = -1

    with raises(
        ValueError,
    ):
        vstate.chain_length_diag = -2

    with raises(
        ValueError,
    ):
        vstate.n_discard_per_chain = -1

    vstate.n_samples_diag = 2
    assert (
        vstate.diagonal.samples.shape[0:2]
        == (1, vstate.sampler_diag.n_chains)
        == (1, vstate.diagonal.sampler.n_chains)
    )

    vstate.chain_length_diag = 2
    assert (
        vstate.n_samples_diag
        == 2 * vstate.sampler_diag.n_chains
        == 2 * vstate.diagonal.sampler.n_chains
    )
    assert (
        vstate.diagonal.samples.shape[0:2]
        == (2, vstate.diagonal.sampler.n_chains)
        == (2, vstate.sampler_diag.n_chains)
    )

    vstate.n_samples = 1000
    vstate.n_discard_per_chain = None
    assert vstate.n_discard_per_chain_diag == 0

    vstate.sampler_diag = nk.sampler.MetropolisLocal(hilbert=hi, n_chains=16)
    vstate.n_discard_per_chain_diag = None
    assert vstate.n_discard_per_chain_diag == vstate.n_samples_diag // 10


def test_deprecations(vstate):
    vstate.sampler_diag = nk.sampler.MetropolisLocal(hilbert=hi, n_chains=16)

    # deprecation
    with pytest.warns(FutureWarning):
        vstate.n_discard_diag = 10

    with pytest.warns(FutureWarning):
        vstate.n_discard_diag

    vstate.n_discard_diag = 10
    assert vstate.n_discard_diag == 10
    assert vstate.n_discard_per_chain_diag == 10


def test_serialization(vstate):
    from flax import serialization

    bdata = serialization.to_bytes(vstate)

    vstate_new = nk.vqs.MCMixedState(
        vstate.sampler, vstate.model, n_samples=10, seed=SEED + 313
    )

    vstate_new = serialization.from_bytes(vstate_new, bdata)

    jax.tree_multimap(
        np.testing.assert_allclose, vstate.parameters, vstate_new.parameters
    )
    np.testing.assert_allclose(vstate.samples, vstate_new.samples)
    np.testing.assert_allclose(vstate.diagonal.samples, vstate_new.diagonal.samples)
    assert vstate.n_samples == vstate_new.n_samples
    assert vstate.n_discard_per_chain == vstate_new.n_discard_per_chain
    assert vstate.n_samples_diag == vstate_new.n_samples_diag
    assert vstate.n_discard_per_chain_diag == vstate_new.n_discard_per_chain_diag


@pytest.mark.parametrize(
    "operator",
    [
        pytest.param(
            op,
            id=name,
        )
        for name, op in operators.items()
    ],
)
def test_expect_numpysampler_works(vstate, operator):
    sampl = nk.sampler.MetropolisLocalNumpy(vstate.hilbert)
    vstate.sampler = sampl
    out = vstate.expect(operator)
    assert isinstance(out, nk.stats.Stats)


@pytest.mark.parametrize(
    "operator",
    [
        pytest.param(
            op,
            id=name,
        )
        for name, op in operators.items()
        if op.is_hermitian
    ],
)
@pytest.mark.parametrize("n_chunks", [1, 2])
def test_expect_chunking(vstate, operator, n_chunks):
    vstate.n_samples = 208
    chunk_size = vstate.n_samples_per_rank // n_chunks
    chunk_size_diag = vstate.diagonal.n_samples_per_rank // n_chunks

    eval_nochunk = vstate.expect(operator)
    vstate.chunk_size = chunk_size
    vstate.diagonal.chunk_size = chunk_size_diag
    eval_chunk = vstate.expect(operator)

    jax.tree_multimap(
        partial(np.testing.assert_allclose, atol=1e-13), eval_nochunk, eval_chunk
    )


@pytest.mark.parametrize("n_chunks", [1, 2])
def test_expect_grad_chunking(vstate, n_chunks):
    operator = LdagL

    vstate.n_samples = 208
    chunk_size = vstate.n_samples_per_rank // n_chunks
    chunk_size_diag = vstate.diagonal.n_samples_per_rank // n_chunks

    vstate.chunk_size = None
    vstate.diagonal.chunk_size = None
    grad_nochunk = vstate.grad(operator)
    vstate.chunk_size = chunk_size
    vstate.diagonal.chunk_size = chunk_size_diag
    grad_chunk = vstate.grad(operator)

    jax.tree_multimap(
        partial(np.testing.assert_allclose, atol=1e-13), grad_nochunk, grad_chunk
    )


def test_qutip_conversion(vstate):
    # skip test if qutip not installed
    pytest.importorskip("qutip")

    rho = vstate.to_matrix()
    q_obj = vstate.to_qobj()

    assert q_obj.type == "oper"
    assert len(q_obj.dims) == 2
    assert q_obj.dims[0] == list(vstate.hilbert_physical.shape)
    assert q_obj.dims[1] == list(vstate.hilbert_physical.shape)

    assert q_obj.shape == (
        vstate.hilbert_physical.n_states,
        vstate.hilbert_physical.n_states,
    )
    np.testing.assert_allclose(q_obj.data.todense(), rho)