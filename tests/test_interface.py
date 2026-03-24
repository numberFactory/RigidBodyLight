import numpy as np
import pytest
from Rigid import RigidBody
from scipy.spatial.transform import Rotation
import utils


def test_create():
    a = 1.0
    eta = 1.0
    _, config = utils.load_config(utils.struct_shell_12)

    N = 10
    X = np.random.randn(N, 3)
    Q = np.random.randn(N, 4)

    cb = RigidBody(config, X, Q, a, eta, dt=0.01)
    cb = RigidBody(config, X, Q, a, eta, dt=0.01, wall_PC=True)
    cb = RigidBody(config, X, Q, a, eta, dt=0.01, block_PC=True)

    with pytest.raises(RuntimeError):
        config = config.flatten()[:-1]
        cb = RigidBody(config, X, Q, a, eta, dt=0.01)


@pytest.mark.parametrize(
    ("X_shape", "Q_shape"), (((10, 3), (10, 4)), ((15, 1), (20, 1)))
)
@pytest.mark.parametrize("vector_type", (list, np.array))
def test_config(X_shape, Q_shape, vector_type):
    X_0 = np.random.rand(*X_shape)
    Q_0 = np.random.rand(*Q_shape)
    X_0 = vector_type(X_0)
    Q_0 = vector_type(Q_0)

    cb = utils.create_solver(X=X_0, Q=Q_0)
    cb.set_config(X_0, Q_0)

    Q_0 = Rotation.from_quat(np.reshape(Q_0, (-1, 4))).as_quat().reshape(Q_shape)

    X, Q = cb.get_config()
    assert np.allclose(X, X_0)
    assert np.allclose(Q, Q_0)


def test_bad_config():
    n = 10
    X_0 = np.random.rand(n, 3)
    Q_0 = np.random.rand(n, 4)

    cb = utils.create_solver(X=X_0, Q=Q_0)

    with pytest.raises(RuntimeError):
        cb.set_config(X_0, Q_0[: n - 1])

    with pytest.raises(RuntimeError):
        cb.set_config(X_0[: n - 1], Q_0)


def test_blob_positions():
    N = 5
    X, Q = utils.create_random_positions(N)
    _, config = utils.load_config(utils.struct_shell_12)
    blobs_per_body = config.shape[0]
    cb = utils.create_solver(rigid_config=config, X=X, Q=Q)

    N_blobs = N * blobs_per_body
    pos = cb.get_blob_positions()
    assert pos.shape == (N_blobs, 3)

    ref_pos = np.zeros((N_blobs, 3))
    for i in range(N):
        x_i = X[i, :]
        r_i = Rotation.from_quat(Q[i, :], scalar_first=True)
        pos_i = r_i.apply(config.copy()) + x_i
        ref_pos[i * blobs_per_body : (i + 1) * blobs_per_body, :] = pos_i

    assert np.allclose(pos, ref_pos, atol=1e-5)


@pytest.mark.parametrize("vector_type", (list, np.array))
@pytest.mark.parametrize("flat_inputs", [True, False])
# @pytest.mark.parametrize("include_fixed", [True, False])
@pytest.mark.parametrize("include_fixed", [False])
def test_all(vector_type, flat_inputs, include_fixed):
    N_rigid = 3
    X, Q = utils.create_random_positions(N_rigid)
    _, config = utils.load_config(utils.struct_shell_12)
    fixed_config = np.random.randn(3, 3) if include_fixed else None
    cb = utils.create_solver(rigid_config=config, X=X, Q=Q, fixed_config=fixed_config)
    blobs_per_body = config.shape[0]

    # TODO: include apply_M. evolve probably stays stand-alone since it has no direct outputs
    # TODO: also need to modify this for include_fixed=True
    funcs_to_input = {
        cb.K_dot: {"input_type": "body", "output_type": "blob"},
        cb.KT_dot: {"input_type": "blob", "output_type": "body"},
        cb.apply_PC: {"input_type": "system", "output_type": "system"},
        cb.apply_saddle: {"input_type": "system", "output_type": "system"},
        # cb.apply_M: {"input_type": "blob", "output_type": "blob"},
        # cb.evolve_rigid_bodies: {"input_type": "body", "output_type": "body"},
    }
    type_mapping = {
        "body": 6 * N_rigid,
        "blob": 3 * blobs_per_body * N_rigid,
        "system": 3 * blobs_per_body * N_rigid + 6 * N_rigid,
    }
    for func, types in funcs_to_input.items():
        in_size = type_mapping[types["input_type"]]
        out_size = type_mapping[types["output_type"]]

        input_vec = np.random.randn(in_size)
        if flat_inputs and types["input_type"] != "system":
            if types["input_type"] == "body":
                input_vec = np.reshape(input_vec, (-1, 6))
                out_shape = (blobs_per_body * N_rigid, 3)
            elif types["input_type"] == "blob":
                input_vec = np.reshape(input_vec, (-1, 3))
                out_shape = (2 * N_rigid, 3)
        else:
            out_shape = (out_size,)

        result = func(vector_type(input_vec))
        msg = "func: {}, expected output shape: {}, got shape: {}".format(
            func.__name__, out_shape, np.shape(result)
        )
        assert np.shape(result) == out_shape, msg
        assert np.linalg.norm(result) > 0.0
    pass


@pytest.mark.parametrize("vector_type", (list, np.array))
@pytest.mark.parametrize("flat_inputs", [False, True])
def test_K_dot(vector_type, flat_inputs):
    N_rigid = 3
    X, Q = utils.create_random_positions(N_rigid)
    _, config = utils.load_config(utils.struct_shell_12)
    cb = utils.create_solver(rigid_config=config, X=X, Q=Q)
    blobs_per_body = config.shape[0]

    U_bad_size = vector_type(np.random.randn(6 * N_rigid - 3))
    with pytest.raises(RuntimeError):
        cb.K_dot(U_bad_size)

    U_vec = np.random.randn(6 * N_rigid)
    if flat_inputs:
        out_shape = (3 * blobs_per_body * N_rigid,)
    else:
        U_vec = np.reshape(U_vec, (-1, 6))
        out_shape = (blobs_per_body * N_rigid, 3)

    result = cb.K_dot(vector_type(U_vec))
    assert result.shape == out_shape
    assert np.linalg.norm(result) > 0.0


@pytest.mark.parametrize("vector_type", (list, np.array))
@pytest.mark.parametrize("flat_inputs", [False, True])
def test_KT_dot(vector_type, flat_inputs):
    N_rigid = 3
    X, Q = utils.create_random_positions(N_rigid)
    _, config = utils.load_config(utils.struct_shell_12)
    cb = utils.create_solver(rigid_config=config, X=X, Q=Q)
    blobs_per_body = config.shape[0]

    lambda_bad_size = vector_type(np.random.randn(3 * blobs_per_body * N_rigid - 5))
    with pytest.raises(RuntimeError):
        cb.KT_dot(lambda_bad_size)

    lambda_vec = np.random.randn(3 * blobs_per_body * N_rigid)
    if flat_inputs:
        out_shape = (6 * N_rigid,)
    else:
        lambda_vec = np.reshape(lambda_vec, (-1, 3))
        out_shape = (2 * N_rigid, 3)
    result = cb.KT_dot(vector_type(lambda_vec))
    print("result shape:", (result.shape), "lambda_vec shape:", np.shape(lambda_vec))
    assert result.shape == out_shape
    assert np.linalg.norm(result) > 0.0


def test_get_K_Kinv():
    N_rigid = 3
    X, Q = utils.create_random_positions(N_rigid)
    _, config = utils.load_config(utils.struct_shell_12)
    cb = utils.create_solver(rigid_config=config, X=X, Q=Q)

    K = np.array(cb.get_K())
    K_inv = np.array(cb.get_Kinv())

    assert np.sum(np.abs(K)) > 0.0
    assert np.sum(np.abs(K_inv)) > 0.0


@pytest.mark.parametrize(
    ("block_PC", "wall_PC"),
    ((False, False), (True, False), (False, True), (True, True)),
)
@pytest.mark.parametrize("vector_type", (list, np.array))
def test_apply_PC(block_PC, wall_PC, vector_type):
    N_rigid = 3
    X, Q = utils.create_random_positions(N_rigid, wall_PC=wall_PC)
    _, config = utils.load_config(utils.struct_shell_12)
    cb = utils.create_solver(
        rigid_config=config, X=X, Q=Q, block_PC=block_PC, wall_PC=wall_PC
    )
    blobs_per_body = config.shape[0]

    size = 3 * blobs_per_body * N_rigid + 6 * N_rigid
    b = np.random.randn(size)
    PC = cb.apply_PC(vector_type(b))

    assert PC.shape == (size,)
    assert np.linalg.norm(PC) > 0.0

    with pytest.raises(RuntimeError):
        b_bad_size = np.random.randn(size - 4)
        cb.apply_PC(vector_type(b_bad_size))
    with pytest.raises(RuntimeError):
        b_bad_shape = np.random.randn(size).reshape(-1, 3)
        cb.apply_PC(vector_type(b_bad_shape))


@pytest.mark.parametrize("vector_type", (list, np.array))
@pytest.mark.parametrize("flat_inputs", [False, True])
def test_apply_M(vector_type, flat_inputs):
    N_rigid = 2
    X, Q = utils.create_random_positions(N_rigid)
    _, config = utils.load_config(utils.struct_shell_12)
    cb = utils.create_solver(rigid_config=config, X=X, Q=Q)
    blobs_per_body = config.shape[0]

    F = np.random.randn(3 * blobs_per_body * N_rigid)
    pos = cb.get_blob_positions()
    with pytest.raises(RuntimeError):
        cb.apply_M(F[:-4], pos)
    with pytest.raises(RuntimeError):
        cb.apply_M(F, pos[:-3])
    with pytest.raises(RuntimeError):
        cb.apply_M(F[:-1], pos[:-1])

    result = cb.apply_M(F, pos)
    F_bad_size = np.random.randn(3 * blobs_per_body * N_rigid - 4)
    with pytest.raises(RuntimeError):
        cb.apply_M(F_bad_size, pos)

    F = vector_type(np.random.randn(3 * blobs_per_body * N_rigid))
    if not flat_inputs:
        F = np.reshape(F, (-1, 3))
        pos = np.reshape(pos, (-1, 3))
    print(type(F))
    result = cb.apply_M(vector_type(F), vector_type(pos))
    shape = (3 * blobs_per_body * N_rigid,)
    assert result.shape == shape
    assert np.linalg.norm(result) > 0.0

    # check that we can also apply to a longer vector (e.g., if we have extra blobs)
    F = vector_type(np.random.randn(3 * blobs_per_body * N_rigid + 3))
    pos = vector_type(np.random.randn(3 * blobs_per_body * N_rigid + 3))
    result_long = cb.apply_M(F, pos)
    shape = (3 * blobs_per_body * N_rigid + 3,)
    assert result_long.shape == shape
    assert np.linalg.norm(result_long) > 0.0


@pytest.mark.parametrize("vector_type", (list, np.array))
def test_apply_saddle(vector_type):
    N_rigid = 2
    X, Q = utils.create_random_positions(N_rigid)
    _, config = utils.load_config(utils.struct_shell_12)
    cb = utils.create_solver(rigid_config=config, X=X, Q=Q)
    blobs_per_body = config.shape[0]

    size = 3 * blobs_per_body * N_rigid + 6 * N_rigid
    x = np.random.randn(size)

    out = cb.apply_saddle(x)
    assert out.shape == (size,)
    assert np.linalg.norm(out) > 0.0

    x_bad_size = np.random.randn(size - 2)
    with pytest.raises(RuntimeError):
        cb.apply_saddle(x_bad_size)

    x_bad_shape = np.random.randn(size).reshape(-1, 3)
    with pytest.raises(RuntimeError):
        cb.apply_saddle(vector_type(x_bad_shape))


@pytest.mark.parametrize("vector_type", (list, np.array))
@pytest.mark.parametrize("flat_inputs", [False, True])
def test_evolve_rigid_bodies(vector_type, flat_inputs):
    N_rigid = 3
    X, Q = utils.create_random_positions(N_rigid)
    _, config = utils.load_config(utils.struct_shell_12)
    cb = utils.create_solver(rigid_config=config, X=X, Q=Q)

    U = np.random.randn(6 * N_rigid)
    if not flat_inputs:
        U = np.reshape(U, (-1, 6))
    cb.evolve_rigid_bodies(vector_type(U))

    X_new, Q_new = cb.get_config()

    assert np.linalg.norm(X_new - X) > 0.0
    assert np.linalg.norm(Q_new - Q) > 0.0
