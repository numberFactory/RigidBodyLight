import numpy as np
from Rigid import RigidBody
import scipy.sparse as sp
import utils
import time
from pyamg.krylov import gmres
from scipy.sparse.linalg import LinearOperator
from scipy.spatial.distance import pdist


# solve the system with fixed blobs and check that the velocity of the fixed blobs is close to zero.
def test_apply_with_fixed():
    X = np.array([[-2.0, 0.0, 1.5], [2.0, 0.0, 1.5]])
    Q = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    params, config = utils.load_config(utils.struct_shell_12)

    a = 0.5 * params["sep"]
    wall_x = 0.0
    wall_y = np.arange(-5.0, 5.0, 2 * a)
    wall_z = np.arange(a, 5.0, 2 * a)
    wall_X, wall_Y, wall_Z = np.meshgrid(wall_x, wall_y, wall_z)
    fixed_wall = np.column_stack((wall_X.flatten(), wall_Y.flatten(), wall_Z.flatten()))
    cb = utils.create_solver(
        X=X,
        Q=Q,
        rigid_config=config,
        fixed_config=fixed_wall,
        a=a,
        eta=1e-3,
        wall_PC=False,
    )

    sz = int(3 * cb.total_blobs + 6 * cb.N_bodies)
    RHS = np.random.rand(sz).astype(cb.precision)
    RHS[3 * len(config) : 3 * cb.total_blobs] = 0.0
    RHS_norm = np.linalg.norm(RHS)
    print("precision:", cb.precision)

    A = LinearOperator(shape=(sz, sz), matvec=cb.apply_saddle, dtype=cb.precision)
    PC = LinearOperator(shape=(sz, sz), matvec=cb.apply_PC, dtype=cb.precision)
    tol = 1e-4
    res_norms = []
    (sol, _) = gmres(
        A,
        (RHS / RHS_norm),
        M=PC,
        x0=None,
        tol=tol,
        callback=lambda rk: res_norms.append(np.linalg.norm(rk)),
    )
    sol *= RHS_norm

    lambda_vec = sol[: 3 * cb.total_blobs]

    v = cb.apply_M(lambda_vec, cb.get_blob_positions())

    v_fixed = v[3 * cb.blobs_per_body * cb.N_bodies :]
    mob_fact = 6 * np.pi * cb.eta * cb.a
    assert np.all(np.abs(v_fixed) < 10 * tol / mob_fact)
