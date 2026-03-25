from Rigid import c_rigid as crigid
import numpy as np
from typing import TypeAlias
import scipy.sparse as sp

vector: TypeAlias = list | np.ndarray
sparse_m: TypeAlias = sp.csc_matrix
"""Rigid body interface for Python.

Dev notes:
- The C++ code expects inputs as flattened numpy arrays. There's a decent bit of wrangling done in this Python interface to make it work seamlessly with lists & multi-dimensional arrays.
"""


class RigidBody:
    X_shape: tuple[int, ...]
    Q_shape: tuple[int, ...]
    K: sparse_m
    K_inv: sparse_m

    def __init__(
        self,
        rigid_config: vector,
        X: vector,
        Q: vector,
        a: float,
        eta: float,
        dt: float,
        fixed_config: vector | None = None,
        wall_PC=False,
        block_PC=False,
    ):
        self.cb = crigid.CManyBodies()
        self.precision = self.cb.precision

        kbt = 1.0  # TODO temp, do we need kbt in c_rigid at all?

        self.__check_configs(rigid_config, fixed_config)

        self.blobs_per_body = np.size(rigid_config) // 3
        self.fixed_blobs = 0 if fixed_config is None else np.size(fixed_config) // 3

        self.cb.setParameters(a, dt, kbt, eta, rigid_config)
        self.cb.setBlkPC(block_PC)
        self.cb.setWallPC(wall_PC)

        self.set_config(X, Q)

    def get_config(self) -> tuple[np.ndarray, np.ndarray]:
        X, Q = self.cb.getConfig()

        X = X.reshape(self.X_shape)
        Q = Q.reshape(self.Q_shape)
        return X, Q

    def set_config(self, X: vector, Q: vector) -> None:
        self.__check_and_set_configs(X, Q)
        X = np.array(X).ravel()
        Q = np.array(Q).ravel()
        self.cb.setConfig(X, Q)
        self.cb.set_K_mats()

        self.total_blobs = self.N_bodies * self.blobs_per_body + self.fixed_blobs
        self.__construct_K_mats()

    def get_blob_positions(self) -> np.ndarray:
        shape = (-1, 3) if len(self.X_shape) == 2 else (-1)
        return np.array(self.cb.multi_body_pos()).reshape(shape)

    def K_dot(self, U: vector) -> np.ndarray:
        self.__check_input_size(body_input=U)
        result = self.K.dot(np.array(U).ravel())
        shape = (-1, 3) if np.ndim(U) == 2 else (-1)
        return np.array(result).reshape(shape)

    def KT_dot(self, lambda_vec: vector) -> np.ndarray:
        self.__check_input_size(blob_input=lambda_vec)
        result = self.K_inv.dot(np.array(lambda_vec).ravel())
        shape = (-1, 3) if np.ndim(lambda_vec) == 2 else (-1)
        return np.array(result).reshape(shape)

    def apply_PC(self, b: vector) -> np.ndarray:
        self.__check_input_size(system_input=b)
        return self.cb.apply_PC(np.array(b))

    def apply_saddle(self, x: vector) -> np.ndarray:
        self.__check_input_size(system_input=x)
        lambda_vec = x[: 3 * self.total_blobs]
        U = x[3 * self.total_blobs :]
        r_vecs = self.get_blob_positions().flatten()
        slip = (
            self.apply_M(forces=lambda_vec, positions=r_vecs) - self.K_dot(U).flatten()
        )
        F = self.KT_dot(lambda_vec).flatten()
        return np.concatenate((slip, F))

    def apply_M(self, forces: vector, positions: vector) -> np.ndarray:
        if np.size(positions) != np.size(forces):
            raise RuntimeError("Positions and forces must be of the same size")
        self.__check_input_size(blob_input=forces)
        self.__check_input_size(blob_input=positions)
        shape = (-1, 3) if np.ndim(forces) == 2 and np.ndim(positions) == 2 else (-1)

        return self.cb.apply_M(
            np.reshape(forces, (-1)), np.reshape(positions, (-1))
        ).reshape(shape)

    def get_K(self) -> sparse_m:
        return self.K

    def get_Kinv(self) -> sparse_m:
        return self.K_inv

    def evolve_rigid_bodies(self, U: vector) -> None:
        self.__check_input_size(body_input=U)
        self.cb.evolve_X_Q(np.array(U).ravel())

    def __construct_K_mats(self):
        self.K = self.cb.get_K()
        self.K_inv = self.cb.get_Kinv()

        if self.fixed_blobs > 0:
            padded_shape = (3 * self.total_blobs, 6 * self.N_bodies)
            self.K.resize(padded_shape)
            self.K_inv.resize(padded_shape)
            # self.K = sp.csc_matrix(self.K, shape=padded_shape)
            # self.K_inv = sp.csc_matrix(self.K_inv, shape=padded_shape)

    def __check_configs(
        self, rigid_config: vector, fixed_config: vector | None
    ) -> None:
        if np.size(rigid_config) % 3 != 0:
            raise RuntimeError(
                f"Rigid config must have length 3N. Rigid config size: {np.size(rigid_config)}"
            )
        if fixed_config is not None:
            if np.size(fixed_config) % 3 != 0:
                raise RuntimeError(
                    f"Fixed config must have length 3N. Fixed config size: {np.size(fixed_config)}"
                )

    def __check_and_set_configs(self, X: vector, Q: vector) -> None:
        x_size = np.prod(np.shape(X))
        q_size = np.prod(np.shape(Q))

        if x_size % 3 != 0:
            raise RuntimeError("X must have total length 3N")
        if q_size % 4 != 0:
            raise RuntimeError("Q must have total length 4N")

        nx = x_size // 3
        nq = q_size // 4

        if nx != nq:
            raise RuntimeError("X and Q must have the same number of bodies")

        self.N_bodies = nx
        self.X_shape = np.shape(X)
        self.Q_shape = np.shape(Q)

    def __check_input_size(
        self,
        blob_input: vector | None = None,
        body_input: vector | None = None,
        system_input: vector | None = None,
    ):
        if blob_input is not None:
            if np.size(blob_input) != 3 * self.total_blobs:
                raise RuntimeError(
                    f"lambda must have total size 3*N_blobs = {3 * self.total_blobs}. lambda_vec shape: {np.shape(blob_input)}"
                )
        if body_input is not None:
            if np.size(body_input) != 6 * self.N_bodies:
                raise RuntimeError(
                    f"U must have total size 6*N_bodies = {6*self.N_bodies}. U shape: {np.shape(body_input)}"
                )

        if system_input is not None:
            expected_size = 3 * self.total_blobs + 6 * self.N_bodies
            if np.size(system_input) != expected_size:
                raise RuntimeError(
                    f"Rigid system input vector must have total size 3*N_blobs + 6*N_bodies = {expected_size}. system_input shape: {np.shape(system_input)}"
                )
            if np.ndim(system_input) != 1:
                raise RuntimeError("Rigid system input vector must be 1D")
