"""Shared exact-history and Fourier-Hermite phase-space diagnostic helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from zipfile import ZipFile

import numpy as np

from vpml.core import hermite_basis_phi_scaled


def _read_npy_header_from_npz(zf: ZipFile, member: str):
    fp = zf.open(member, "r")
    version = np.lib.format.read_magic(fp)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(fp)
    elif version == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(fp)
    else:
        raise ValueError(f"Unsupported npy version {version} for {member}")
    if fortran_order:
        raise ValueError(f"{member} is Fortran ordered; expected C order")
    return fp, tuple(int(value) for value in shape), np.dtype(dtype), fp.tell()


@dataclass(frozen=True)
class FourierHermiteHistoryReader:
    """Read a single coefficient frame from an uncompressed ``.npy`` in an ``.npz`` cache."""

    cache_path: Path
    array_name: str

    def read_slice(self, case_idx: int, time_idx: int, n_min: int, n_max: int) -> np.ndarray:
        with ZipFile(self.cache_path) as zf:
            fp, shape, dtype, data_start = _read_npy_header_from_npz(zf, self.array_name)
            if len(shape) != 4:
                raise ValueError(f"{self.array_name} must have shape (cases,time,Nv,Nk), got {shape}")
            ncase, ntime, nv, nk = shape
            if not 0 <= int(case_idx) < ncase:
                raise IndexError(f"case index {case_idx} is outside [0,{ncase})")
            if not 0 <= int(time_idx) < ntime:
                raise IndexError(f"time index {time_idx} is outside [0,{ntime})")
            if not 0 <= int(n_min) < int(n_max) <= nv:
                raise ValueError(f"Hermite range [{n_min},{n_max}) is outside [0,{nv})")
            offset_items = (((int(case_idx) * ntime + int(time_idx)) * nv + int(n_min)) * nk)
            count = (int(n_max) - int(n_min)) * nk
            fp.seek(data_start + offset_items * dtype.itemsize)
            data = fp.read(count * dtype.itemsize)
        return np.frombuffer(data, dtype=dtype).reshape((int(n_max) - int(n_min), nk)).astype(np.complex128)


def select_nearest_case(cache_path: Path, *, eps: float, eps_key: str = "strong_eps") -> Tuple[int, float]:
    with np.load(cache_path, allow_pickle=True) as data:
        values = np.asarray(data[eps_key], dtype=np.float64)
    index = int(np.argmin(np.abs(values - float(eps))))
    return index, float(values[index])


def resample_periodic_rows(rows: np.ndarray, *, Lx: float, target_nx: int) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    source_nx = int(rows.shape[1])
    target_nx = int(target_nx)
    if source_nx == target_nx:
        return rows.copy()
    x_source = np.linspace(0.0, float(Lx), source_nx, endpoint=False, dtype=np.float64)
    x_target = np.linspace(0.0, float(Lx), target_nx, endpoint=False, dtype=np.float64)
    x_ext = np.concatenate([x_source, np.asarray([float(Lx)])])
    values_ext = np.concatenate([rows, rows[:, :1]], axis=1)
    out = np.empty((rows.shape[0], target_nx), dtype=np.float64)
    for row_idx in range(rows.shape[0]):
        out[row_idx] = np.interp(x_target, x_ext, values_ext[row_idx])
    return out


def phase_space_from_hermite_phys(a_phys: np.ndarray, v_grid: np.ndarray, *, vth: float = 1.0) -> np.ndarray:
    """Reconstruct ``f(v,x)`` including the equilibrium contribution to ``C_0``."""

    a_phys = np.asarray(a_phys, dtype=np.float64)
    v_grid = np.asarray(v_grid, dtype=np.float64)
    phi = np.asarray(hermite_basis_phi_scaled(int(a_phys.shape[0]), v_grid, vth=float(vth)), dtype=np.float64)
    equilibrium = np.zeros((int(a_phys.shape[0]),), dtype=np.float64)
    equilibrium[0] = 1.0
    return ((a_phys + equilibrium[:, None]).T @ phi).T.astype(np.float64)
