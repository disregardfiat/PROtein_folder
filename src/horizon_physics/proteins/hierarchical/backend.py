"""
Backend abstraction for HKE: JAX (primary) with optional NumPy fallback.

GPU/TPU: use JAX with device placement. CPU: JAX or NumPy. All heavy loops
are JIT-compilable when JAX is available. Graceful degradation if JAX is not installed.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

_JAX_AVAILABLE = False
_jnp: Any = None
_jit: Any = None
_grad: Any = None
_vmap: Any = None
_pmap: Any = None
_device_put: Any = None
_default_backend: str = "numpy"

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap, device_put
    _JAX_AVAILABLE = True
    _jnp = jnp
    _jit = jit
    _grad = grad
    _vmap = vmap
    _device_put = device_put
    try:
        from jax import pmap
        _pmap = pmap
    except Exception:
        _pmap = None
except ImportError:
    import numpy as np
    _jnp = np
    _jit = lambda f: f
    _grad = None
    _vmap = None
    _pmap = None
    _device_put = lambda x, *_args, **_kwargs: x


def get_backend() -> str:
    """Return 'jax' or 'numpy'."""
    return "jax" if _JAX_AVAILABLE else "numpy"


def is_jax_available() -> bool:
    return _JAX_AVAILABLE


def np_or_jnp():
    """Return the array module (numpy or jax.numpy)."""
    return _jnp


def jit_if_available(fun: Any) -> Any:
    """JIT-compile if JAX is available; otherwise return fun unchanged."""
    return _jit(fun) if _JAX_AVAILABLE else fun


def grad_if_available(fun: Any) -> Any:
    """Return JAX grad(fun) if available; else None (use finite diff or external grad)."""
    return _grad(fun) if _JAX_AVAILABLE else None


def vmap_if_available(fun: Any, in_axes: int = 0, out_axes: int = 0) -> Any:
    """Vectorize over leading axis if JAX available."""
    if _JAX_AVAILABLE and _vmap is not None:
        return _vmap(fun, in_axes=in_axes, out_axes=out_axes)
    return fun


def device_put_if_available(x: Any, device: Optional[str] = None) -> Any:
    """Place array on device (gpu/tpu/cpu) if JAX available."""
    if not _JAX_AVAILABLE:
        return x
    if device is None:
        return x
    try:
        import jax
        dev = jax.devices(device)[0] if device in ("gpu", "tpu", "cuda") else jax.devices("cpu")[0]
        return _device_put(x, dev)
    except Exception:
        return x


def to_numpy(x: Any) -> Any:
    """Convert JAX array to NumPy if needed; otherwise return as-is."""
    import numpy as np
    if hasattr(x, "__array__") and callable(x.__array__):
        return np.asarray(x)
    return x
