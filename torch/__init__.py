"""A self contained, pure Python subset of the PyTorch API used in the kata."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union


Scalar = Union[int, float]


# ---------------------------------------------------------------------------
# device handling


@dataclass(frozen=True)
class Device:
    type: str

    def __str__(self) -> str:  # pragma: no cover - debugging helper
        return self.type


def device(name: str) -> Device:
    return Device(name)


class _CudaModule:
    def is_available(self) -> bool:
        return False

    def get_device_name(self, index: int = 0) -> str:  # pragma: no cover - defensive
        raise RuntimeError("CUDA is not available in the torch stub")


cuda = _CudaModule()


# ---------------------------------------------------------------------------
# tensor helpers


def _infer_shape(data: Any) -> Tuple[int, ...]:
    if isinstance(data, (int, float)):
        return ()
    if isinstance(data, (list, tuple)):
        if not data:
            return (0,)
        first_shape = _infer_shape(data[0])
        for element in data:
            if _infer_shape(element) != first_shape:
                raise ValueError("Jagged arrays are not supported")
        return (len(data),) + first_shape
    raise TypeError(f"Unsupported tensor input: {type(data)!r}")


def _flatten(data: Any) -> List[Scalar]:
    if isinstance(data, (int, float)):
        return [data]
    flat: List[Scalar] = []
    for element in data:
        flat.extend(_flatten(element))
    return flat


def _reshape(flat: List[Scalar], shape: Tuple[int, ...]) -> Any:
    if not shape:
        if len(flat) != 1:
            raise ValueError("Shape does not match data")
        return flat[0]
    size = shape[0]
    step = len(flat) // size
    return [_reshape(flat[i * step : (i + 1) * step], shape[1:]) for i in range(size)]


def _transpose(data: Any, dims: Tuple[int, ...]) -> Any:
    if not dims:
        return data
    if len(dims) == 1:
        return data
    if len(dims) == 2:
        return list(map(list, zip(*data)))
    # Recursive transpose for higher dimensional tensors.
    first, *rest = dims
    axes = list(range(len(dims)))
    axes[0], axes[first] = axes[first], axes[0]
    swapped = _move_axis(data, 0, first)
    return _transpose(swapped, tuple(rest))


def _move_axis(data: Any, source: int, destination: int) -> Any:
    if source == destination:
        return data
    axes = list(range(len(_infer_shape(data))))
    axis = axes.pop(source)
    axes.insert(destination, axis)
    return _reorder_axes(data, axes)


def _reorder_axes(data: Any, order: List[int]) -> Any:
    if len(order) == 1:
        return data
    axis = order[0]
    remainder = order[1:]
    swapped = list(zip(*data)) if axis == 1 else data
    return [_reorder_axes(item, remainder) for item in swapped]


class Tensor:
    def __init__(self, data: Any, shape: Optional[Tuple[int, ...]] = None, dtype: str = "float32", *, requires_grad: bool = False) -> None:
        if shape is None:
            shape = _infer_shape(data)
        self._shape = shape
        self._data = tuple(_flatten(data))
        self.dtype = dtype
        self.requires_grad = requires_grad

    # ------------------------------------------------------------------
    # container protocol helpers
    def __iter__(self) -> Iterator[Scalar]:  # pragma: no cover - rarely used
        return iter(self._data)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"tensor({self.tolist()})"

    # ------------------------------------------------------------------
    # metadata
    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    def numel(self) -> int:
        return len(self._data)

    def dim(self) -> int:
        return len(self._shape)

    # ------------------------------------------------------------------
    # conversions
    def to(self, *args: Any, dtype: Optional[str] = None, device: Optional[Device] = None) -> "Tensor":
        if args:
            arg = args[0]
            if isinstance(arg, Device):
                device = arg  # pragma: no cover - device is informational only
            elif isinstance(arg, str):
                dtype = arg
            else:  # pragma: no cover - defensive
                raise TypeError("Unsupported argument for Tensor.to()")

        target_dtype = dtype or self.dtype
        if target_dtype == self.dtype:
            data = list(self._data)
        elif target_dtype == "uint8":
            data = [int(value) for value in self._data]
        else:
            data = [float(value) for value in self._data]
        return Tensor(data, self._shape, target_dtype, requires_grad=self.requires_grad)

    def cpu(self) -> "Tensor":  # pragma: no cover - parity with torch
        return Tensor(self._data, self._shape, self.dtype, requires_grad=self.requires_grad)

    def detach(self) -> "Tensor":
        return Tensor(self._data, self._shape, self.dtype, requires_grad=False)

    def clone(self) -> "Tensor":
        return Tensor(list(self._data), self._shape, self.dtype, requires_grad=self.requires_grad)

    def contiguous(self) -> "Tensor":  # pragma: no cover - tensors are always contiguous
        return self.clone()

    # ------------------------------------------------------------------
    # reshaping
    def view(self, *shape: int) -> "Tensor":
        if shape.count(-1) > 1:
            raise ValueError("Only a single inferred dimension is supported")
        if -1 in shape:
            index = shape.index(-1)
            known = 1
            for value in shape:
                if value != -1:
                    known *= value
            inferred = self.numel() // known
            shape = list(shape)
            shape[index] = inferred
        if self.numel() != _num_from_shape(shape):
            raise ValueError("Shape is incompatible with tensor size")
        return Tensor(self._data, tuple(shape), self.dtype, requires_grad=self.requires_grad)

    def unsqueeze(self, dim: int) -> "Tensor":
        shape = list(self._shape)
        if dim < 0:
            dim += len(shape) + 1
        shape.insert(dim, 1)
        return Tensor(self._data, tuple(shape), self.dtype, requires_grad=self.requires_grad)

    def squeeze(self, dim: Optional[int] = None) -> "Tensor":
        shape = list(self._shape)
        if dim is None:
            shape = [value for value in shape if value != 1]
        else:
            if shape[dim] != 1:
                return Tensor(self._data, self._shape, self.dtype, requires_grad=self.requires_grad)
            shape.pop(dim)
        if not shape:
            shape = [1]
        return Tensor(self._data, tuple(shape), self.dtype, requires_grad=self.requires_grad)

    def permute(self, *dims: int) -> "Tensor":
        if sorted(dims) != list(range(len(self._shape))):
            raise ValueError("Invalid permutation")
        new_shape = tuple(self._shape[i] for i in dims)
        flat = list(self._data)
        permuted: List[Scalar] = [0] * len(flat)
        for index, value in enumerate(flat):
            coords = _coords_from_index(index, self._shape)
            new_coords = [coords[i] for i in dims]
            new_index = _index_from_coords(new_coords, new_shape)
            permuted[new_index] = value
        return Tensor(permuted, new_shape, self.dtype, requires_grad=self.requires_grad)

    # ------------------------------------------------------------------
    # arithmetic operations
    def _elementwise(self, other: Union["Tensor", Scalar], op) -> "Tensor":
        if isinstance(other, Tensor):
            if other.shape == self.shape:
                other_data = other._data
            elif other.numel() == 1:
                other_data = (other._data[0],) * self.numel()
            else:
                raise ValueError("Shapes must match for elementwise operations")
        else:
            other_data = (other,) * self.numel()
        result = [op(a, b) for a, b in zip(self._data, other_data)]
        return Tensor(result, self._shape, self.dtype, requires_grad=self.requires_grad)

    def __add__(self, other: Union["Tensor", Scalar]) -> "Tensor":
        return self._elementwise(other, lambda a, b: a + b)

    def __radd__(self, other: Scalar) -> "Tensor":  # pragma: no cover - symmetry
        return self.__add__(other)

    def __sub__(self, other: Union["Tensor", Scalar]) -> "Tensor":
        return self._elementwise(other, lambda a, b: a - b)

    def __mul__(self, other: Union["Tensor", Scalar]) -> "Tensor":
        return self._elementwise(other, lambda a, b: a * b)

    def __rmul__(self, other: Scalar) -> "Tensor":  # pragma: no cover - symmetry
        return self.__mul__(other)

    def __truediv__(self, other: Union["Tensor", Scalar]) -> "Tensor":
        return self._elementwise(other, lambda a, b: a / b)

    def clamp(self, min: Optional[Scalar] = None, max: Optional[Scalar] = None) -> "Tensor":
        def _apply(value: Scalar) -> Scalar:
            if min is not None:
                value = value if value >= min else min
            if max is not None:
                value = value if value <= max else max
            return value

        return Tensor([_apply(v) for v in self._data], self._shape, self.dtype, requires_grad=self.requires_grad)

    # ------------------------------------------------------------------
    # representation helpers
    def tolist(self) -> Any:
        return _reshape(list(self._data), self._shape)


def _num_from_shape(shape: Sequence[int]) -> int:
    result = 1
    for value in shape:
        result *= value
    return result


def tensor(data: Any, dtype: Optional[str] = None) -> Tensor:
    return Tensor(data, dtype=dtype or "float32")


float32 = "float32"
uint8 = "uint8"


class no_grad:
    def __enter__(self) -> None:  # pragma: no cover - trivial
        return None

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        return None


class Parameter(Tensor):
    def __init__(self, data: Any) -> None:
        super().__init__(data, requires_grad=True)


class Module:
    training = True

    def parameters(self) -> Iterable[Parameter]:
        for value in self.__dict__.values():
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - delegated
        return self.forward(*args, **kwargs)

    def to(self, *args: Any, **kwargs: Any) -> "Module":  # pragma: no cover - device handling is a no-op
        return self

    def eval(self) -> "Module":  # pragma: no cover - state flag only
        self.training = False
        return self

    def train(self, mode: bool = True) -> "Module":  # pragma: no cover - state flag only
        self.training = bool(mode)
        return self


nn = SimpleNamespace(Module=Module, Parameter=Parameter)


def _coords_from_index(index: int, shape: Tuple[int, ...]) -> List[int]:
    coords = []
    for size in reversed(shape):
        coords.append(index % size)
        index //= size
    return list(reversed(coords))


def _index_from_coords(coords: Sequence[int], shape: Tuple[int, ...]) -> int:
    index = 0
    for coord, size in zip(coords, shape):
        index = index * size + coord
    return index


# ---------------------------------------------------------------------------
# persistence helpers


def load(path: Union[str, Path], map_location: Optional[Device] = None) -> Dict[str, Tensor]:
    payload = json.loads(Path(path).read_text(encoding="utf8"))
    state: Dict[str, Tensor] = {}
    for name, item in payload.items():
        state[name] = Tensor(item["data"], tuple(item["shape"]), dtype=item.get("dtype", "float32"))
    return state


def save_state_dict(state_dict: Dict[str, Tensor], path: Union[str, Path]) -> None:
    serialisable: Dict[str, Dict[str, Any]] = {}
    for name, value in state_dict.items():
        serialisable[name] = {
            "shape": list(value.shape),
            "dtype": value.dtype,
            "data": value.tolist(),
        }
    Path(path).write_text(json.dumps(serialisable), encoding="utf8")


__all__ = [
    "Tensor",
    "tensor",
    "float32",
    "uint8",
    "Parameter",
    "Module",
    "nn",
    "device",
    "Device",
    "cuda",
    "no_grad",
    "load",
    "save_state_dict",
]

