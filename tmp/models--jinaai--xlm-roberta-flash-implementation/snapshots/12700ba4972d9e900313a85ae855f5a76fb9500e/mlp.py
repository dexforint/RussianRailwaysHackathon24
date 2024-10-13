# This implementation was adapted from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/modules/mlp.py
# Commit id: c3b219665292c61a51153d0ded4473c494296382

# Copyright (c) 2023, Tri Dao.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup

try:
    from flash_attn.ops.activations import swiglu
except ImportError:
    swiglu = None

try:
    from flash_attn.ops.fused_dense import (ColumnParallelLinear,
                                            RowParallelLinear)
except ImportError:
    ColumnParallelLinear, RowParallelLinear = None, None

try:
    from flash_attn.ops.fused_dense import FusedMLP, ParallelFusedMLP
except ImportError:
    FusedMLP, ParallelFusedMLP = None, None


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation=F.gelu,
        bias1=True,
        bias2=True,
        return_residual=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = (
            hidden_features if hidden_features is not None else in_features * 4
        )
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias1, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(
            hidden_features, out_features, bias=bias2, **factory_kwargs
        )

    def forward(self, x, adapter_mask=None):
        if adapter_mask is not None:
            unique_tasks = torch.unique(adapter_mask)
            fc1_dtype = next(self.fc1.parameters()).dtype
            y = torch.empty(
                *x.shape[:-1], self.fc1.out_features, dtype=fc1_dtype, device=x.device
            )
            for task_id in unique_tasks:
                task_indices = (adapter_mask == task_id).nonzero(as_tuple=True)[0]
                task_tensor = x[task_indices]
                task_y = self.fc1(task_tensor, task_id=task_id)
                y[task_indices] = task_y
        else:
            y = self.fc1(x)

        y = self.activation(y)

        if adapter_mask is not None:
            unique_tasks = torch.unique(adapter_mask)
            fc2_dtype = next(self.fc2.parameters()).dtype
            out = torch.empty(
                *y.shape[:-1], self.fc2.out_features, dtype=fc2_dtype, device=y.device
            )
            for task_id in unique_tasks:
                task_indices = (adapter_mask == task_id).nonzero(as_tuple=True)[0]
                task_tensor = y[task_indices]
                task_out = self.fc2(task_tensor, task_id=task_id)
                out[task_indices] = task_out
        else:
            out = self.fc2(y)

        return out if not self.return_residual else (out, x)


class ParallelMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation=F.gelu,
        process_group: ProcessGroup = None,
        sequence_parallel=True,
        bias1=True,
        bias2=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert ColumnParallelLinear is not None, "Need to install fused_dense"
        assert RowParallelLinear is not None, "Need to install fused_dense"
        out_features = out_features if out_features is not None else in_features
        hidden_features = (
            hidden_features if hidden_features is not None else in_features * 4
        )
        self.fc1 = ColumnParallelLinear(
            in_features,
            hidden_features,
            process_group,
            bias=bias1,
            sequence_parallel=sequence_parallel,
            **factory_kwargs,
        )
        self.activation = activation
        self.fc2 = RowParallelLinear(
            hidden_features,
            out_features,
            process_group,
            bias=bias2,
            sequence_parallel=sequence_parallel,
            **factory_kwargs,
        )

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y


class GatedMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation=F.sigmoid,
        bias1=True,
        bias2=True,
        multiple_of=128,
        return_residual=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = (
            hidden_features if hidden_features is not None else int(8 * in_features / 3)
        )
        hidden_features = (
            (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        )
        self.return_residual = return_residual
        self.fc1 = nn.Linear(
            in_features, 2 * hidden_features, bias=bias1, **factory_kwargs
        )
        self.activation = activation
        self.fc2 = nn.Linear(
            hidden_features, out_features, bias=bias2, **factory_kwargs
        )

    def forward(self, x):
        y = self.fc1(x)
        if self.activation == F.sigmoid:  # Special case for GLU
            y = F.glu(y, dim=-1)
        elif (
            self.activation == F.silu and swiglu is not None
        ):  # Special case for SwiGLU
            y, gate = y.chunk(2, dim=-1)
            y = swiglu(gate, y)
        else:
            y, gate = y.chunk(2, dim=-1)
            y = y * self.activation(gate)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)


class ParallelGatedMlp(nn.Module):
    """Parallel GatedMlp"""

    def __init__(
        self,
        in_features,
        process_group,
        hidden_features=None,
        out_features=None,
        activation=F.sigmoid,
        bias1=True,
        bias2=True,
        multiple_of=128,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = (
            hidden_features if hidden_features is not None else int(8 * in_features / 3)
        )
        hidden_features = (
            (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        )
        if ColumnParallelLinear is None or RowParallelLinear is None:
            raise ImportError("fused_dense is not installed")
        self.fc1 = ColumnParallelLinear(
            in_features,
            2 * hidden_features,
            process_group,
            bias=bias1,
            sequence_parallel=sequence_parallel,
            **factory_kwargs,
        )
        self.activation = activation
        self.fc2 = RowParallelLinear(
            hidden_features,
            out_features,
            process_group,
            bias=bias2,
            sequence_parallel=sequence_parallel,
            **factory_kwargs,
        )

    def forward(self, x):
        y = self.fc1(x)
        if self.activation == F.sigmoid:  # Special case for GLU
            y = F.glu(y, dim=-1)
        else:
            y, gate = y.chunk(2, dim=-1)
            y = y * self.activation(gate)
        y = self.fc2(y)
        return y