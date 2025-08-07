# This file is custom implementation - NOT from base model
# Custom LoRA module for task-specific fine-tuning

from __future__ import annotations

import math
import warnings
from typing import Any, Optional, Union, List

import torch
import torch.nn as nn

from peft.tuners.lora import LoraLayer

class MultiAdapterLinear(nn.Module, LoraLayer):
    """
    Custom LoRA module supporting multiple adapters for a linear layer.
    
    This module extends the standard LoRA implementation to support multiple task-specific
    adapters that can be dynamically selected during the forward pass. The task_label
    parameter passed to the forward function determines which LoRA adapter(s) to use:
    - If task_label is a string, all examples in the batch use the same adapter
    - If task_label is a list of strings, each example can use a different adapter
    
    This enables efficient multi-task inference where all task-specific LoRA adapters
    are loaded in memory simultaneously and dynamically selected per example, eliminating
    the need to switch adapter states between tasks and allowing optimal throughput
    for mixed-task batches.
    
    Derived from peft.tuners.lora.Linear.
    """
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        task_names: List[str],
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)

        self.fan_in_fan_out = fan_in_fan_out
        self.task_names = task_names
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer


    def forward(self, x: torch.Tensor, task_label: Union[str, List[str]], *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            lora_A_keys = self.lora_A.keys()
            for active_adapter in self.active_adapters:
                if active_adapter not in lora_A_keys:
                    continue
                
                if isinstance(task_label, str):
                    lora_A = self.lora_A[active_adapter][task_label]
                    lora_B = self.lora_B[active_adapter][task_label]
                    dropout = self.lora_dropout[active_adapter]
                    scaling = self.scaling[active_adapter]
                    x = self._cast_input_dtype(x, lora_A.weight.dtype)
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    unique_tasks = list(set(task_label))
                    lora_output = torch.zeros_like(result)
                    
                    for task in unique_tasks:
                        task_indices = [i for i, t in enumerate(task_label) if t == task]
                        task_x = x[task_indices]
                        
                        lora_A = self.lora_A[active_adapter][task]
                        lora_B = self.lora_B[active_adapter][task]
                        dropout = self.lora_dropout[active_adapter]
                        scaling = self.scaling[active_adapter]
                        
                        task_x = self._cast_input_dtype(task_x, lora_A.weight.dtype)
                        task_lora_value = lora_B(lora_A(dropout(task_x))) * scaling
                        
                        for i, idx in enumerate(task_indices):
                            lora_output[idx] = task_lora_value[i]
                    
                    result = result + lora_output

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        use_dora: bool = False,
        lora_bias: bool = False,
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        self.lora_A[adapter_name] = nn.ModuleDict({
            task_name: nn.Linear(self.in_features, r, bias=False)
            for task_name in self.task_names
        })
        self.lora_B[adapter_name] = nn.ModuleDict({
            task_name: nn.Linear(r, self.out_features, bias=lora_bias)
            for task_name in self.task_names
        })
        self.lora_bias[adapter_name] = lora_bias

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        self.reset_lora_parameters(adapter_name, init_lora_weights)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.use_dora[adapter_name] = False
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return
        if init_lora_weights is True:
            # initialize A the same way as the default for nn.Linear and B to zero
            # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
            for task_name in self.task_names:
                nn.init.kaiming_uniform_(self.lora_A[adapter_name][task_name].weight, a=math.sqrt(5))
        elif init_lora_weights.lower() == "gaussian":
            for task_name in self.task_names:
                nn.init.normal_(self.lora_A[adapter_name][task_name].weight, std=1 / self.r[adapter_name])
        else:
            raise ValueError(f"Unknown initialization {init_lora_weights=}")
        for task_name in self.task_names:
            nn.init.zeros_(self.lora_B[adapter_name][task_name].weight)
        if self.lora_bias[adapter_name]:
            for task_name in self.task_names:
                nn.init.zeros_(self.lora_B[adapter_name][task_name].bias)
    

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights
        """
        raise NotImplementedError("Merge operation is not supported")

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        raise NotImplementedError("Unmerge operation is not supported")
