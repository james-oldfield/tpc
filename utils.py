from typing import List
import math

# helper functions

def get_current_order(epoch: int, num_epochs: int, min_order: int, max_order: int) -> int:
    if not 1 <= epoch <= num_epochs:
        raise ValueError(f"Epoch {epoch} must be between 1 and {num_epochs}")
    if min_order > max_order:
        raise ValueError(f"min_order ({min_order}) cannot be greater than max_order ({max_order})")
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive.")

    num_distinct_orders = max_order - min_order + 1
    order_index = int(((epoch - 1) * num_distinct_orders) / num_epochs)
    current_order = min_order + order_index
    
    return current_order

def compute_ee_hidden_sizes(target_params, input_dim, output_dim=1):
    hidden_dims = []
    main_branch_params_cumulative = 0
    prev_hidden_dim = input_dim

    for i, target in enumerate(target_params):
        denominator = prev_hidden_dim + 1 + output_dim
        numerator = target - main_branch_params_cumulative - output_dim

        if denominator <= 0:
            raise ValueError(f"Denominator is non-positive at layer {i+1}. Check input dimensions.")
        if numerator < 0:
            raise ValueError(
                f"Target param count {target:,} at layer {i+1} is too small "
                f"for the existing main branch params ({main_branch_params_cumulative:,})."
            )

        current_hidden_dim = int(round(numerator / denominator))

        hidden_dims.append(current_hidden_dim)

        params_current_w_layer = prev_hidden_dim * current_hidden_dim + current_hidden_dim
        main_branch_params_cumulative += params_current_w_layer

        prev_hidden_dim = current_hidden_dim

    return hidden_dims

def compute_hidden_size(target_params, input_dim, output_dim=1):
    denom = input_dim + output_dim + 1
    numerator = target_params - output_dim
    approx_H = numerator / denom

    h_floor = max(1, math.floor(approx_H))
    h_ceil  = max(1, math.ceil(approx_H))

    def total_params(H: int) -> int:
        return input_dim * H + H + H * output_dim + output_dim

    p_floor = total_params(h_floor)
    p_ceil  = total_params(h_ceil)

    # pick whichever is closer to target_params
    if abs(p_floor - target_params) <= abs(p_ceil - target_params):
        return h_floor
    else:
        return h_ceil


