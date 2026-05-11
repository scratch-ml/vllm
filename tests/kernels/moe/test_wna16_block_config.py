# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.layers.fused_moe.fused_moe import (
    get_moe_wna16_block_config,
)


def test_wna16_block_config_uses_expert_count_not_output_size():
    """The WNA16 CUDA tile heuristic depends on the number of experts.

    B is shaped as [num_experts, size_n, size_k].  A previous call-site bug
    passed B.size(1) (the output dimension) as num_experts, which can choose a
    different BLOCK_SIZE_K for small-expert models.
    """
    common_kwargs = dict(
        config={"BLOCK_SIZE_M": 16},
        use_moe_wna16_cuda=True,
        num_valid_tokens=16,
        size_k=1024,
        size_n=256,
        group_size=128,
        real_top_k=1,
        block_size_m=16,
    )

    correct_config = get_moe_wna16_block_config(
        **common_kwargs,
        num_experts=4,
    )
    output_size_as_experts_config = get_moe_wna16_block_config(
        **common_kwargs,
        num_experts=256,
    )

    assert correct_config == {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128}
    assert output_size_as_experts_config == {
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 256,
    }
