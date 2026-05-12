#!/usr/bin/env python

from __future__ import annotations

import torch

from fluxvla.engines.utils.model_utils import (
    build_shared_obs_att_2d_masks_and_position_ids)


def main() -> None:
    prefix_pad_masks = torch.tensor([[True, True]])
    prefix_att_masks = torch.tensor([[False, False]])
    suffix_pad_masks = torch.tensor([[True, True]])
    suffix_att_masks = torch.tensor([[True, False]])
    offset_mask = torch.tensor([[True, True]])

    att_2d_masks, position_ids = (
        build_shared_obs_att_2d_masks_and_position_ids(
            prefix_pad_masks=prefix_pad_masks,
            prefix_att_masks=prefix_att_masks,
            suffix_pad_masks=suffix_pad_masks,
            suffix_att_masks=suffix_att_masks,
            num_offsets=2,
            offset_mask=offset_mask,
        ))

    assert att_2d_masks.shape == (1, 6, 6)
    assert position_ids.shape == (1, 6)

    # Suffix branch 0: indices [2, 3], branch 1: indices [4, 5]
    assert bool(att_2d_masks[0, 3, 2])
    assert bool(att_2d_masks[0, 5, 4])
    assert not bool(att_2d_masks[0, 3, 4])
    assert not bool(att_2d_masks[0, 5, 2])

    print('shared_obs_attention_mask: PASS')


if __name__ == '__main__':
    main()
