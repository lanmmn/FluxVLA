import pytest
import torch


def _load_position_embedding_module():
    import importlib.util
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / 'fluxvla' / 'ops' / 'triton' / 'position_embedding.py'
    spec = importlib.util.spec_from_file_location(
        'position_embedding_under_test', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(
    condition=torch.cuda.is_available() is False, reason='No GPU available.')
def test_fused_concat_with_pos_emb_matches_torch_with_preallocated_output():
    position_embedding_ops = _load_position_embedding_module()
    torch.manual_seed(0)
    batch = 2
    t_state = 1
    t_future = 3
    t_action = 4
    dim = 192

    state_features = torch.randn(
        batch, t_state, dim, device='cuda', dtype=torch.bfloat16)
    future_tokens = torch.randn(
        t_future, dim, device='cuda', dtype=torch.bfloat16)
    action_features = torch.randn(
        batch, t_action, dim, device='cuda', dtype=torch.bfloat16)
    position_embedding = torch.randn(
        t_action, dim, device='cuda', dtype=torch.bfloat16)
    out = torch.empty(
        batch,
        t_state + t_future + t_action,
        dim,
        device='cuda',
        dtype=torch.bfloat16)

    actual = position_embedding_ops.fused_concat_with_pos_emb(
        state_features,
        future_tokens,
        action_features,
        position_embedding,
        out=out)
    expected = torch.cat((state_features, future_tokens.unsqueeze(0).expand(
        batch, -1, -1), action_features + position_embedding.unsqueeze(0)),
                         dim=1)

    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    condition=torch.cuda.is_available() is False, reason='No GPU available.')
def test_fused_concat_with_pos_emb_matches_torch_without_position_embedding():
    position_embedding_ops = _load_position_embedding_module()
    torch.manual_seed(1)
    batch = 1
    t_state = 2
    t_future = 2
    t_action = 3
    dim = 64

    state_features = torch.randn(
        batch, t_state, dim, device='cuda', dtype=torch.bfloat16)
    future_tokens = torch.randn(
        t_future, dim, device='cuda', dtype=torch.bfloat16)
    action_features = torch.randn(
        batch, t_action, dim, device='cuda', dtype=torch.bfloat16)

    actual = position_embedding_ops.fused_concat_with_pos_emb(
        state_features, future_tokens, action_features)
    expected = torch.cat((state_features, future_tokens.unsqueeze(0).expand(
        batch, -1, -1), action_features),
                         dim=1)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
