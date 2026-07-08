# Orin RTC Stable Prefix 7 Log

## Config

```python
inference['async_remaining_actions_threshold'] = 8
inference['execute_horizon'] = 16
inference['target_hz'] = 50
inference['rtc_config']['prefix_len'] = 7
```

## Observed behavior

This run shows stable async RTC handoff for a long sequence. The log repeatedly reports:

```text
prefix_len=7, remaining_actions=8
```

and chunks are accepted and sent by the actor without `Dropping inferred chunk` warnings in the captured section.

Representative actor handoff:

```text
[ACTOR] Sent chunk 28 at action_count=350
[GET_ACTIONS] Chunk inference took 147.2 ms (total 207.8 ms, postprocess 0.6 ms, prefix_len=7, remaining_actions=8)
[ACTOR] Sent chunk 29 at action_count=363
[GET_ACTIONS] Chunk inference took 147.0 ms (total 220.2 ms, postprocess 0.6 ms, prefix_len=7, remaining_actions=8)
[ACTOR] Sent chunk 30 at action_count=376
[GET_ACTIONS] Chunk inference took 137.3 ms (total 201.3 ms, postprocess 0.6 ms, prefix_len=7, remaining_actions=8)
[ACTOR] Sent chunk 31 at action_count=389
[GET_ACTIONS] Chunk inference took 149.4 ms (total 190.2 ms, postprocess 0.5 ms, prefix_len=7, remaining_actions=8)
```

Later stable sequence:

```text
[ACTOR] Sent chunk 48 at action_count=602
[GET_ACTIONS] Chunk inference took 151.8 ms (total 224.9 ms, postprocess 0.8 ms, prefix_len=7, remaining_actions=8)
[ACTOR] Sent chunk 49 at action_count=614
[GET_ACTIONS] Chunk inference took 137.9 ms (total 217.3 ms, postprocess 0.7 ms, prefix_len=7, remaining_actions=8)
[ACTOR] Sent chunk 50 at action_count=626
[GET_ACTIONS] Chunk inference took 136.2 ms (total 200.2 ms, postprocess 0.6 ms, prefix_len=7, remaining_actions=8)
[ACTOR] Sent chunk 51 at action_count=639
[GET_ACTIONS] Chunk inference took 140.4 ms (total 210.3 ms, postprocess 0.7 ms, prefix_len=7, remaining_actions=8)
```

Final captured section:

```text
[ACTOR] Sent chunk 60 at action_count=751
[GET_ACTIONS] Chunk inference took 147.3 ms (total 223.9 ms, postprocess 0.7 ms, prefix_len=7, remaining_actions=8)
[ACTOR] Sent chunk 61 at action_count=764
[GET_ACTIONS] Chunk inference took 141.2 ms (total 217.2 ms, postprocess 0.7 ms, prefix_len=7, remaining_actions=8)
[ACTOR] Sent chunk 62 at action_count=776
[GET_ACTIONS] Chunk inference took 137.7 ms (total 201.4 ms, postprocess 0.7 ms, prefix_len=7, remaining_actions=8)
[ACTOR] Sent chunk 63 at action_count=789
[GET_ACTIONS] Chunk inference took 135.3 ms (total 165.4 ms, postprocess 0.5 ms, prefix_len=7, remaining_actions=8)
[ACTOR] Sent chunk 64 at action_count=802
[GET_ACTIONS] Chunk inference took 134.8 ms (total 174.5 ms, postprocess 0.7 ms, prefix_len=7, remaining_actions=8)
```

## Latency range in captured section

Approximate observed ranges:

```text
predict_action: 130.6 - 155.0 ms
total:          165.4 - 228.4 ms
postprocess:    0.5 - 1.6 ms
```

Most total inference times are around:

```text
200 - 224 ms
```

## Interpretation

With `prefix_len=7` and `remaining_actions=8`, this run appears to stay within the prefix window in the captured section.

At the default source rate of 30 Hz:

```text
prefix window = 7 / 30 = 233 ms
```

The observed total latency is mostly below 233 ms, which explains why the RTC handoff succeeds here.

The worst captured total latency is approximately:

```text
228.4 ms
```

This is close to the 233 ms prefix window, so the configuration is still tight. Any latency spike above roughly 233 ms can still cause:

```text
prefix_window_passed
```

## Notes

This log was captured after the Orin kernel padding fix in:

```text
fluxvla/models/heads/flow_matching_inference_head.py
```

The captured section does not show chunk drops, unlike earlier runs where total latency often exceeded the effective prefix window.

