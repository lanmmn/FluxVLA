#!/usr/bin/env bash
# GR00T RTC WBT inference entry.
# Same script for simulation and real robot: MROS topics are the interface.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLUXVLA_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOY_ROOT="$(cd "$FLUXVLA_ROOT/.." && pwd)"

# -------- model/config --------------------------------------------------------
CONFIG="${CONFIG:-$FLUXVLA_ROOT/configs/gr00t/gr00t_hud04_rtc_done_full_finetune.py}"
MODELS_ROOT="${MODELS_ROOT:-/data/ckpts}"
MODEL_DIR="${MODEL_DIR:-$MODELS_ROOT/gr00t_rtc_wbt_june_task7_0630_latesttrash_8gpu_20260630_134743_epoch30}"
CKPT_PATH="${CKPT_PATH:-$MODEL_DIR/checkpoints/step-490560-epoch-30-loss=0.0123.safetensors}"

# -------- task prompts --------------------------------------------------------
PROMPT_0="${PROMPT_0:-Navigate forward to the first box. Grasp the plush toy from the first box. Turn around and move to the chair. Release the plush toy onto the chair.}"
PROMPT_1="${PROMPT_1:-Turn around and move back to the first box. Bend down, grasp the first box with both hands, and lift it. Carry the first box to the second box located in front of you. Place the first box on top of the second box.}"
PROMPT_2="${PROMPT_2:-Turn right and walk to the table. Pick up the basket from the floor with the right hand. Pick up the plush toys on the table with the left hand, one by one, and place them into the basket. After all plush toys are in the basket, place the basket on the floor.}"
PROMPT_3="${PROMPT_3:-Walk behind the sofa. Grasp the clothes with the left hand and drape them over the right forearm. Walk to the clothes rack and grasp the clothes with the left hand. Walk to the laundry basket and put the clothes into it one by one.}"
PROMPT_4="${PROMPT_4:-Turn right and walk to the low table. Bend down and pick up the plastic cup on the table with the right hand, and pick up the paper ball with the left hand. Turn left and walk to the trash can. Drop the trash into the trash can one by one.}"
PROMPT_5="${PROMPT_5:-Walk to the chair on the left. Rotate the chair with the left hand. Push the chair under the table with both hands.}"
PROMPT_6="${PROMPT_6:-Turn left and walk to the bookshelf. Grasp a wallet from the bookshelf. Turn around and walk to the person behind you. Hand the wallet to the person.}"
PROMPT_7="${PROMPT_7:-Turn right and walk back to the starting position.}"
PROMPT_8="${PROMPT_8:-Walk towards the white door, use the right hand to grasp the door handle, rotate the handle to open the door, use the left hand to further push the opened door, and finally stop at the side of the door.}"

# -------- runtime environment -------------------------------------------------
CONDA_ENV="${CONDA_ENV:-}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
RELEASE_SETUP="${RELEASE_SETUP:-/opt/limx/robot-tron2-r/install/local_setup.sh}"
RELEASE_PREFIX="$(dirname "$RELEASE_SETUP")"
SIM="${SIM:-false}"  # true=仿真，跳过 MROS release setup；默认 false=真机
STATUS_INTERVAL="${STATUS_INTERVAL:-1.0}"
TASK_SWITCH="${TASK_SWITCH:-1}"
META_MODE="${META_MODE:-0}"  # 1=只推理 META_ID 指定的单个 task
META_ID="${META_ID:-0}"
ONLINE_DONE_PLOT="${ONLINE_DONE_PLOT:-0}"
ONLINE_DONE_PLOT_DIR="${ONLINE_DONE_PLOT_DIR:-$FLUXVLA_ROOT/outputs/online_done_plot}"

if [[ ! -f "$CONFIG" ]]; then
  echo "config not found: $CONFIG" >&2
  exit 1
fi

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "checkpoint not found: $CKPT_PATH" >&2
  exit 1
fi

SIM_LOWER="${SIM,,}"
if [[ "$SIM_LOWER" == "true" || "$SIM_LOWER" == "1" ]]; then
  echo "[inference] SIM=true (simulation), skipping MROS release setup"
else
  export LD_LIBRARY_PATH="/opt/mros/tools/lib:${LD_LIBRARY_PATH:-}"

  if [[ -f "$RELEASE_SETUP" ]]; then
    set +u
    export COLCON_CURRENT_PREFIX="$RELEASE_PREFIX"
    source "$RELEASE_SETUP" >/dev/null 2>&1 || true
    unset COLCON_CURRENT_PREFIX
    set -u
  else
    echo "warning: MROS release setup not found: $RELEASE_SETUP" >&2
  fi
fi

if [[ -n "$CONDA_ENV" ]]; then
  if [[ -f "$CONDA_SH" ]]; then
    set +u
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    set -u
  else
    echo "warning: conda setup not found: $CONDA_SH" >&2
  fi
fi

cd "$FLUXVLA_ROOT"

echo "[inference] GR00T RTC WBT"
echo "[inference] ckpt: $CKPT_PATH"
echo "[inference] task switch: $TASK_SWITCH"
echo "[inference] single-task mode: $META_MODE (id=$META_ID)"
echo "[inference] online done plot: $ONLINE_DONE_PLOT"
echo "[inference] status prints every ${STATUS_INTERVAL}s; pass --verbose for full logs"

exec python - \
  --config "$CONFIG" \
  --ckpt-path "$CKPT_PATH" \
  --prompt-0 "$PROMPT_0" \
  --prompt-1 "$PROMPT_1" \
  --prompt-2 "$PROMPT_2" \
  --prompt-3 "$PROMPT_3" \
  --prompt-4 "$PROMPT_4" \
  --prompt-5 "$PROMPT_5" \
  --prompt-6 "$PROMPT_6" \
  --prompt-7 "$PROMPT_7" \
  --prompt-8 "$PROMPT_8" \
  --status-interval "$STATUS_INTERVAL" \
  --task-switch "$TASK_SWITCH" \
  --meta-mode "$META_MODE" \
  --meta-id "$META_ID" \
  --plot-done "$ONLINE_DONE_PLOT" \
  --plot-done-dir "$ONLINE_DONE_PLOT_DIR" \
  "$@" <<'PY'
import argparse
import inspect
import math
import os
import queue
import sys
import threading
import traceback
import warnings

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
warnings.filterwarnings(
    'ignore', message='.*UnsupportedFieldAttributeWarning.*')

_original_getsourcefile = inspect.getsourcefile


def _safe_getsourcefile(obj):
    try:
        return _original_getsourcefile(obj)
    except TypeError:
        return None


inspect.getsourcefile = _safe_getsourcefile


def _coerce_prompt(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ' '.join(_coerce_prompt(item) for item in value)
    return str(value)


class OutputFilter:
    def __init__(self, enabled):
        self.enabled = enabled
        self.stdout_fd = os.dup(1)
        self.stderr_fd = os.dup(2)
        self._error_lines_remaining = 0

    def write_status(self, message):
        if not message.endswith('\n'):
            message += '\n'
        os.write(self.stdout_fd, message.encode('utf-8', errors='replace'))

    def start(self):
        if not self.enabled:
            return
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, 1)
        os.dup2(write_fd, 2)
        os.close(write_fd)
        threading.Thread(
            target=self._drain,
            args=(read_fd,),
            daemon=True,
            name='QuietInferenceOutput').start()

    def _drain(self, read_fd):
        pending = b''
        with os.fdopen(read_fd, 'rb', buffering=0) as stream:
            while True:
                chunk = stream.read(4096)
                if not chunk:
                    if pending:
                        self._emit(pending.decode('utf-8', errors='replace'))
                    return
                pending += chunk
                while b'\n' in pending:
                    raw, pending = pending.split(b'\n', 1)
                    self._emit(raw.decode('utf-8', errors='replace'))

    def _emit(self, line):
        markers = (
            'ERROR', 'Error', 'Traceback', 'Exception', 'AssertionError',
            'RuntimeError', 'TypeError', 'KeyboardInterrupt',
            'Shutdown requested', 'Cleaning up',
        )
        if any(marker in line for marker in markers):
            self._error_lines_remaining = 80
        if self._error_lines_remaining > 0:
            os.write(self.stderr_fd,
                     (line + '\n').encode('utf-8', errors='replace'))
            self._error_lines_remaining -= 1


class TaskStatus:
    def __init__(self, output, interval):
        self.output = output
        self.interval = max(0.2, float(interval))
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.task_idx = 0
        self.task_count = 2
        self.task_id = '0'
        self.instruction = ''
        self.thread = threading.Thread(
            target=self._loop, daemon=True, name='TaskStatus')

    def set(self, task_idx, task_count, task_id, instruction):
        with self.lock:
            self.task_idx = int(task_idx)
            self.task_count = max(1, int(task_count))
            self.task_id = str(task_id)
            self.instruction = _coerce_prompt(instruction)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=1.0)

    def _loop(self):
        while not self.stop_event.is_set():
            with self.lock:
                instruction = self.instruction
                if len(instruction) > 120:
                    instruction = instruction[:117] + '...'
                line = (
                    f'[task] {self.task_idx + 1}/{self.task_count} '
                    f'id={self.task_id}: {instruction}')
            self.output.write_status(line)
            self.stop_event.wait(self.interval)


class OnlineDonePlotter:
    """Render model done/progress chunks into a live OpenCV window."""

    def __init__(self, output, enabled, out_dir, chunk_len, prefix_len,
                 threshold):
        self.output = output
        self.enabled = enabled
        self.out_dir = out_dir
        self.chunk_len = max(1, int(chunk_len))
        self.prefix_len = max(0, int(prefix_len))
        self.threshold = threshold
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._loop, daemon=True, name='OnlineDonePlot')
        self.chunks = []
        self.window_enabled = True
        self.latest_path = os.path.join(self.out_dir, 'latest_done_plot.png')

    def start(self):
        if not self.enabled:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        self.thread.start()

    def stop(self):
        if not self.enabled:
            return
        self.stop_event.set()
        self.queue.put(None)
        self.thread.join(timeout=2.0)

    def push(self, done_chunk):
        if not self.enabled or done_chunk is None:
            return
        try:
            import numpy as np
            values = np.asarray(done_chunk, dtype=np.float32).reshape(-1)
            values = values[:self.chunk_len].copy()
            self.queue.put(values)
        except Exception as exc:
            self.output.write_status(
                f'[online_done_plot] failed to enqueue chunk: {exc}')

    def _loop(self):
        try:
            import cv2
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception as exc:
            self.output.write_status(
                f'[online_done_plot] disabled: import failed: {exc}')
            return

        chunk_idx = 0
        stride = max(1, self.chunk_len - self.prefix_len)
        while not self.stop_event.is_set():
            item = self.queue.get()
            if item is None:
                break
            start = chunk_idx * stride
            self.chunks.append((start, item))
            chunk_idx += 1

            fig, ax = plt.subplots(figsize=(12, 4))
            colors = plt.cm.tab20(np.linspace(0, 1, 20))
            max_x = 1
            for idx, (chunk_start, values) in enumerate(self.chunks):
                xs = np.arange(chunk_start, chunk_start + len(values))
                max_x = max(max_x, int(xs[-1]) + 1)
                ax.plot(
                    xs,
                    values,
                    color=colors[idx % len(colors)],
                    alpha=0.85,
                    linewidth=1.4)
            if self.threshold is not None and math.isfinite(self.threshold):
                ax.axhline(
                    self.threshold,
                    color='red',
                    linestyle='--',
                    linewidth=1.0,
                    label='threshold')
                ax.legend(loc='upper right')
            ax.set_title(
                'Online VLA done/progress output '
                f'(chunks={len(self.chunks)}, prefix={self.prefix_len})')
            ax.set_xlabel('stitched action step')
            ax.set_ylabel('action[42] done/progress')
            ax.set_xlim(0, max_x)
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            rgba = rgba.reshape(height, width, 4)
            bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
            plt.close(fig)

            cv2.imwrite(self.latest_path, bgr)
            if self.window_enabled:
                try:
                    cv2.imshow('FluxVLA done/progress online', bgr)
                    cv2.waitKey(1)
                except Exception as exc:
                    self.window_enabled = False
                    self.output.write_status(
                        '[online_done_plot] cv2 window disabled; '
                        f'writing latest image to {self.latest_path}: {exc}')


def patch_status(runner, status):
    original_get_task_description = runner._get_task_description

    def get_task_description(task_id):
        task_id = str(task_id)
        instruction = _coerce_prompt(original_get_task_description(task_id))
        order = list(getattr(runner, 'done_subtask_order', []) or ['0'])
        try:
            task_idx = order.index(task_id)
        except ValueError:
            task_idx = getattr(runner, '_current_idx', 0)
        status.set(task_idx, len(order), task_id, instruction)
        return instruction

    runner._get_task_description = get_task_description

    if hasattr(runner, '_run_async_instruction'):
        original_run_async_instruction = runner._run_async_instruction

        def run_async_instruction(instruction, advance_event,
                                  final_done_event):
            order = list(getattr(runner, 'done_subtask_order', []) or ['0'])
            task_idx = getattr(runner, '_current_idx', 0)
            task_id = order[min(task_idx, len(order) - 1)]
            status.set(task_idx, len(order), task_id, instruction)
            return original_run_async_instruction(
                instruction, advance_event, final_done_event)

        runner._run_async_instruction = run_async_instruction


def patch_done_plot(runner, plotter):
    original_postprocess_actions = runner._postprocess_actions

    def postprocess_actions(raw_action):
        actions = original_postprocess_actions(raw_action)
        done_chunk = getattr(runner._action_ctx, 'done_chunk', None)
        plotter.push(done_chunk)
        return actions

    runner._postprocess_actions = postprocess_actions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt-path', required=True)
    parser.add_argument('--prompt-0', required=True)
    parser.add_argument('--prompt-1', required=True)
    parser.add_argument('--prompt-2', required=True)
    parser.add_argument('--prompt-3', required=True)
    parser.add_argument('--prompt-4', required=True)
    parser.add_argument('--prompt-5', required=True)
    parser.add_argument('--prompt-6', required=True)
    parser.add_argument('--prompt-7', required=True)
    parser.add_argument('--prompt-8', required=True)
    parser.add_argument('--status-interval', type=float, default=1.0)
    parser.add_argument(
        '--task-switch',
        default='0',
        choices=('0', '1', 'false', 'true', 'False', 'True'),
        help='enable done/progress-based task0->task1 switching')
    parser.add_argument(
        '--meta-mode',
        default='0',
        choices=('0', '1', 'false', 'true', 'False', 'True'),
        help='run only the task selected by --meta-id; overrides switching')
    parser.add_argument(
        '--meta-id',
        default='0',
        help='task id to run when --meta-mode is enabled, e.g. 8')
    parser.add_argument(
        '--plot-done',
        default='1',
        choices=('0', '1', 'false', 'true', 'False', 'True'),
        help='show online done/progress chunk plot window')
    parser.add_argument('--plot-done-dir', default='outputs/online_done_plot')
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='show full framework/runner logs')
    return parser.parse_args()


def main():
    args = parse_args()
    output = OutputFilter(enabled=not args.verbose)
    output.start()

    try:
        from mmengine import Config
        from fluxvla.engines import build_runner_from_cfg

        prompts = {
            str(task_id): _coerce_prompt(getattr(args, f'prompt_{task_id}'))
            for task_id in range(9)
        }
        task_switch_enabled = args.task_switch in ('1', 'true', 'True')
        meta_mode_enabled = args.meta_mode in ('1', 'true', 'True')
        meta_id = str(args.meta_id).strip()

        cfg = Config.fromfile(args.config)
        use_done_state_machine = bool(
            cfg.inference.get('use_done_state_machine', True))
        if meta_mode_enabled:
            if meta_id not in prompts:
                raise ValueError(
                    f'--meta-id {meta_id!r} is invalid; '
                    f'expected one of {sorted(prompts)}')
            cfg.inference.task_descriptions = {meta_id: prompts[meta_id]}
            cfg.inference.done_subtask_order = [meta_id]
            if use_done_state_machine:
                cfg.inference.done_threshold = float('inf')
            output.write_status(
                f'[inference] single-task mode: task {meta_id}')
        elif not use_done_state_machine:
            # A no-done checkpoint cannot switch prompts. Keep the task id
            # selected by its config (task8 for the door baseline).
            configured_order = list(
                cfg.inference.get('done_subtask_order', []) or [])
            task_id = str(configured_order[0]) if configured_order else '0'
            if task_id in prompts:
                prompt = prompts[task_id]
            else:
                prompt = _coerce_prompt(
                    cfg.inference.task_descriptions[task_id])
            cfg.inference.task_descriptions = {task_id: prompt}
            cfg.inference.done_subtask_order = [task_id]
            output.write_status(
                f'[inference] no-done checkpoint: fixed task {task_id}')
        elif task_switch_enabled:
            task_order = ['3', '0', '1', '2', '5', '4', '6', '7']
            cfg.inference.task_descriptions = {
                task_id: prompts[task_id] for task_id in task_order
            }
            # 抓衣服(3) -> 抓鲨鱼(0) -> 搬箱子(1)
            cfg.inference.done_subtask_order = task_order
        else:
            cfg.inference.task_descriptions = {'0': prompts['0']}
            cfg.inference.done_subtask_order = ['0']
            cfg.inference.done_threshold = float('inf')
        if use_done_state_machine:
            cfg.inference.done_dim_index = 42
            cfg.inference.denormalize_action.action_dim = 43
        cfg.inference.interactive = False
        cfg.inference.cfg = cfg
        cfg.inference.ckpt_path = args.ckpt_path

        runner = build_runner_from_cfg(cfg.inference)
        status = TaskStatus(output, args.status_interval)
        first_task_id = cfg.inference.done_subtask_order[0]
        first_prompt = cfg.inference.task_descriptions[first_task_id]
        status.set(0, len(cfg.inference.done_subtask_order), first_task_id,
                   first_prompt)
        patch_status(runner, status)
        plot_done_requested = args.plot_done in ('1', 'true', 'True')
        plot_done_enabled = plot_done_requested and use_done_state_machine
        if plot_done_requested and not use_done_state_machine:
            output.write_status(
                '[inference] online done plot disabled: checkpoint has no '
                'done/progress output')
        execute_horizon = cfg.inference.get('execute_horizon', None)
        chunk_len = execute_horizon or cfg.inference.get('action_chunk', 32)
        rtc_config = cfg.inference.get('rtc_config', {}) or {}
        plotter = OnlineDonePlotter(
            output=output,
            enabled=plot_done_enabled,
            out_dir=args.plot_done_dir,
            chunk_len=chunk_len,
            prefix_len=rtc_config.get('prefix_len', 0),
            threshold=cfg.inference.get('done_threshold', None))
        if use_done_state_machine:
            patch_done_plot(runner, plotter)

        runner.run_setup()
        output.write_status('[inference] model loaded; waiting for frames')
        status.start()
        plotter.start()
        try:
            runner.run(initial_instruction=first_prompt)
        finally:
            plotter.stop()
            status.stop()
            output.write_status('[inference] stopped')
    except Exception:
        output.write_status(traceback.format_exc())
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
PY
