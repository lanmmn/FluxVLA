# Git 将分支 Commit 压成一个的安全流程

> 适用场景：当前开发分支上有多个 commit，希望在 rebase 到最新 `main` 后整理成一个干净 commit，再推送到远端。

## 1. 基本原则

压 commit 前先保证三件事：

- 有备份分支，可以随时回退。
- 工作区没有未提交改动，或已提交成临时 WIP commit。
- 尽量在试验分支上操作，不直接动原开发分支。

推荐不要直接在原分支上做破坏性历史改写。先复制出一个 `test/*` 分支，确认结果没问题后，再让原分支指过去。

## 2. 假设分支名

以下命令假设你的开发分支是：

```bash
feat/lzh/orin-triton-optimise
```

如果实际分支名不同，替换成自己的分支名即可。

## 3. 先检查当前状态

```bash
git branch --show-current
git status --short
```

如果有未提交改动，建议先提交成临时 commit：

```bash
git add -A
git commit -m "WIP before squash"
```

不建议只依赖 `git stash`。复杂 rebase / squash 场景里，临时 commit 更容易追踪和恢复。

## 4. 创建保险分支

```bash
git switch feat/lzh/orin-triton-optimise
git branch backup/feat-lzh-orin-triton-optimise-before-squash
```

这个 backup 分支是最后的保险。如果 squash 或 rebase 过程出问题，可以回到它：

```bash
git switch feat/lzh/orin-triton-optimise
git reset --hard backup/feat-lzh-orin-triton-optimise-before-squash
```

## 5. 在试验分支上操作

```bash
git switch -c test/squash-feat-lzh-orin-triton-optimise
```

后续 squash 都在这个 `test/*` 分支上做。原分支和 backup 分支都不动。

## 6. 如果 main 已经是最新的

如果本地 `main` 已经同步到远程最新，可以直接压缩当前分支相对 `main` 的所有改动：

```bash
git reset --soft main
git commit -m "feat: add Orin Triton optimization support"
```

这两条命令的含义：

```text
git reset --soft main
= 把 HEAD 移回 main
= 保留当前分支相对 main 的所有改动在暂存区
= 不删除文件内容

git commit
= 把这些暂存改动重新提交成一个 commit
```

## 7. 如果还需要先 rebase main

如果你的开发分支还没有基于最新 `main`，推荐先 rebase，再 squash：

```bash
git fetch origin
git switch main
git merge --ff-only origin/main

git switch test/squash-feat-lzh-orin-triton-optimise
git rebase main
```

如果 rebase 出现冲突，先看冲突文件：

```bash
git status
git diff --name-only --diff-filter=U
```

解决冲突后：

```bash
git add <resolved-files>
git rebase --continue
```

如果发现处理错了，可以回退整个 rebase：

```bash
git rebase --abort
```

rebase 完成后，再压成一个 commit：

```bash
git reset --soft main
git commit -m "feat: add Orin Triton optimization support"
```

## 8. rebase 冲突时不要搞反 ours / theirs

rebase 冲突里经常容易把两边搞反：

```text
HEAD / ours  = 当前 rebase 基底，也就是 main 侧
theirs       = 正在 replay 的你的 commit 侧
```

如果需要看三方版本，可以用：

```bash
git show :1:path/to/file   # common ancestor
git show :2:path/to/file   # ours，main 侧
git show :3:path/to/file   # theirs，你的 commit 侧
```

看当前 Git 正在 replay 哪个 commit：

```bash
git rebase --show-current-patch
```

谨慎使用：

```bash
git rebase --skip
```

`--skip` 会丢掉当前正在 replay 的 commit，只有确认 main 已包含等价改动时才使用。

## 9. 验证只剩一个 commit

压缩完成后检查：

```bash
git log --oneline main..HEAD
```

理想情况下只显示一行。

再看改动范围：

```bash
git diff main...HEAD --stat
git diff main...HEAD --name-only
```

也建议跑必要检查，例如：

```bash
python3 -m py_compile path/to/touched.py
```

或项目自己的 test / lint 命令。

## 10. 用 squash 结果更新原分支

确认 `test/*` 分支结果正确后，再让原开发分支指向这个 squash 结果：

```bash
git switch feat/lzh/orin-triton-optimise
git reset --hard test/squash-feat-lzh-orin-triton-optimise
```

此时原开发分支就变成了：

```text
main
└── 一个干净 commit
```

## 11. 推送到远端

如果这个分支之前已经推送过，squash 后历史发生变化，需要强制更新远端分支：

```bash
git push --force-with-lease github feat/lzh/orin-triton-optimise
```

不要使用裸 `--force`。`--force-with-lease` 会检查远端是否被别人更新过，避免覆盖别人的新提交。

## 12. 最推荐的完整命令序列

```bash
git switch feat/lzh/orin-triton-optimise
git status --short

git branch backup/feat-lzh-orin-triton-optimise-before-rebase-squash
git switch -c test/rebase-squash-feat-lzh-orin-triton-optimise

git fetch origin
git switch main
git merge --ff-only origin/main

git switch test/rebase-squash-feat-lzh-orin-triton-optimise
git rebase main

# 如有冲突：解决冲突 -> git add -> git rebase --continue

git reset --soft main
git commit -m "feat: add Orin Triton optimization support"

git log --oneline main..HEAD
git diff main...HEAD --stat
```

确认没问题后：

```bash
git switch feat/lzh/orin-triton-optimise
git reset --hard test/rebase-squash-feat-lzh-orin-triton-optimise
git push --force-with-lease github feat/lzh/orin-triton-optimise
```

## 13. 恢复方式

如果最后不满意，回到备份分支：

```bash
git switch feat/lzh/orin-triton-optimise
git reset --hard backup/feat-lzh-orin-triton-optimise-before-rebase-squash
```

只要 backup 分支还在，原始 commit 历史就不会丢。

## 14. 本次 Orin 分支 commit message 示例

本次 Orin runtime、Docker 文档、Eagle 加速修复和 benchmark 相关改动，可以使用：

```text
feat: enable accelerated FluxVLA deployment and benchmark on Jetson Orin
```

如果需要填写完整 description，可以写成：

```text
feat: enable accelerated FluxVLA deployment and benchmark on Jetson Orin

- Add layered Docker runtime and build flow for Jetson Orin.
- Add Orin inference benchmark coverage and runtime notes.
- Fix EagleInferenceBackbone fully-masked attention rows for left-padded prompts.
- Restore accelerated GR00T inference with EagleInferenceBackbone and FlowMatchingInferenceHead.
- Add safetensors and image-token-aware benchmark loading for Orin tests.
- Document UR3 real-robot runtime setup and troubleshooting.
```

如果该 commit 已经推送到远端，`git commit --amend` 修改 message 后需要：

```bash
git push --force-with-lease github feat/lzh/orin-triton-optimise
```