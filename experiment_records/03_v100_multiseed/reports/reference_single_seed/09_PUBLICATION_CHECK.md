# 公开发布检查

日期：2026-10-14（UTC）。目标：`https://github.com/yixuanluo197/lyx-msa`。

## 整理范围

- 本地目录：`/data2/yb/paper/lyx-msa/CMERC`，对应公开仓库根目录。
- 保留上游 Git 历史及其来源 `HITSZ-HLT/CMERC@815c7c060914f5b55db816c0ba3662b11d08552f`。
- 原 README 按字节保存在 `docs/UPSTREAM_README.md`；新增中文首页和 `docs/GPT_REVIEW_GUIDE.md`。
- 增补 `.gitignore`，排除数据、权重、缓存、运行产物和常见凭据文件。上游已跟踪的两份小型课程难度文件继续保留。
- 本次整理没有修改模型、训练、评测、测试、Shell 脚本或依赖快照；发布包含整理前已有的复现与因果变体修改。

## 实际验证

使用既有 `multimodalerc310` 环境，借用 `msa_extract` 的 pytest，未安装新依赖：

```bash
CUDA_VISIBLE_DEVICES='' /home/yangbin/miniconda3/envs/multimodalerc310/bin/python -c \
  'import sys; sys.path.append("/home/yangbin/miniconda3/envs/msa_extract/lib/python3.10/site-packages"); import pytest; raise SystemExit(pytest.main(["-q", "tests"]))'
```

结果：**138 passed in 5.48s**。

- 28 份 Python 文件的语法编译通过，Shell 脚本的 `bash -n` 通过。
- 原有已跟踪代码的 `git -c core.whitespace=cr-at-eol diff --check` 通过；本次新写的首页、审查指南、发布报告与忽略规则的暂存空白检查通过。
- 中文首页和新审查指南中的本地链接均存在。
- 整理前后的 38 份代码、脚本、配置及小型难度文件 SHA-256 一致。
- 候选文件和已有 Git 历史的常见凭据格式扫描未命中；候选文件没有大于 10 MiB 的文件或软链接。

系统 Python 编译上游 `HypergraphConv.py` 的数学 docstring 时有既有的 `invalid escape sequence` 提示；未改动该说明文本，语法检查与测试均通过。

完整暂存检查发现原始 README 有行尾空格／末尾空行，既有 `scripts/bootstrap_environment.sh`、`scripts/evaluate_checkpoint.py`、`scripts/with_cmerc_env.sh` 有末尾空行。这些是整理前已有的格式问题，保留原文件字节，未作代码格式清理。

## 未验证范围

此次只整理和验证公开代码，没有重新安装历史环境、启动训练或重跑真实数据评测。数据特征、checkpoint、预测明细、日志和完整运行目录不在公开仓库；历史报告中的性能数字仍需对应原始产物才能独立复核。当前因果变体仍没有训练后的性能结果。
