# CDTR / SAQT：AutoDL RTX 5090 迁移准备

用于古籍竖排单列文本识别的独立迁移副本。代码取自 2026-10-15 正在运行的匹配实验冻结目录；CUDA C++ 源码补自同一项目仓库。源服务器上的训练、代码、环境均未改动。

当前状态：**已准备代码与候选兼容补丁，尚无 AutoDL 实例，未进行 5090 编译、推理或训练验证。** 代码同步目标为 [Longdawn/SAQT 的迁移分支](https://github.com/Longdawn/SAQT/tree/codex/autodl5090-migration)。

## 租卡配置

- RTX 5090 32GB，一张卡即可；建议主机内存至少 64GB、8 个或更多 CPU 逻辑核。
- 选择包含 **CUDA 12.8 编译器 nvcc** 的 Linux 开发镜像，例如 AutoDL 的 CUDA 12.8 / PyTorch 2.7 系列镜像。只有驱动或运行库不够编译自定义算子。
- 本包使用独立 Python 3.11 环境、PyTorch 2.7.1 + cu128、torchvision 0.22.1。镜像原有 Python 不必与此相同。
- 数据盘建议至少 100GB。迁移的裁剪图像、必要标注和初始权重约 7～8GB；训练还会持续生成检查点。

参考：[PyTorch 2.7 Blackwell 支持](https://pytorch.org/blog/pytorch-2-7/)、[官方历史版本安装命令](https://pytorch.org/get-started/previous-versions/)、[AutoDL 镜像说明](https://www.autodl.com/docs/base_config/)。

## GitHub 保存范围

上传代码、配置、依赖脚本、CUDA 源码和迁移校验清单。数据集、模型权重、编译产物和完整训练日志不进 Git；`.gitignore` 已覆盖这些内容。

外部 `text_renderer` 工作区也不上传；当前真实古籍行图训练不需要它。将来需要重新生成合成图像时，应单独准备该工具及其依赖。

沿用 3090 服务器已经配置并有推送权限的 `Longdawn/SAQT` 仓库，在 `codex/autodl5090-migration` 独立分支整合，保留仓库历史。未把源仓库旧历史中的已跟踪数据、旧 `.so` 和 `.o` 文件带入本快照；迁移分支也会从当前文件清单取消跟踪这类生成物，旧历史不改写。

目标仓库为现有公开仓库。迁移分支只同步已检查的代码、配置、安装脚本和校验清单，数据及权重通过独立 SSH 传输。

## 实例就绪后的步骤

1. 把迁移分支拉到数据盘，进入项目根目录：

   ```bash
   git clone --branch codex/autodl5090-migration --single-branch https://github.com/Longdawn/SAQT.git /root/autodl-tmp/SAQT
   cd /root/autodl-tmp/SAQT
   ```
2. 运行 `bash migration/bootstrap_5090.sh`。脚本建立独立 Conda 环境、编译 sm_120 CUDA 扩展，检查扩展前向和反向梯度，不启动训练。
3. 在本地配置两个 SSH 别名，然后从本地运行：

   ```powershell
   python migration/transfer_assets.py --source 3090-Codex --destination autodl5090 --target-root /root/autodl-tmp/SAQT
   ```

   实际 `--target-root` 应与 clone 目录相同。两端均使用本机 SSH 认证，不向云主机复制源服务器私钥。传输仅读取清单中的静态数据和历史初始权重；逐文件 SHA-256 随传输生成并在目标核对，已有同内容文件可以复用，不覆盖不同内容文件。

4. 在 AutoDL 项目根目录运行：

   ```bash
   /root/autodl-tmp/envs/cdtr5090-cu128/bin/python migration/preflight_data.py
   ```

   核对原始标签和字符框标注、所有训练/验证图像存在性、字表及样本数。当前应为训练 84,176 行、验证 10,948 行、字表 6,727，batch=2 时每轮 42,088 步。

5. 由执行迁移的助手完成严格权重加载、真实图像推理和少量训练步验证，记录每步耗时与显存。先保持 FP32、batch=2、图像尺寸、候选数和优化步数；验证通过后才制定新的长训练任务。所有长训练在 tmux 中运行。

本包没有会自动重启、续跑或复制当前 3090 两组长实验的入口。旧 `pipeline.py`、`preflight.py`、`diagnose_matched.py` 保存在本地 `migration/reference/*.txt` 供核对，并被 Git 忽略；其中的硬编码目录不能直接用于新实例。

## 关键兼容修改

- CUDA C++：用 `scalar_type()`、`data_ptr()`、`is_cuda()`、`TORCH_CHECK` 和当前 ATen 原子操作头文件替代旧接口；不修改算子计算公式。
- PyTorch 2.6 之后 `torch.load` 默认值变化：对项目已有、来源可信的历史检查点显式指定 `weights_only=False`，不做全局反序列化开关。
- 新环境只重编 CUDA 扩展，不复用 3090 环境下的 `.so`。
- 保留现有训练精度与超参数；混合精度与增大 batch 留待独立验证。

`migration/compatibility.patch` 提供逐行差异；`compatibility_changes.json` 保存修改前后哈希。GPU 运行结果以实例上生成的 `migration_reports/` 为准，当前这些 GPU 检查均未执行。

## 数据范围和复现边界

`migration/assets.json` 指定四个处理后的行图数据集、MTH1000/MTH1200/TKH 原始字符和行标注、可信初始 stage1 权重及字表。原始页面图像不是此训练入口的必要迁移项。

**字符框标注不能省略。** 数据加载器在找不到原始标注时会退回占位框，本包的 preflight 会对此直接报错。

当前 3090 上无预算/有预算匹配实验继续按原流水线运行；本包不会更改其进程或文件。既有 CPU/CUDA CER 复现差异仍是待调查问题，5090 迁移不等于已解决该问题。跨机器 CER 应在相同权重、样本顺序、batch、预处理和解码参数下比较；不能把混用旧新环境的结果当成单变量改进。

## 来源文件

- `migration/provenance.json`：源提交、源/目标环境与尚未验证的状态。
- `migration/source_files.json`：原始代码逐文件哈希与来源类型。
- `migration/source_package_versions.json`：源环境包版本；不含认证信息。
- `migration/source_data_fingerprints.json`：当前实验原始标签/标注指纹和样本数。
- `README_UPSTREAM.md`、`LICENSE`：保留项目说明和许可来源。
- `AGENTS.md`、`docs/CODEX_HANDOFF.md`：原项目约定与历史背景；旧文档中的“当前运行状态”以服务器实时日志和 2026-10-15 记录为准。
