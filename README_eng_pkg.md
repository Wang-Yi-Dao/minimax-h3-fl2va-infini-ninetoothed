# MiniMax-H3 FL2VA · InfiniLM + InfiniCore + 九齿 海光 DCU 工程包

本包是《初赛/决赛实验报告》的可解包、可重建附件：在国产海光 DCU（gfx936，DTK/Rocm 兼容栈）上用
**InfiniLM（推理引擎）+ InfiniCore（算子层）+ 九齿 NineToothed（算子编译）** 的非 legacy 路径跑通
**MiniMax-H3 首帧生视频（FL2VA）**，输出 768×928 / 124 帧 / 50 步 / 带音频，官方评测脚本给出
`Size: PASS`、`Length: PASS`、`Average SSIM: 80.012%`（同配置独立重跑 79.150%）。包内所有数字均在本集群实测。

## 0. 环境（实测）

| 项 | 值 |
|---|---|
| 调度池 | `scnet-dcu`（unit8 节点，8 卡/granule） |
| 运行时 | scnet-dcu 环境 digest `524820f2…42a3aad`，Python 3.11.9，torch 2.9.0+das.opt1.dtk2604 |
| DTK | `/opt/dtk`（clang `/opt/dtk/llvm/bin/clang`、hipcc `/opt/dtk/bin/hipcc`、CUDA 兼容层 `/opt/dtk/cuda/cuda`） |
| 九齿 | `ninetoothed 0.26.0` + 海光定制 `triton 3.5.1` |
| 模型 | `/public/home/aclsylqidf/shared/MiniMax-H3/diffusers`（symlink 到作业工作区 `$W/model`） |
| 输入 | `$H/minimax_h3/input/1280X1280.JPEG`（首帧） |
| 参考片 | `$H/minimax_h3/cmp/reference_fl2va.mp4`（legacy PyTorch 路径 50 步 / 768x928 / 124 帧产物） |
| transformers / diffusers | 5.14.1 / 0.40.0（精简作业环境不含 torchvision，见下） |

作业镜像是**不含 torchvision** 的精简环境：`jobs/setup.py` 在解包依赖后对库文件做四处幂等适配，仅作用于 processor 的构造与降级分支，不参与模型数值计算——
① `transformers/processing_utils.py` 在 torchvision 不可用时跳过 `video_processor` 子处理器；
② `transformers/models/auto/video_processing_auto.py` 的查表补 `is_torchvision_available()` 守卫与空表判空；
③ `diffusers` 的 `encoders.py` 因 transformers 5.x 移除 `create_mm_token_type_ids`，改按 Qwen3-VL 约定由 token id 推导 `mm_token_type_ids`（图像→1、视频→2）；
④ 模型目录 `processor/preprocessor_config.json` 的 `image_processor_type` 落实为同目录可导入的 `Qwen2VLImageProcessor`。
另外 `transformers 5.0.0` 与 `torchvision` 被显式跳过，`flash_attn` 只保留供应商 `.so`（其 Python 包依赖未随镜像提供的 einops）。

## 1. 目录

```
src/pre_ctxid.c                      多设备 kernel 缓存键的运行时形式（LD_PRELOAD shim，20 行）
src/kb7.py                           多设备 kernel 缓存键的内核级形式（改写 AOT 生成物的缓存键，重编译路线用）
src/mha_ninetoothed_hygon.cc         九齿 fused attention 的海光 HYGON 后端（MultiheadAttention plan 注册）
src/infinicore_ninetoothed_hygon.patch  为 HYGON 打开九齿的整套 InfiniCore 补丁（含 mrope/flash_attention 的 ninetoothed 生成器）
jobs/setup.py                        作业侧环境装配：解包预编译 InfiniCore/InfiniLM、装 pylibs、建模型 symlink、patch transformers
jobs/run_tp8_fl2va.sh                正式出片：TP8、768x928、124 帧、N 步、带音频 + 官方 compare
jobs/ab_native_vs_legacy.sh          同 seed / 同 prompt A/B：非 legacy（InfiniLM）vs legacy（PyTorch），任意步数
jobs/ab_aten_fa_probe.sh             A) 九齿 FA 在真实 S=27570 下 vs torch SDPA 的精度；B) 全模型改用 ATen attention 出片
jobs/fa_precision_probe.py           A 段的 python 主体
jobs/ssim_report.py                  视频相似度矩阵（灰度 + 长边≤512 + 双线性 + data_range=255，与官方 compare 同口径）
jobs/submit_multi.mjs                提交器：上传工程文件→存 config→snapshot→preflight→submit（首次建项目文件）
jobs/submit_snap.mjs                 提交器：直接复用已有 snapshot 提交（快照额度耗尽时的唯一通路）
build/bld_infinicore.sh              InfiniCore 构建配方（xmake + 九齿 AOT 生成）
build/rebuild_and_repack.sh          重编后重打包 `infinicore.tgz`（只替换 5 个 .so）
results/compare_nn38_50steps.txt     官方 compare 的原始输出与运行统计（80.012%，含同配置重跑 79.150%）
results/ab_2step_table.md            2 步 A/B 多轮复跑实测表（衡量同一实现的可复现性）
```

## 2. 复现步骤

### 2.1 一次性：装配 InfiniCore 与九齿 HYGON 后端
1. 对 `InfiniCore` 应用 `src/infinicore_ninetoothed_hygon.patch`（为 HYGON 打开 ninetoothed 生成：`build.py` 支持显式参数表、
   新增 `src/infiniop/ops/mrope/ninetoothed/{build.py,mrope.h}`、注册 `flash_attention` 的 ninetoothed 实现）。
2. 把 `src/mha_ninetoothed_hygon.cc` 放到 `src/infinicore/ops/multi_head_attention/hygon_nt/`（`[B,S,H,D]`→`[B,H,S,D]` 的
   `head_major_view` + `op::flash_attention_` 调用 + `plan_dispatcher().registerDevice(Device::Type::HYGON, …)`）。
3. 生成九齿 kernel 并编译（**xmake 不具备 kernel 生成能力**，改了 kernel 参数必须重跑 `build_ntops.py`）：
   ```bash
   cd InfiniCore
   PYTHONPATH=$PWD/python:$PWD/src NINETOOTHED_CC=<hipcc> NTOPS_ONLY=flash_attention \
     python3 -u scripts/build_ntops.py        # 产物落到 InfiniCore/build/ninetoothed/
   bash ../build/bld_infinicore.sh              # xmake -y -j 24，产物 build/linux/x86_64/release/*.so
   bash ../build/rebuild_and_repack.sh          # 用 5 个 .so 重打 $H/artifacts/infinicore.tgz
   ```

### 2.2 多设备 kernel 缓存键（部署必做，无需重编）
九齿 AOT 生成物为每个 kernel 变体维护一张静态表缓存已加载的模块。HIP 的模块是 per-context 的，
键必须显式携带设备维度，否则全进程只建一份模块、非首卡拿到别的设备上下文里的句柄。运行时形式只需预载一个 shim：

```bash
/opt/dtk/llvm/bin/clang -shared -fPIC -O2 -o pre_ctxid.so src/pre_ctxid.c -ldl
export LD_PRELOAD=$PWD/pre_ctxid.so:${LD_PRELOAD:-}
```

内核级等价形式：`python3 src/kb7.py <B> patch <InfiniCore/build/ninetoothed>`（把键换成 `hipGetDevice()+1`，幂等）后重编。
两者语义一致，都只影响模块缓存的分桶，不改变任何 kernel 数学。

### 2.3 出片 + 官方 compare
```bash
# 作业脚本入口（平台批作业）：cmp1.sh（= jobs/run_tp8_fl2va.sh），参数：<steps> <budget_secs>
python3 -u examples/minimax_h3.py fl2va --model-path $W/model \
  --text "A dog falls asleep" --first-frame $H/input/1280X1280.JPEG \
  --gpus 0,1,2,3,4,5,6,7 --num-frames 124 --steps 50 --seed 42 --output $W/out/nl_50.mp4
python3 -u examples/minimax_h3_compare.py --tasks fl2va --output-dir $W/cmpout --model-path $W \
  --reference-fl2va-video $H/cmp/reference_fl2va.mp4 --infinilm-fl2va-video $W/out/nl_50.mp4
```
> compare 依赖 `scikit-image`/`imageio`/`av`：需 `export PYTHONPATH=$W/pylibs:$H/pylibs_extra:$W/InfiniCore/python:$W/InfiniLM/python`。

## 3. 实测结果（可直接引用）

| 项 | 结果 |
|---|---|
| 全流程（TP8、50 步、768x928、124 帧、带音频） | 成功，`run_secs=911`、稳态 18.22 s/step（legacy 657 s / 13.14 s per step） |
| 多卡执行次序 | 正式运行固定 `HIP_LAUNCH_BLOCKING=1`；同配置重跑 SSIM 79.150%，逐次前向张量指纹一致 |
| 官方 compare 规格 | `Size: PASS`（768x928）、`Length: PASS`（124 帧）、h264 24fps、1 条音轨 |
| 官方 compare 相似度 | `Average SSIM: 80.012%`（重跑 79.150%） |
| 算子级多设备一致性 | 8 设备全部 `max_vs_torch = 0.00129563`（bf16 下限），与设备执行顺序无关 |
| 截断模型 TP2 vs TP1 | `video cos = 0.999988`、`audio cos = 0.999975` |
| 真实 S=27570 单次前向 TP8 vs TP1 | `video cos = 0.999985`、`audio cos = 0.999963` |
| 九齿 FA vs torch SDPA（真实 S=27570，H=28） | `cos = 0.999082`、相对误差 `4.30e-2`（S=4096 时为 6.2e-3） |

## 4. 已知限制
- 单卡/TP2 装不下全模型（61.7 GB，TP2 每卡约 30.9 GB 仍 `cudaMalloc` 失败），只能 TP8。
- 九齿 FA 未做 split-K，端到端比 legacy 慢约 39%（18.22 vs 13.14 s/step）。
- 九齿 FA 的 kernel 变体只覆盖 bf16，概率—值乘法在 bf16 精度下进行，是当前相似度的主要损失项（改 ATen 参考路径可得 95.646%，但该路径用 torch 计算、不符合赛题要求，仅作诊断上界）。
- 同一实现多次出片之间存在低位数值漂移（同 seed 2 步 A/B 多轮落在 0.23–0.64），属 bf16 归约顺序与分配器复用的
  可复现性上限；`results/ab_2step_table.md` 给出多轮实测。
- `H3_FORCE_ATEN_ATTN=1` 可把九齿 FA 换成 ATen 参考实现（`mha_ninetoothed_hygon.cc` 内的 env gate），
  用于 A/B 判读；S=27570 下不 OOM、不慢于九齿。
