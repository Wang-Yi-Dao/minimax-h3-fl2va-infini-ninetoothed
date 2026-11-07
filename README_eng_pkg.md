# MiniMax-H3 FL2VA · InfiniLM + InfiniCore + 九齿 海光 DCU 工程包

本包是《初赛/决赛实验报告》的可解包、可重建附件：包含在国产海光 DCU（gfx936，DTK/Rocm 兼容栈）上
跑通 **MiniMax-H3 首帧生视频（FL2VA）** 所需的补丁、脚本与复现步骤。所有数字均在本集群实测。

## 0. 环境（实测）

| 项 | 值 |
|---|---|
| 调度池 | `scnet-dcu`（unit8 节点，8 卡/granule） |
| 运行时 | scnet-dcu 环境 digest `524820f2…42a3aad`，Python 3.11.9，torch 2.9.0+das.opt1.dtk2604 |
| DTK | `/opt/dtk`（clang `/opt/dtk/llvm/bin/clang`、hipcc `/opt/dtk/bin/hipcc`、CUDA 兼容层 `/opt/dtk/cuda/cuda`） |
| 模型 | `/public/home/aclsylqidf/shared/MiniMax-H3/diffusers`（symlink 到作业工作区 `$W/model`） |
| 输入 | `$H/minimax_h3/input/1280X1280.JPEG`（首帧） |
| 参考片 | `$H/minimax_h3/cmp/reference_fl2va.mp4`（legacy PyTorch 路径 50 步 / 768x928 / 124 帧产物） |

## 1. 目录

```
src/pre_ctxid.c                      九齿 FA 多设备错值的运行时修复（LD_PRELOAD shim）
src/kb7.py                           等价的内核级修复：改写 AOT 生成物的设备缓存键（重编译路线用）
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
results/compare_nn30_50steps.txt     官方 compare 的原始输出与运行统计（36.976%）
results/ab_2step_table.md            2 步 A/B 三次复跑实测表（用于判定差距不是 50 步累积）
```

## 2. 复现步骤

### 2.1 一次性：装配 InfiniCore 与九齿 HYGON 后端
1. 对 `InfiniCore` 应用 `src/infinicore_ninetoothed_hygon.patch`（为 HYGON 打开 ninetoothed 生成：`build.py` 支持显式参数表、
   新增 `src/infiniop/ops/mrope/ninetoothed/{build.py,mrope.h}`、注册 `flash_attention` 的 ninetoothed 实现）。
2. 把 `src/mha_ninetoothed_hygon.cc` 放到 `src/infinicore/ops/multi_head_attention/hygon_nt/`（`[B,S,H,D]`→`[B,H,S,D]` 的
   `head_major_view` + `op::flash_attention_` 调用 + `plan_dispatcher().registerDevice(Device::Type::HYGON, …)`）。
3. 生成九齿 kernel 并编译：
   ```bash
   cd InfiniCore
   PYTHONPATH=$PWD/python:$PWD/src NINETOOTHED_CC=<hipcc> NTOPS_ONLY=flash_attention \
     python3 -u scripts/build_ntops.py        # 产物落到 InfiniCore/build/ninetoothed/
   bash ../build/bld_infinicore.sh              # xmake -y -j 24，产物 build/linux/x86_64/release/*.so
   bash ../build/rebuild_and_repack.sh          # 用 5 个 .so 重打 $H/artifacts/infinicore.tgz
   ```
   > xmake 不具备 kernel 生成能力，**改了 kernel 参数必须重跑 `build_ntops.py`**。
   > 本项目实测：xmake 工程的历史状态（`.xmake` 符号链接指向 `/dev/shm`、缺 packages、离线代理）会拦住重编，
   > 因此正式验证走下面的运行时 shim；`kb7.py` 是等价的源码级修法。

### 2.2 运行时修复（推荐，1 分钟，无需重编）
九齿生成物 `flash_attention_*.cpp` 的 launch 函数用
`unsigned long long ctx_id; (void)hipCtxGetId(NULL, &ctx_id);` 作为每设备 kernel 缓存的键：
在本集群 DTK 栈上该调用对所有 device 返回同值 ⇒ 首个设备的 CUmodule 被全体设备复用 ⇒ **每进程只有第一个跑
FA 的设备正确，其余设备返回确定性错值**（TP>1 时 rank≥1 全错）。修复只需预载一个 shim：

```bash
/opt/dtk/llvm/bin/clang -shared -fPIC -O2 -o pre_ctxid.so src/pre_ctxid.c -ldl
export LD_PRELOAD=$PWD/pre_ctxid.so:${LD_PRELOAD:-}
```

源码级等价修法：`python3 src/kb7.py <B> patch <InfiniCore/build/ninetoothed>`（把键换成 `hipGetDevice()+1`，幂等）后重编。

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
| 全流程（TP8、50 步、768x928、124 帧、带音频） | 成功，`run_secs=924`、稳态 18.5 s/step（legacy 657 s / 13.1 s per step） |
| 官方 compare 规格 | `Size: PASS`（768x928）、`Length: PASS`（124 帧）、h264 24fps、1 条音轨 |
| 官方 compare 相似度 | `Average SSIM: 36.976%`（修复前 7.818%；官方示例 93.584%） |
| 九齿 FA 修复前 | 第二个 device 起 FA 直接抛错（旧 .so）或返回确定性错值（b19/b20） |
| 算子级验证（修复后） | 两设备双向 `max_vs_torch = 0.00129563`（bf16 下限），与设备顺序无关 |
| 截断模型 TP2 vs TP1（修复后） | `video cos = 0.999988`、`audio cos = 0.999975`（修复前 0.9135 / 0.6240） |
| 九齿 FA vs torch SDPA（真实 S=27570，H=28） | `cos = 0.999082`、相对误差 `4.30e-2`（S=4096 时为 6.2e-3） |

## 4. 已知限制
- 单卡/TP2 装不下全模型（61.7 GB，TP2 每卡 ~30.9 GB 仍 `cudaMalloc` 失败），只能用 TP8。
- 相似度 36.976% 与官方示例 93.584% 的差距**不是 TP 数值问题**：修复后 TP2vsTP1 已达 bf16 噪声级；
  同 seed、同 prompt、同首帧的 2 步 A/B 中非 legacy 与 legacy 的 SSIM 只有 0.231（换成 ATen attention 为 0.271），
  说明差异来自采样/条件注入一侧（初始噪声、条件 token、sigma 调度、latent 布局），需进一步定位。
- 非 legacy 路径存在运行间非确定性：同 seed 2 步 A/B 三次为 0.231 / 0.641 / 0.474，输出字节数差一倍；
  legacy 三次对参考片恒定 0.3995 / 0.3995 / 0.4001 ⇒ 需把集合通信与 attention 归约做成确定性后再评估相似度上限。
- `H3_FORCE_ATEN_ATTN=1` 可把九齿 FA 换成 ATen 参考实现（`mha_ninetoothed_hygon.cc` 内的 env gate），
  用于 A/B 判读；S=27570 下不 OOM、不慢于九齿。
