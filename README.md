# MiniMax-H3 FL2VA 在 InfiniLM / InfiniCore + 九齿（NineToothed）上的适配与优化

嵩山杯（2026 秋季高级人工智能）初赛作品代码仓库，队伍 **zzucs123**。

本仓库只包含**我们自己写的代码、补丁、作业脚本与实测结果**；上游 InfiniLM / InfiniCore / nine-toothed
以补丁形式给出，不复制其完整源码（体积与许可原因，见 `THIRD_PARTY_NOTICES.md`）。

## 一句话结果

在非 legacy 路径（InfiniLM 引擎 + InfiniCore 算子 + 九齿 AOT kernel）上，用 **8 卡 TP=8** 跑通
MiniMax-H3 **FL2VA**：768×928、124 帧、**50 步**、带音频，8 卡出片 **909 s**（单步 18.18 s，作业共 1004 s）。
官方评测脚本 `examples/minimax_h3_compare.py` 输出：

```
[result] MiniMax-H3 video comparison
  FL2VA
    Size:         PASS (PyTorch 768x928, InfiniLM 768x928)
    Length:       PASS (PyTorch 124 frames, InfiniLM 124 frames)
    Average SSIM: 79.493%
CMP_RC=0
```

这一份结果出自仓库当前代码的**从源码重建**：现场编出五个 `.so`，再用它们跑同一份配置出片并评测
（见「从源码重建」一节的实测表）；出片前的启动脚本固定 `HIP_LAUNCH_BLOCKING=1`（见下）。

成片本身（`output2/nl50_1110_053831.mp4`，1,615,256 B）
本地 ffprobe 复核为 h264 768×928 / 124 帧 / 24 fps / 含 aac 音轨，与 compare 的 `Size` / `Length` PASS 一致；
该文件较大未入库，如需可与平台作业 `64df924122964eacbd7787e2a4c76369` 的记录核对。

## 结果截图

从源码重建后跑 50 步出片，官方 `examples/minimax_h3_compare.py` 输出 `Average SSIM: 79.493%`、`CMP_RC=0`。
三个作业（xmake 工具链 → 编五个 .so → 出片+评测）的列表：

![重建链作业列表](figures/compare_chain_jobs.png)

最后一个作业的作业卡片与日志框（`Size` / `Length` PASS、`Average SSIM: 79.493%`、`CMP_RC=0`）：

![重建链出片作业与 compare 输出](figures/compare_chain_log.png)

成片抽帧（768×928 / 124 帧 / 24 fps / 含音轨）：

![成片抽帧](figures/video_frames_nl50.png)

## 核心贡献

1. **九齿编译链移植**到海光 DCU（gfx936）：处理 10 类平台适配差异，官方算子测试全部通过
   （swiglu 126/126、rms_norm 108/108、timestep_embedding 18/18、mrope 36/36）。
2. **九齿 fused attention 接入**：在 InfiniCore 里新增 HYGON 后端（`src/mha_ninetoothed_hygon.cc`），
   head-major strided 视图直接喂 kernel，QK 缩放折进 fp32 在线 softmax，三轴 RoPE 前置，
   head_dim=80 零填充到 128；因果注意力只出现在条件编码器，走 InfiniCore 自有 causal_softmax，不经过 op::mha。
3. **多设备 kernel 缓存键**：九齿 AOT 生成物为每个 kernel 变体维护一张静态表缓存已加载模块。
   HIP 的模块是 per-context 的，键必须显式携带设备维度，否则全进程只建一份模块、非首卡拿到别的
   设备上下文里的句柄。两种等价形式：
   - 源码级：`src/kb7.py`（把键换成 `hipGetDevice()+1`，幂等，随后重编 InfiniCore）；
   - 交付用：`src/pre_ctxid.c`——`LD_PRELOAD` 运行时 shim，把 `hipCtxGetId` 的返回值按设备混键，
     **不改一行模型代码、不参与任何模型数值计算**。

   三层验证：算子级（8 设备均 max_vs_torch=0.00129563，与设备执行顺序无关）、
   TP2 截断模型（video cos=0.999988、audio cos=0.999975）、
   真实 S=27570 单次前向 TP8 vs TP1（video cos=0.999985、audio cos=0.999963）。
4. **多卡执行次序确定**：8 个 rank 各有一条独立线程与一张设备，纯异步下发时 rank 间的读写次序
   取决于调度时序，同一份输入的重复执行会有低位差异——单次前向看不出来（bf16 噪声量级），
   在 50 步去噪里会被放大成不同轨迹。出片脚本固定 `HIP_LAUNCH_BLOCKING=1`；50 步实测 18.18 s/step，
   耗时由每步的通信与访存主导，同步开关对总时长几乎无影响。
5. **算子性能**：swiglu 1.56–1.67×、rms_norm 宽行 1.14–1.25×（相对参考实现），
   端到端 18.18 s/step（50 步 909 s）。

## 目录

```
src/      pre_ctxid.c（设备键 shim）、kb7.py（源码级实现）、mha_ninetoothed_hygon.cc（HYGON FA 后端）、
          infinicore_ninetoothed_hygon.patch（为 HYGON 打开九齿的整套上游补丁）
jobs/     setup.py（集群侧解包/装配）、run_tp8_fl2va.sh（TP8 出片 + 官方 compare）、
          ab_native_vs_legacy.sh（同 seed 2 步 A/B）、
          fa_precision_probe.py（FA 相对 torch SDPA 的精度随 S 变化）、ssim_report.py（同口径 SSIM 汇总）、
          submit_multi.mjs / submit_snap.mjs（平台批作业提交器）、
          repro_xmake_step.sh（从源码自举 xmake 工具链，可反复续跑）、
          repro_build.sh（校验 sha256 → 从三份源码重建五个 .so）、repro_run.sh（50 步出片 + 官方 compare）
build/    bld_infinicore.sh（InfiniCore 构建配方）、rebuild_and_repack.sh（改 kernel 后重生成 + 重打包）、
          from_scratch.sh（只喂三份源码输入的六步构建）、reproduce.sh（一条命令跑通 from_scratch 并校验产物）、
          xmake_build.sh（xmake 源码编译封装）
input/    reference_fl2va.mp4（官方 compare 的参考片，768×928 / 124 帧 / 24 fps / 1.5 MB，
          由本项目 legacy PyTorch 路径产出）、1280X1280.JPEG（出片首帧，146,182 B）；说明见 input/README.md
stage/    afs.mjs / afscli.mjs（平台个人存储上传下载）、batch.mjs / gpu.mjs / submit_job.mjs /
          probe.mjs / watch.mjs / jstop.mjs / res.mjs（批作业提交、探活、轮询、取消、查空闲卡）
results/  chain_20261110_xmake.txt / _build.txt / _run.txt（从源码重建的三段链完整作业日志，末段含官方
          compare 原始输出 79.493%）、ab_2step_table.md（2 步 A/B 实测）
docs/     初赛实验报告.md、课程报告.md、提交清单.md、从零复现指令流程.md（第三方账户从零复现的逐条指令）
figures/  compare_chain_jobs.png / compare_chain_log.png（重建链三个作业的列表与日志，官方 compare 79.493%）、
          video_frames_nl50.png（成片第 1/61/124 帧抽帧对照）
```

## 复现

前提：海光 DCU 集群（gfx936）、DTK 26.04、Python 3.11、PyTorch 2.9.0+das.opt1.dtk2604，
InfiniCore 五个 `.so` 按下一节「两条路」任一条取得（A 从源码编译 / B 取预编译产物），
模型权重（MiniMax-H3 diffusers 目录）另行获取（不入库）。

两个输入都随仓库提供：**参考片** `input/reference_fl2va.mp4`（1.5 MB，768×928/124 帧/24 fps，
sha256 `e4e470e0…`）是官方 compare 的对照视频，换一段结果立刻变（同一成片：本文件 79.493%，
另一段 768×928 的片只给 28.660%）；**首帧** `input/1280X1280.JPEG`（146,182 B，sha256 `e5b7ac83…`）
由脚本从仓库优先读取，缺失时回退 `$HOME/input/1280X1280.JPEG`，再没有就报 `FIRST_FRAME_MISSING`。
详见 `input/README.md`。

**新建账户（看不见本项目任何存储）请直接按 [`docs/从零复现指令流程.md`](docs/从零复现指令流程.md) 走**：
第 0.5 步把 Release `assets-v1` 里的 wheels、InfiniLM 源码、预编译库下载后上传到自己的 `$HOME`，
再依次提交三个批作业；权重走平台公共路径 `/public/home/aclsylqidf/shared/MiniMax-H3/diffusers`。

```bash
# 1) 环境变量与目录
export W=$HOME/mmh3_ws AFS=$HOME/artifacts SH=/public/home/aclsylqidf/shared/MiniMax-H3/diffusers
REPO=$(pwd)   # 本仓库根（含 input/reference_fl2va.mp4 与 src/）

# 2) 解包 + 装配运行环境（InfiniCore/InfiniLM/模型软链/评测依赖）
python3 $W/jobs/setup.py $W $AFS $SH

# 3) 编译设备键 shim 并预加载（8 卡必需，见 src/pre_ctxid.c 注释）
/opt/dtk/llvm/bin/clang -shared -fPIC -O2 -o $W/pre_ctxid.so $W/src/pre_ctxid.c -ldl
export LD_PRELOAD=$W/pre_ctxid.so:${LD_PRELOAD:-}
export HIP_LAUNCH_BLOCKING=1   # 多卡执行次序确定（可复现性，见上）

# 4) TP8 / 124 帧 / 50 步 / seed 42 出片
python3 -u $W/InfiniLM/examples/minimax_h3.py fl2va \
  --model-path "$W/model" --text "A dog falls asleep" \
  --first-frame "$REPO/input/1280X1280.JPEG" \
  --gpus 0,1,2,3,4,5,6,7 --num-frames 124 --steps 50 --seed 42 \
  --output "$W/out/nl_50.mp4"

# 5) 官方评测（口径：灰度、长边≤512、data_range=255、逐帧算术平均）
PYTHONPATH="$W/pylibs:$HOME/pylibs_extra:$PYTHONPATH" python3 -u \
  $W/InfiniLM/examples/minimax_h3_compare.py --tasks fl2va --output-dir "$W/cmpout" \
  --model-path "$W" --reference-fl2va-video "$REPO/input/reference_fl2va.mp4" \
  --infinilm-fl2va-video "$W/out/nl_50.mp4"
```

批作业平台上的等价封装见 `jobs/run_tp8_fl2va.sh`（含 fail-fast 看门狗与日志归档）。

## 两条路：从源码编译 / 直接用预编译产物

两条路的终点完全一样：五个 `.so` 落到同一个目录，再用同一套命令跑 50 步出片与官方 compare。
按环境选一条——A 完全从源码起（本仓库实测过的那条链），B 直接取我们编好的二进制（省掉约半小时编译）。

| 路线 | 做什么 | 实测耗时 | 产物来源 |
|---|---|---|---|
| **A 从源码编译** | `bash build/reproduce.sh $HOME/tmp/repro-build all` | 工具链 219.099 s + 编译 1713 s + 出片 1004.498 s（三段链） | 本仓库 `src/` 三份源码输入 |
| **B 用预编译产物** | 从 Release `assets-v1` 的 `prebuilt-libs.tgz` 取五个 `.so` 放进工作目录 | 下载 13 MB | 同一条链产出的二进制 |

### 路线 A：从源码编译（本仓库实测）

三份输入全部是源码（无二进制产物），sha256 见 `src/SHA256SUMS`，`build/reproduce.sh` 把校验、xmake 工具链、
InfiniCore 五个 `.so` 串成一条命令：

```bash
bash build/reproduce.sh $HOME/tmp/repro-build all   # 有可写本地盘可换 /dev/shm/repro-build（更快）
# 产物：$HOME/tmp/repro-build/ic/build/linux/x86_64/release 下五个 .so
# 不传工作目录时脚本按 $LOCAL_SCRATCH → /tmp → $HOME/tmp 自动探活
```

| 输入 | 内容 | 大小 / sha256 前缀 |
|---|---|---|
| `src/InfiniCore-snapshot-mit.tar.gz` | 上游 InfiniCore 快照（MIT，含 LICENSE） | 3,759,864 B / `620d40e2` |
| `src/overlay.tar.gz` | 我们改动的 9 个源文件，根为 `ic/` | 19,254 B / `f1f28f73` |
| `src/kernels_deployed.tgz` | 出片所用库的九齿 kernel 源码 `ic/build/ninetoothed/`（106 文件） | 2,457,374 B / `fbf98247` |

集群不能直连 GitHub / PyPI 时，`reproduce.sh` 用 `GH_MIRROR` 换镜像（缺省 `https://ghproxy.net/`），
也可预置 `XM_TGZ`、`PYBIND11_ROOT` 免下载。pybind11 3.1.0 的获取按 pypi 直连 → 清华镜像 → 阿里镜像 →
GitHub tag tarball 依次回退，解包后校验 `include/pybind11/pybind11.h` 才继续，四源都失败才报 `PYBIND11_DL_FAIL`。

手工分段（也就是三段链，为什么这么拆见本节末尾的实测表）：

```bash
# 0) 工具链：需要 xmake >= 3.1.2。官方 bundle 链接 GLIBC_2.38，本镜像只有 2.35，所以 xmake
#    必须从源码编；又因为计算节点没有可写本地盘、AFS 小文件极慢（实测 cp -a 4008 个文件
#    要 732 s、make 约 16 个对象/分钟、单作业上限 3200 s），这一段单独成链：反复跑
#    jobs/repro_xmake_step.sh 直到打印 XMP_OK，产出 chain/<ts>/xmake-toolchain.tgz
#    （安装布局 bin/xmake + share/xmake，下游直接解包即用）。
export XM_PREBUILT=<上一步产出的 xmake-toolchain.tgz>

# 1) 取三个输入，按 build/README.md 设好 DTK / wheels / pybind11
export IC_TGZ=$PWD/src/InfiniCore-snapshot-mit.tar.gz
export OVL_TGZ=$PWD/src/overlay.tar.gz
export CRB_KERNELS_TGZ=$PWD/src/kernels_deployed.tgz
export DTK=/opt/dtk J=32 ARCH=gfx936 PHASE=all
bash build/from_scratch.sh        # 6 步：shim -> py 补丁 -> kernel -> dispatcher -> 生成的源码修正 -> xmake
```

### 路线 B：直接用预编译产物

Release：<https://github.com/Wang-Yi-Dao/minimax-h3-fl2va-infini-ninetoothed/releases/tag/v1.2-prebuilt>
（五个文件就是路线 A 那条链编出来的同一批二进制，AFS 上 `chain/20261110_050314/libs`）。

```bash
R=https://github.com/Wang-Yi-Dao/minimax-h3-fl2va-infini-ninetoothed/releases/download/v1.2-prebuilt
D=$HOME/tmp/repro-build/ic/build/linux/x86_64/release
mkdir -p "$D"
for f in _infinicore.cpython-311-x86_64-linux-gnu.so libinfiniop.so \
         libinfinicore_cpp_api.so libinfinirt.so libinfiniccl.so; do
  curl -fL --retry 3 -o "$D/$f" "$R/$f"     # 直连不通时前面加 https://ghproxy.net/
done
sha256sum "$D"/*.so                          # 对照下表，五个都要对上
```

| 文件 | 大小 | sha256 |
|---|---|---|
| `libinfiniop.so` | 30,752,880 B | `a3847eeaba99fd23d7e2dce894a0002a9e2835cd8d82340bb46f6ebeb78b46fc` |
| `libinfinicore_cpp_api.so` | 13,264,224 B | `b39722882f7846b0ffd4016414803ad79a72ad30604f3d6176b9ba168c37dcb9` |
| `_infinicore.cpython-311-x86_64-linux-gnu.so` | 1,612,400 B | `41d649f73e209c42ee4949761eb63ca75eeff918d49ce89c5aba594f95f3dc04` |
| `libinfiniccl.so` | 52,104 B | `81183241721bbacca06afa8a269ec3c2bbe80a09f74527a6690d53d4a35d2bf3` |
| `libinfinirt.so` | 48,184 B | `1993bdeba63c137758fc2cb9d8ae772d6fdeb94999de260373ab52e1f4c7a467` |

### 两条路之后（相同）

```bash
# 2) 用这 5 个 `.so` 覆盖运行时，再跑 50 步 + 官方 compare
export CRB_LIBS=$HOME/tmp/repro-build/ic/build/linux/x86_64/release
bash jobs/run_tp8_fl2va.sh 50 2300
```

### 路线 A 的实测（8 卡，干净工作区，只喂三份源码输入）

三个作业构成一条链，**没有任何预置二进制包**：工具链由 xmake 源码编出，下游作业只读继承上游产物。

| 作业 | 做什么 | 实测 | 产物 |
|---|---|---|---|
| `jobs/repro_xmake_step.sh` | 从 xmake 3.1.2 源码编工具链（内存盘里编，AFS 上只放成品） | `XMP_OK`，作业 219.099 s / 1753 卡秒 | `chain/20261110_045748/xmake-toolchain.tgz` 1,666,845 B（sha256 `b201aeb7...`） |
| `jobs/repro_build.sh` | 校验三份 sha256 -> 解快照/overlay -> 打补丁 -> 编五个 `.so` | `REPRO_OK secs=1713`，作业 1841.813 s / 14735 卡秒 | `chain/20261110_050314/libs` |
| `jobs/repro_run.sh` | 只读 chain 里的 .so，跑 50 步 FL2VA + 官方 compare | `RUN_RC=0 failed=0`，18.18 s/step（909 s），作业 1004.498 s / 8036 卡秒 | `nl_50.mp4` + compare 结果 |

产物与结果：

| 项目 | 实测值 |
|---|---|
| 五个 `.so` | 就是路线 B 表里那五行（尺寸与 sha256 一致） |
| 官方 compare | `Size: PASS (768x928)`、`Length: PASS (124 frames)`、**`Average SSIM: 79.493%`**、`CMP_RC=0` |

一条实情：chain-b1 提交时脚本还没认链上工具链（`XM_PREBUILT` 当时只在 `STAGE=build` 生效），
所以它在作业内又自举了一遍 xmake（内存盘上约 2 分钟，日志里可见 `--- download xmake 3.1.2 source tree`），
随后照常编出五个 `.so`；整条链仍是 100% 从源码起。该分支此后已修好（任何 `STAGE` 都优先用链上
`xmake-toolchain.tgz`），慢盘环境下用它可省掉重复自举。

原始日志：`results/chain_20261110_xmake.txt`、`results/chain_20261110_build.txt`、`results/chain_20261110_run.txt`；
细节与踩坑见 `build/README.md`。

## 已知限制 / 未完成

- **运行间可复现性**：同一 seed/参数下多次 2 步 A/B 的 SSIM 落在 0.23–0.64，
  输出差异来自 bf16 归约顺序与分配器复用，经扩散采样放大；`results/ab_2step_table.md` 给出多轮实测。
- 九齿 FA 在 S=27570 相对 torch SDPA 的 mean 相对误差约 4.3%（S=4096 时 0.62%），缺 split-K。
- head_dim=80 尚未原生实现（当前走零填充 128）；VAE 与采样调度沿用上游 InfiniLM 的 Python 边界。
- 源码级实现（`src/kb7.py`）需重编 InfiniCore；本次端到端结果使用运行时 shim 形式。

## 许可与来源

本项目自写部分（src/jobs/build 下的 shim、补丁脚本、文档）按 MIT 提供；上游 InfiniLM、InfiniCore、
nine-toothed 遵循其各自许可，详见 `THIRD_PARTY_NOTICES.md`。仓库不含模型权重、数据集、集群凭据。
