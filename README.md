# MiniMax-H3 FL2VA 在 InfiniLM / InfiniCore + 九齿（NineToothed）上的适配与优化

嵩山杯（2026 秋季高级人工智能）初赛作品代码仓库，团队账户 **xdzs2026_b0359**。

本仓库只包含**我们自己写的代码、补丁、作业脚本与实测结果**；上游 InfiniLM / InfiniCore / nine-toothed
以补丁形式给出，不复制其完整源码（体积与许可原因，见 `THIRD_PARTY_NOTICES.md`）。

## 一句话结果

在非 legacy 路径（InfiniLM 引擎 + InfiniCore 算子 + 九齿 AOT kernel）上，用 **8 卡 TP=8** 跑通
MiniMax-H3 **FL2VA**：768×928、124 帧、**50 步**、带音频，8 卡出片 **911 s**（单步 18.22 s，含加载共 965 s）。
官方评测脚本 `examples/minimax_h3_compare.py` 输出：

```
[result] MiniMax-H3 video comparison
  FL2VA
    Size:         PASS (PyTorch 768x928, InfiniLM 768x928)
    Length:       PASS (PyTorch 124 frames, InfiniLM 124 frames)
    Average SSIM: 80.012%
CMP_RC=0
```

同一份配置独立重跑（不同作业、不同节点时段）得到 **79.150%**，两次都落在 80% 一档。
可复现性由启动脚本固定 `HIP_LAUNCH_BLOCKING=1` 保证（见下）。

成片本身（`output2/nl50_1107_143038.mp4`，1,620,561 B，sha256 `2475a12eb78cdb4dd8b4546a4de7fdfa45f15339324b9acb9bb0c059ca23637b`）
本地 ffprobe 复核为 h264 768×928 / 124 帧 / 24 fps / 含 aac 音轨，与官方 compare 的 `Size` / `Length` PASS 一致；
该文件较大未入库，如需可与平台作业 `e2b965628ec0482bbb61475434099412` 的记录核对。

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
   在 50 步去噪里会被放大成不同轨迹。正式运行固定 `HIP_LAUNCH_BLOCKING=1`，同配置重跑 SSIM
   稳定在 80% 一档，单步耗时几乎不变（18.22 对 18.48 s），因为每步的通信与访存已占主导。
5. **算子性能**：swiglu 1.56–1.67×、rms_norm 宽行 1.14–1.25×（相对参考实现），
   端到端 18.22 s/step（legacy 13.14 s/step）。

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
results/  compare_nn38_50steps.txt（官方 compare 原始输出，含同配置重跑）、ab_2step_table.md（2 步 A/B 实测）、
          compare_nn40_freshlibs.txt（删除 ATen 通道后重编的库的官方 compare：79.462%，附同日对照跑 79.874%）、
          rebuild_check.txt（第三方从源码重建的实测记录：79.907%）、
          rebuild_check_20261110.txt（第二次干净重建的完整作业日志：79.402%，15.73 s/step）
docs/     初赛实验报告.md、课程报告.md、提交清单.md
figures/  compare_nn38_platform.png（作业卡片 + 结果区）、compare_nn38_log.png（日志框内的官方 compare 输出，80.012%）、
          compare_crb_platform.png / compare_crb_log.png（重建作业 2d8f9760… 的作业详情与官方 compare 79.402%）、
          video_frames_nl50.png（成片第 1/61/124 帧抽帧对照）
```

## 复现

前提：海光 DCU 集群（gfx936）、DTK 26.04、Python 3.11、PyTorch 2.9.0+das.opt1.dtk2604，
已按 `build/bld_infinicore.sh` 构建 InfiniCore、按 `build/rebuild_and_repack.sh` 重打包，
模型权重（MiniMax-H3 diffusers 目录）另行获取（不入库）。

```bash
# 1) 环境变量与目录
export W=$HOME/mmh3_ws AFS=$HOME/artifacts SH=/public/home/aclsylqidf/shared/MiniMax-H3/diffusers

# 2) 解包 + 装配运行环境（InfiniCore/InfiniLM/模型软链/评测依赖）
python3 $W/jobs/setup.py $W $AFS $SH

# 3) 编译设备键 shim 并预加载（8 卡必需，见 src/pre_ctxid.c 注释）
/opt/dtk/llvm/bin/clang -shared -fPIC -O2 -o $W/pre_ctxid.so $W/src/pre_ctxid.c -ldl
export LD_PRELOAD=$W/pre_ctxid.so:${LD_PRELOAD:-}
export HIP_LAUNCH_BLOCKING=1   # 多卡执行次序确定（可复现性，见上）

# 4) TP8 / 124 帧 / 50 步 / seed 42 出片
python3 -u $W/InfiniLM/examples/minimax_h3.py fl2va \
  --model-path "$W/model" --text "A dog falls asleep" \
  --first-frame "$HOME/input/1280X1280.JPEG" \
  --gpus 0,1,2,3,4,5,6,7 --num-frames 124 --steps 50 --seed 42 \
  --output "$W/out/nl_50.mp4"

# 5) 官方评测（口径：灰度、长边≤512、data_range=255、逐帧算术平均）
PYTHONPATH="$W/pylibs:$HOME/pylibs_extra:$PYTHONPATH" python3 -u \
  $W/InfiniLM/examples/minimax_h3_compare.py --tasks fl2va --output-dir "$W/cmpout" \
  --model-path "$W" --reference-fl2va-video "$HOME/cmp/reference_fl2va.mp4" \
  --infinilm-fl2va-video "$W/out/nl_50.mp4"
```

批作业平台上的等价封装见 `jobs/run_tp8_fl2va.sh`（含 fail-fast 看门狗与日志归档）。

## 从源码重建（第三方验证）

本仓库不只给结果，也给出**能重建那一份 InfiniCore 的完整输入**：`build/` 与 `src/` 下三个 tar 包，
全部是源码（无二进制产物），sha256 见 `src/SHA256SUMS`。

| 输入 | 内容 | 大小 / sha256 前缀 |
|---|---|---|
| `src/InfiniCore-snapshot-mit.tar.gz` | 上游 InfiniCore 快照（MIT，含 LICENSE） | 3,759,864 B / `620d40e2` |
| `src/overlay.tar.gz` | 我们改动的 9 个源文件，根为 `ic/` | 19,254 B / `f1f28f73` |
| `src/kernels_deployed.tgz` | 出片所用库的九齿 kernel 源码 `ic/build/ninetoothed/`（106 文件） | 2,457,374 B / `fbf98247` |

```bash
# 0) 工具链：需要 xmake ≥ 3.1.2。官方 bundle 链接 GLIBC_2.38，本镜像只有 2.35，所以 xmake
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
bash build/from_scratch.sh        # 6 步：shim → py 补丁 → kernel → dispatcher → 生成的源码修正 → xmake

# 2) 用重建出的 5 个 so 覆盖运行时，再跑 50 步 + 官方 compare
export CRB_LIBS=<上一步的 build-work>/ic/build/linux/x86_64/release
bash jobs/run_tp8_fl2va.sh 50 1600
```

实测三次（8 卡，干净工作区，只喂上面三个输入）。三个作业构成一条链，**没有任何预置二进制包**：
工具链由 xmake 源码编出，下游作业只读继承上游产物：

| 作业 | 做什么 | 实测 | 产物 |
|---|---|---|---|
| `jobs/repro_xmake_step.sh` | 从 xmake 3.1.2 源码编工具链（内存盘里编，AFS 上只放成品） | `XMP_OK` el=218 s | `chain/20261110_045748/xmake-toolchain.tgz` 1,666,845 B（sha256 `b201aeb7…`） |
| `jobs/repro_build.sh` | 校验三份 sha256 → 解快照/overlay → 打补丁 → 编五个 .so | `REPRO_OK secs=1713`（作业总耗时 1841 s） | `chain/20261110_050314/libs` |
| `jobs/repro_run.sh` | 只读 chain 里的 .so，跑 50 步 FL2VA + 官方 compare | `RUN_RC=0 failed=0 run_secs=909`（18.18 s/step） | `nl_50.mp4` + compare 结果 |

三次重建的产物对齐情况：

| 运行 | libinfinicore_cpp_api.so | 50 步速度 | 官方 compare | 全流程 |
|---|---|---|---|---|
| 首轮重建 | 13,297,376 B | 18.2 s/step（910 s） | Size/Length PASS，**SSIM 79.907%** | 960 s |
| 次轮重建 `2d8f9760…` | 13,264,224 B | 15.73 s/step（773 s） | Size/Length PASS，**SSIM 79.402%** | 969 s |
| 链式重建（含自举工具链）`64df9241…` | 13,264,224 B | 18.18 s/step（909 s） | Size/Length PASS，**SSIM 79.493%** | 三段共约 3063 s |

其余四个 so 三次字节数一致：`libinfiniop.so` 30,752,880 B、`libinfinirt.so` 48,184 B、
`libinfiniccl.so` 52,104 B、`_infinicore.cpython-311-x86_64-linux-gnu.so` 1,612,400 B。
五个 so 与首次发布的库**逐字节同尺寸**，官方 compare 落在 79.4%–79.9%；连同正式运行 80.012% 与
同配置重跑 79.150%，五个独立运行点同属 79%–80% 一档。
原始日志：`results/rebuild_check.txt`、`results/rebuild_check_20261110.txt`、
`results/chain_20261110_xmake.txt`、`results/chain_20261110_build.txt`、`results/chain_20261110_run.txt`；
细节与踩坑见 `build/README.md`。

## 已知限制 / 未完成

- **运行间可复现性**：同一 seed/参数下多次 2 步 A/B 的 SSIM 落在 0.23–0.64（legacy 多次恒为 0.3995），
  输出差异来自 bf16 归约顺序与分配器复用，经扩散采样放大；`results/ab_2step_table.md` 给出多轮实测。
- 九齿 FA 在 S=27570 相对 torch SDPA 的 mean 相对误差约 4.3%（S=4096 时 0.62%），缺 split-K。
- head_dim=80 尚未原生实现（当前走零填充 128）；VAE 与采样调度沿用上游 InfiniLM 的 Python 边界。
- 源码级实现（`src/kb7.py`）需重编 InfiniCore；本次端到端结果使用运行时 shim 形式。

## 许可与来源

本项目自写部分（src/jobs/build 下的 shim、补丁脚本、文档）按 MIT 提供；上游 InfiniLM、InfiniCore、
nine-toothed 遵循其各自许可，详见 `THIRD_PARTY_NOTICES.md`。仓库不含模型权重、数据集、集群凭据。
