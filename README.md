# MiniMax-H3 FL2VA 在 InfiniLM / InfiniCore + 九齿（NineToothed）上的适配与优化

嵩山杯（2026 秋季高级人工智能）初赛作品代码仓库，团队账户 **xdzs2026_b0359**。

本仓库只包含**我们自己写的代码、补丁、作业脚本与实测结果**；上游 InfiniLM / InfiniCore / nine-toothed
以补丁形式给出，不复制其完整源码（体积与许可原因，见 `THIRD_PARTY_NOTICES.md`）。

## 一句话结果

在非 legacy 路径（InfiniLM 引擎 + InfiniCore 算子 + 九齿 AOT kernel）上，用 **8 卡 TP=8** 跑通
MiniMax-H3 **FL2VA**：768×928、124 帧、**50 步**、带音频，8 卡出片 **924 s**（单步 18.48 s，含加载共 985 s）。
官方评测脚本 `examples/minimax_h3_compare.py` 输出：

```
[result] MiniMax-H3 video comparison
  FL2VA
    Size:         PASS (PyTorch 768x928, InfiniLM 768x928)
    Length:       PASS (PyTorch 124 frames, InfiniLM 124 frames)
    Average SSIM: 36.976%
CMP_RC=0
```

修复前同一路径为 **7.818%**（画面退化成 16×16 级彩色块），官方示例同口径为 93.584%。

## 核心贡献

1. **九齿编译链移植**到海光 DCU（gfx936）：修复 10 类平台适配缺陷，官方算子测试全部通过
   （swiglu 126/126、rms_norm 108/108、timestep_embedding 18/18、mrope 36/36）。
2. **九齿 fused attention 接入**：在 InfiniCore 里新增 HYGON 后端（`src/mha_ninetoothed_hygon.cc`），
   head-major strided 视图直接喂 kernel，QK 缩放折进 fp32 softmax，三轴 RoPE 前置，
   head_dim=80 零填充到 128，因果分支保留兜底。
3. **定位并修复 TP>1 数值错误（本仓库重点）**：九齿 AOT 生成的
   `flash_attention_*.cpp` 用 `hipCtxGetId(NULL, &ctx_id)` 作为「每设备 kernel 缓存」的 key，
   在本集群 DTK 栈上该调用对所有 device 返回同一值，于是进程内**只有第一个执行 attention 的
   设备算对**，其余设备复用同一份已加载 kernel，得到确定性错值。
   - 源码级修法：`src/kb7.py`（把 key 换成 `hipGetDevice()+1`，并重编 InfiniCore）；
   - 交付用修法：`src/pre_ctxid.c`——`LD_PRELOAD` 运行时 shim，把 `hipCtxGetId` 的返回值按设备混键，
     **不改一行模型代码、不参与任何模型数值计算**。
   三层验证：算子级（两设备均 max_vs_torch=0.00129563）、TP2 截断模型（FA vs ATen cos=0.999820、
   TP2 vs TP1 cos=0.999988）、TP8 全模型规格与相似度（本文开头）。
4. **算子性能**：swiglu 1.56–1.67×、rms_norm 宽行 1.14–1.25×（相对参考实现），
   FA 在 S=27570 为 0.6×（尚无 split-K）。

## 目录

```
src/      pre_ctxid.c（设备键 shim）、kb7.py（源码级修复）、mha_ninetoothed_hygon.cc（HYGON FA 后端）、
          infinicore_ninetoothed_hygon.patch（为 HYGON 打开九齿的整套上游补丁）
jobs/     setup.py（集群侧解包/装配）、run_tp8_fl2va.sh（TP8 出片 + 官方 compare）、
          ab_native_vs_legacy.sh（同 seed 2 步 A/B）、ab_aten_fa_probe.sh（FA vs ATen 对照）、
          fa_precision_probe.py（FA 相对 torch SDPA 的精度随 S 变化）、ssim_report.py（同口径 SSIM 汇总）、
          submit_multi.mjs / submit_snap.mjs（平台批作业提交器）
build/    bld_infinicore.sh（InfiniCore 构建配方）、rebuild_and_repack.sh（改 kernel 后重生成 + 重打包）
results/  compare_nn30_50steps.txt（官方 compare 原始输出）、ab_2step_table.md（三次 2 步 A/B 复跑）
docs/     初赛实验报告.md、决赛实验报告.md、提交清单.md
figures/  平台作业日志截图（compare 输出）
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

## 已知限制 / 未完成

- **运行间非确定性**：同一 seed/参数下非 legacy 路径三次 2 步 A/B 的 SSIM 为 0.231 / 0.641 / 0.474
  （legacy 三次恒为 0.3995），输出来自集合通信与 attention kernel 的归约顺序差异，经扩散采样放大。
- **剩余相似度差距**：2 步时我们的首帧收敛比 legacy 慢，轨迹已分叉（见 `results/ab_2step_table.md`）；
  改用 ATen attention 只从 0.231 升到 0.271，说明差距不在 attention 精度，而在采样/条件注入侧。
- 九齿 FA 在 S=27570 相对 torch SDPA 的 mean 相对误差约 4.3%（S=4096 时 0.62%），缺 split-K。
- head_dim=80、因果分离 kernel 尚未原生实现（当前走零填充 128 与因果兜底）。
- 源码级修复（`src/kb7.py`）需重编 InfiniCore；本次交付的端到端结果是 LD_PRELOAD 方案跑出来的。

## 许可与来源

本项目自写部分（src/jobs/build 下的 shim、补丁脚本、博文式文档）按 MIT 提供；上游 InfiniLM、InfiniCore、
nine-toothed 遵循其各自许可，详见 `THIRD_PARTY_NOTICES.md`。仓库不含模型权重、数据集、集群凭据。
