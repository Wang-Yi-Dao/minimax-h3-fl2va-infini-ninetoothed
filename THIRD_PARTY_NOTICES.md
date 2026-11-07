# 第三方组件与来源

本仓库**不重新分发**下列项目的完整源码，只以补丁/脚本形式给出我们做的改动。

| 组件 | 版本 / 来源 | 用途 | 许可 |
|---|---|---|---|
| InfiniLM | 集群源码树 `$HOME/src/InfiniLM.tgz`（上游仓库 HEAD 65deb91） | 推理引擎（examples/minimax_h3.py、rank worker、TP 分片） | 见上游仓库 |
| InfiniCore | `$HOME/src/` 上游源码树 + 本项目补丁 `src/infinicore_ninetoothed_hygon.patch` | 算子层、九齿后端注册、attention 桥接 | 见上游仓库 |
| nine-toothed | 0.26.0（海光定制，基于 triton 3.5.1） | AOT 生成 kernel（flash_attention / swiglu / rms_norm / mrope / timestep_embedding） | 见上游仓库 |
| PyTorch | 2.9.0+das.opt1.dtk2604 | 参考实现 / legacy 路径 / 评测 | BSD-3-Clause |
| diffusers / transformers | 0.40.0 / 4.x | 模型加载与 pipeline | Apache-2.0 |
| MiniMax-H3 权重 | 集群共享目录 `/public/home/aclsylqidf/shared/MiniMax-H3/diffusers` | 模型权重，**不随仓库分发** | 见官方模型许可 |
| scikit-image / imageio / PyAV | 评测环境 | 官方 compare 的 SSIM 与解码 | BSD / BSD / Apache-2.0 |

## 我们自己的文件

- `src/pre_ctxid.c`：LD_PRELOAD shim，改 `hipCtxGetId` 返回值按设备混键（不参与模型数值计算）
- `src/kb7.py`：源码级修复脚本（改写九齿生成的 `flash_attention_*.cpp` 的设备键）
- `src/mha_ninetoothed_hygon.cc`：InfiniCore 内的 HYGON fused attention 后端（`op::mha` 桥接）
- `src/infinicore_ninetoothed_hygon.patch`：为海光打开九齿的整套上游补丁（含 mrope 九齿实现）
- `jobs/`、`build/`：集群侧装配、出片、A/B、精度探针、评测与构建配方
