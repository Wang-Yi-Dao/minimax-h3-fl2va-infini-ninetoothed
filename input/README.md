# input/ — 复现所需的两个输入

官方 `minimax_h3_compare.py` 的 SSIM 是**两段成对视频**的逐帧灰度相似度，所以参考片是必需输入；
出片脚本还需要一张首帧。这个目录放的就是这两样。

## reference_fl2va.mp4（已入库）

| 项 | 值 |
|---|---|
| 用途 | `--reference-fl2va-video`，官方 compare 的对照视频 |
| 规格 | 768×928、124 帧、24 fps、h264 |
| 来源 | 本项目的 **PyTorch legacy 后端**跑 FL2VA 得到（同 prompt「A dog falls asleep」、同首帧、同 50 步、seed 42），不是外部第三方素材 |
| 为什么必须同一条 | 换一段参考片，SSIM 立刻变：对同一成片 `nl50_1110_053831.mp4`，本文件给出 **79.493%**，而另一段 768×928 的 `fl2va.mp4` 只给 28.660% |

它同时是官方 compare「不给参考片就自己现场生成」那条分支的等价物：不给 `--reference-fl2va-video`
时，脚本会用 `--model-path` 与 `--first-frame` 现场跑一遍 legacy 生成器（需权重与 GPU，耗时约一次
legacy 全量生成），得到的参考片正是这一份。

## 1280X1280.JPEG（已入库）

| 项 | 值 |
|---|---|
| 用途 | `minimax_h3.py fl2va --first-frame`，出片首帧 |
| 规格 | 1280×1280 JPEG |
| 来源 | 平台个人存储 `$HOME/input/1280X1280.JPEG`（从平台存储接口取回，与本项目出片所用同一文件） |

查找顺序：仓库 `input/1280X1280.JPEG` → `$HOME/input/1280X1280.JPEG`；两者都没有时
`jobs/run_tp8_fl2va.sh` 打 `FIRST_FRAME_MISSING` 并 `exit 0`（看起来像「成功但没片」，别误判）。
