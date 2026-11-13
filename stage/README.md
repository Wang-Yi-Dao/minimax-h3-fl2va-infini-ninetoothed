# stage/ — 平台交互脚本

全部是 Node ESM 脚本，**在 `stage/` 目录下运行**（彼此用相对路径 import），只依赖 Node 18+ 内置模块。

## 先放 token

平台登录后，浏览器控制台执行：

```js
copy(sessionStorage['infinifabric-token@/compute/scheduler'])
```

把复制到的 43 个字符写进 `stage/token.txt`（单行，无引号）。token 约 1 小时过期，过期后
所有 API 返回 `401 AUTH_REQUIRED`，重新登录再取一次即可。

## 脚本

| 脚本 | 作用 | 用法 |
|---|---|---|
| `afscli.mjs` | 平台**个人存储**的上传/下载/列目录（不需要卡） | `node afscli.mjs ls minimax_h3` / `dl <远端> <本地>` / `up <本地> <远端>` |
| `afs.mjs` | 上面用到的存储 API 封装（`list/stat/download/upload`） | 被 import |
| `batch.mjs` | 批作业 API：`saveText/saveConfig/project/snapshot/preflight/submit/cancel` | 被 import |
| `gpu.mjs` | 通用 `api(path, body)`（带 Bearer 与超时） | 被 import |
| `probe.mjs` | 探活：`PLATFORM ready=… nodes_ready=… free=… VERDICT=USABLE/NOAUTH/UNAVAILABLE` | `node probe.mjs` |
| `submit_job.mjs` | 提交批作业：`JOBSH=<本地sh> NAME=<名> CNT=8 TMO=3300` | `JOBSH=jobs/run_tp8_fl2va.sh NAME=chain-c CNT=8 TMO=3300 node submit_job.mjs` |
| `watch.mjs` | 30 s 轮询作业，只在状态/字节变化时打印，终态打尾 6000 字符 | `node watch.mjs <jobId>` |
| `jstop.mjs` | 取消作业（报错立刻释放卡） | `node jstop.mjs <jobId>` |
| `res.mjs` | 看各节点空闲卡 | `node res.mjs` |

## 三条硬规矩

1. `CNT=8`：`CNT=1` 走临借通道，会报成误导性的 `PROJECT_ENVIRONMENT_CHANGED_OR_UNAVAILABLE`。
2. `TMO<=3300`：3600 直接 `403 LEASE_TIME_LIMIT`。
3. 项目内单文件 `<=16 KiB`（超了 403 `PROJECT_TEXT_LIMIT_16_KIB`，大 py 拆成独立项目文件；新建文件传 `expected_revision=0`）。
