/* pre_ctxid.c —— 九齿 FA 多设备错值的运行时修复（LD_PRELOAD shim）
 *
 * 背景：InfiniCore 用九齿（NineToothed）AOT 生成的 flash_attention_*.cpp 里，
 * 每设备 kernel 缓存的键取自 hipCtxGetId(NULL,&ctx_id)。在本集群的 DTK 栈上该调用
 * 对所有 device 返回同一个值，于是 static ThreadSafeUnorderedMap 只有一个条目：
 * 首个执行 FA 的设备加载的 CUmodule 被其余设备复用，result = 确定性错值。
 *
 * 修复：把 device id 混进键。ctx_id = base*131 + (dev+1)。
 * 构建：/opt/dtk/llvm/bin/clang -shared -fPIC -O2 -o pre_ctxid.so pre_ctxid.c -ldl
 * 使用：export LD_PRELOAD=$PWD/pre_ctxid.so:${LD_PRELOAD:-}
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>

typedef int hipError_t;
typedef void *hipCtx_t;
typedef hipError_t (*fn_getid)(hipCtx_t, unsigned long long *);
typedef hipError_t (*fn_getdev)(int *);

static fn_getid real_getid = 0;
static fn_getdev real_getdev = 0;

static void init_fns(void) {
    if (!real_getid) real_getid = (fn_getid)dlsym(RTLD_NEXT, "hipCtxGetId");
    if (!real_getdev) real_getdev = (fn_getdev)dlsym(RTLD_NEXT, "hipGetDevice");
}

hipError_t hipCtxGetId(hipCtx_t ctx, unsigned long long *id) {
    init_fns();
    hipError_t rc = real_getid ? real_getid(ctx, id) : -1;
    int dev = -1;
    if (real_getdev) (void)real_getdev(&dev);
    if (id) {
        unsigned long long base = *id;
        *id = base * 131ull + (unsigned long long)(dev + 1);
    }
    return rc;
}
