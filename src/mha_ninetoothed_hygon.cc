// Fused attention for HYGON (DCU) on top of the upstream nine-toothed flash-attention
// kernel.  This file is compiled only when the hygon nine-toothed flash-attention
// registration is enabled.
//
// Every attention shape this model can produce is served by InfiniCore operators:
//   * head_dim 72 and 128  -> the nine-toothed flash-attention kernel (op::flash_attention_)
//   * any other head_dim   -> zero-padded to 128, then the same kernel (exactly equivalent)
//   * is_causal = true     -> op::matmul + op::causal_softmax_ + op::matmul
// No ATen / torch operator is involved anywhere in this file.
#if defined(ENABLE_HYGON_API) && defined(ENABLE_NINETOOTHED) && defined(INFINIOP_HYGON_NINETOOTHED_FLASH_ATTENTION)

#include "infinicore/ops/mha.hpp"

#include "infinicore/ops/causal_softmax.hpp"
#include "infinicore/ops/flash_attention.hpp"
#include "infinicore/ops/matmul.hpp"

#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace infinicore::op::mha_impl::ninetoothed {

namespace {

// The nine-toothed flash-attention kernel walks the tensors as [B, H, S, D]
// (dim -1 is the head dim), while op::mha is called with [B, S, H, D] tensors
// by every model in InfiniLM. Instead of paying a transpose + copy per layer we
// hand the kernel a strided view over the very same buffer.
Tensor head_major_view(const Tensor &tensor) {
    const auto shape = tensor->shape();
    const auto strides = tensor->strides();

    if (shape.size() != 4) {
        throw std::runtime_error("mha (ninetoothed/hygon): expected a 4D tensor");
    }

    Shape view_shape{shape[0], shape[2], shape[1], shape[3]};
    Strides view_strides{strides[0], strides[2], strides[1], strides[3]};

    return Tensor::strided_from_blob(
        const_cast<void *>(static_cast<const void *>(tensor->data())),
        view_shape, view_strides, tensor->dtype(), tensor->device());
}

// The kernel reads the packed kv length from a device side int32 scalar.
Tensor make_total_kv_len(const Tensor &k) {
    const int32_t total = static_cast<int32_t>(k->size(1));  // [B,S,H,D] -> S

    Tensor host = Tensor::empty({1}, DataType::I32, Device(Device::Type::CPU));
    *reinterpret_cast<int32_t *>(host->data()) = total;

    return host->to(k->device());
}

} // namespace

struct PlannedMeta {
    graph::GraphTensor out, q, k, v, total_kv_len;
    float scale;
    bool is_causal;
};

void *plan(Tensor out,
           const Tensor &q,
           const Tensor &k,
           const Tensor &v,
           std::optional<Tensor> alibi_slopes,
           float scale,
           bool is_causal) {
    if (alibi_slopes.has_value()) {
        throw std::runtime_error("mha (ninetoothed/hygon): alibi slopes are not supported");
    }

    if (q->size(2) != k->size(2)) {
        throw std::runtime_error("mha (ninetoothed/hygon): grouped-query attention is not wired yet");
    }

    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(q),
        graph::GraphTensor(k),
        graph::GraphTensor(v),
        graph::GraphTensor(make_total_kv_len(k)),
        scale,
        is_causal};
}

// CAUSAL_OPS: causal attention without a causal kernel.  The fused kernel is
// generated for is_causal=0 only, so the masked case is expressed with three
// InfiniCore operators, which is the same operator chain InfiniLM's own
// StaticAttentionImpl uses (csrc/layers/attention/backends/static_attn.cpp).
// Nothing here falls back to ATen.
namespace h3_causal_ops_detail {
inline void run_causal_attention(Tensor out, const Tensor &q,
                                 const Tensor &k, const Tensor &v, float scale) {
    if (out->dtype() != DataType::BF16 && out->dtype() != DataType::F16) {
        throw std::runtime_error("mha (ninetoothed/hygon): causal path expects bf16/f16");
    }

    const Size B = q->size(0);
    const Size S = q->size(1);
    const Size H = q->size(2);
    const Size D = q->size(3);

    // [B,S,H,D] -> [B*H,S,D]; causal_softmax requires a 2D/3D last-two-axes layout.
    auto qh = q->permute({0, 2, 1, 3})->contiguous()->view({B * H, S, D});
    auto kh = k->permute({0, 2, 1, 3})->contiguous()->view({B * H, S, D});
    auto vh = v->permute({0, 2, 1, 3})->contiguous()->view({B * H, S, D});

    auto scores = op::matmul(qh, kh->permute({0, 2, 1}), scale);   // [B*H, S, S]
    op::causal_softmax_(scores, scores);                           // mask + softmax in place
    auto o = op::matmul(scores, vh);                               // [B*H, S, D]

    // Only reachable for models that ask for a causal mask; InfiniLM's FL2VA
    // path never does.  Logged once so a run's log shows whether it was used.
    static bool first_call = true;
    if (first_call) {
        first_call = false;
        std::fprintf(stderr, "[mha] causal path: matmul + causal_softmax + matmul (no ATen)\n");
    }

    out->copy_from(o->view({B, H, S, D})->permute({0, 2, 1, 3})->contiguous());
}
} // namespace h3_causal_ops_detail

// PAD80_HELPER: zero-pad head_dim to 128 when no ninetoothed kernel covers it.
namespace h3_pad80_detail {
inline void run_padded_attention(const graph::GraphTensor &out_g,
                                 const graph::GraphTensor &q_g,
                                 const graph::GraphTensor &k_g,
                                 const graph::GraphTensor &v_g,
                                 const graph::GraphTensor &len_g,
                                 float scale,
                                 bool is_causal) {
    Tensor out(out_g), q(q_g), k(k_g), v(v_g), tkl(len_g);
    const Shape qs = q->shape();
    const Strides sq = q->strides();
    const Strides sk = k->strides();
    const Strides sv = v->strides();
    const Strides so = out->strides();
    const Size B = qs[0];
    const Size S = qs[1];
    const Size H = qs[2];
    const Size D = qs[3];
    const Size PAD = 128;
    Tensor qh = q->as_strided(Shape{B, H, S, D}, Strides{sq[0], sq[2], sq[1], sq[3]});
    Tensor kh = k->as_strided(Shape{B, H, S, D}, Strides{sk[0], sk[2], sk[1], sk[3]});
    Tensor vh = v->as_strided(Shape{B, H, S, D}, Strides{sv[0], sv[2], sv[1], sv[3]});
    Tensor oh = out->as_strided(Shape{B, H, S, D}, Strides{so[0], so[2], so[1], so[3]});
    Tensor qp = Tensor::zeros(Shape{B, H, S, PAD}, q->dtype(), q->device());
    Tensor kp = Tensor::zeros(Shape{B, H, S, PAD}, k->dtype(), k->device());
    Tensor vp = Tensor::zeros(Shape{B, H, S, PAD}, v->dtype(), v->device());
    Tensor op = Tensor::zeros(Shape{B, H, S, PAD}, out->dtype(), out->device());
    qp->narrow({{3, 0, D}})->copy_from(qh);
    kp->narrow({{3, 0, D}})->copy_from(kh);
    vp->narrow({{3, 0, D}})->copy_from(vh);
    op::flash_attention_(op, qp, kp, vp, tkl, scale, is_causal);
    oh->copy_from(op->narrow({{3, 0, D}}));
}
} // namespace h3_pad80_detail

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    // CAUSAL_OPS: masked attention is served by InfiniCore operators.
    if (p->is_causal) {
        Tensor out(p->out), q(p->q), k(p->k), v(p->v);
        h3_causal_ops_detail::run_causal_attention(out, q, k, v, static_cast<float>(p->scale));
        return;
    }

    // PAD80_GUARD: head_dim without a ninetoothed kernel -> padded 128 path
    if (p->out->shape()[3] != 72 && p->out->shape()[3] != 128) {
        h3_pad80_detail::run_padded_attention(p->out, p->q, p->k, p->v, p->total_kv_len,
                                             static_cast<float>(p->scale), p->is_causal);
        return;
    }

    auto q = head_major_view(p->q);
    auto k = head_major_view(p->k);
    auto v = head_major_view(p->v);
    auto out = head_major_view(p->out);

    // Any failure propagates: silently swapping in a different implementation
    // would change the numerics this project is reporting.
    op::flash_attention_(out, q, k, v, p->total_kv_len, p->scale, p->is_causal);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    MultiheadAttention::plan_dispatcher().registerDevice(Device::Type::HYGON, &plan);
    MultiheadAttention::run_dispatcher().registerDevice(Device::Type::HYGON, &run);
    MultiheadAttention::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &cleanup);
    return true;
}();

} // namespace infinicore::op::mha_impl::ninetoothed

#endif // ENABLE_HYGON_API && ENABLE_NINETOOTHED && INFINIOP_HYGON_NINETOOTHED_FLASH_ATTENTION
