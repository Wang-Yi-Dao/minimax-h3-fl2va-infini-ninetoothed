// Fused attention for HYGON (DCU) on top of the upstream nine-toothed flash-attention
// kernel. This file is compiled only when the hygon nine-toothed flash-attention
// registration is enabled, so it never clashes with the ATen based
// mha_flashattn_hygon.cc path.
#if defined(ENABLE_HYGON_API) && defined(ENABLE_NINETOOTHED) && defined(INFINIOP_HYGON_NINETOOTHED_FLASH_ATTENTION)

#include "infinicore/ops/mha.hpp"

#include "infinicore/ops/flash_attention.hpp"

#include <cstdint>
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
    Strides view_strides{strides[0], strides[3], strides[1], strides[2]};

    return Tensor::strided_from_blob(
        const_cast<void *>(static_cast<const void *>(tensor->data())),
        view_shape, view_strides, tensor->dtype(), tensor->device());
}

// The kernel reads the packed kv length from a device side int32 scalar.
Tensor make_total_kv_len(const Tensor &k) {
    const int32_t total = static_cast<int32_t>(k->size(0) * k->size(1));

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

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    auto q = head_major_view(p->q);
    auto k = head_major_view(p->k);
    auto v = head_major_view(p->v);
    auto out = head_major_view(p->out);

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
