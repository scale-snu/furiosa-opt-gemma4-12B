
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Ds, Dummy2, Dummy8, E, Gs, Ns};
use crate::device::layout::Slice;

type Cluster = m![Ns / 4];

type KvHeadsAcrossSlices = m![1 # 64, Ns % 4];

pub(crate) fn apply_rope(
    ctx: &mut Context,
    q: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Gs, Ds]>,
    k: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Ds]>,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
) -> (
    DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Ds]>,
) {
    // 단일 runtime index로 한 행을 먼저 읽는다. 그 행만 작은 HBM 임시 벡터를 통해
    // 두 cluster의 네 head slice로 명시적으로 복제한다.
    let cos: DmTensor<bf16, Chip, crate::device::layout::Cluster, Slice, m![Ds]> = cos.dma_gather_scaled(rope_offset);
    let sin: DmTensor<bf16, Chip, crate::device::layout::Cluster, Slice, m![Ds]> = sin.dma_gather_scaled(rope_offset);
    let cos: HbmTensor<bf16, Chip, m![Ds]> = cos.to_hbm(&mut ctx.tdma);
    let sin: HbmTensor<bf16, Chip, m![Ds]> = sin.to_hbm(&mut ctx.tdma);
    let cos: DmTensor<bf16, Chip, m![Dummy2], m![1 # 64, Dummy8 / 2], m![Ds]> = cos.to_dm(&mut ctx.tdma);
    let sin: DmTensor<bf16, Chip, m![Dummy2], m![1 # 64, Dummy8 / 2], m![Ds]> = sin.to_dm(&mut ctx.tdma);
    // 2 cluster × 4 slice의 모든 위치에 동일한 Ds table row가 들어 있다.
    let cos: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = unsafe { cos.reshape() };
    let sin: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = unsafe { sin.reshape() };

    let cos_vrf: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = ctx
        .sub
        .begin(cos.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let sin_vrf: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = ctx
        .sub
        .begin(sin.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let q: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![1 # 4, Gs, Ds]> = ctx
        .main
        .begin(q.view())
        .fetch::<m![Ns % 4], m![Gs, Ds]>()
        .switch::<KvHeadsAcrossSlices, m![1 # 4]>(SwitchConfig::InterTranspose {
            slice1: 4,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![1 # 4, Gs, Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit();

    let k: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![1 # 4, Ds]> = ctx
        .main
        .begin(k.view())
        .fetch::<m![Ns % 4], m![Ds]>()
        .switch::<KvHeadsAcrossSlices, m![1 # 4]>(SwitchConfig::InterTranspose {
            slice1: 4,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![1 # 4, Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit();

    let q: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]> = ctx
        .main
        .begin(q.view())
        .fetch::<m![1], m![Gs, Ds]>()
        .collect::<m![Gs, Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit();

    let k: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = ctx
        .main
        .begin(k.view())
        .fetch::<m![1], m![Ds]>()
        .collect::<m![Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit();

    let first_half_q = q.view().tile::<m![Ds], 128, m![Gs, Ds = 128 # 256]>(0);
    let second_half_q = q.view().tile::<m![Ds], 128, m![Gs, Ds = 128 # 256]>(128);

    let mut rotate_half_q: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]> = DmTensor::new();

    ctx.main
        .begin(first_half_q)
        .fetch::<m![Gs], m![Ds = 128]>()
        .collect::<m![Gs, Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(
            rotate_half_q
                .view_mut()
                .tile::<m![Ds], 128, m![Gs, Ds = 128 #{!} 256]>(128),
        );

    ctx.main
        .begin(second_half_q)
        .fetch::<m![Gs], m![Ds = 128]>()
        .collect::<m![Gs, Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(
            rotate_half_q
                .view_mut()
                .tile::<m![Ds], 128, m![Gs, Ds = 128 #{!} 256]>(0),
        );

    let first_half_k = k.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(0);
    let second_half_k = k.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(128);

    let mut rotate_half_k: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = DmTensor::new();

    ctx.main
        .begin(first_half_k)
        .fetch::<m![1], m![Ds = 128]>()
        .collect::<m![Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(rotate_half_k.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(128));

    ctx.main
        .begin(second_half_k)
        .fetch::<m![1], m![Ds = 128]>()
        .collect::<m![Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(rotate_half_k.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(0));

    let q_cos: DmTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]> = ctx
        .main
        .begin(q.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos_vrf)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let q_sin: DmTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]> = ctx
        .main
        .begin(rotate_half_q.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_vrf)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let q_sin_vrf: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]> = ctx
        .sub
        .begin(q_sin.view())
        .fetch::<m![Gs, Ds / 8], m![Ds % 8]>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let result_q: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]> = ctx
        .main
        .begin(q_cos.view())
        .fetch::<m![Gs, Ds / 8], m![Ds % 8]>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &q_sin_vrf)
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let k_cos: DmTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = ctx
        .main
        .begin(k.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let k_sin: DmTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = ctx
        .main
        .begin(rotate_half_k.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let k_sin_vrf: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = ctx
        .sub
        .begin(k_sin.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let result_k: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = ctx
        .main
        .begin(k_cos.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &k_sin_vrf)
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let result_q: DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Gs, Ds]> = ctx
        .main
        .begin(result_q.view())
        .fetch::<m![1], m![Gs, Ds]>()
        .switch::<Slice, m![Ns % 4]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 1 })
        .collect::<m![Ns % 4, Gs, Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit();

    let result_k: DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Ds]> = ctx
        .main
        .begin(result_k.view())
        .fetch::<m![1], m![Ds]>()
        .switch::<Slice, m![Ns % 4]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 1 })
        .collect::<m![Ns % 4, Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit();

    (result_q, result_k)
}
