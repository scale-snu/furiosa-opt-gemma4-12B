
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
    scratch: &mut HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
) -> (
    DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Ds]>,
) {
    // 공개 byte offset으로 각 table의 한 행을 단일 cluster에서 읽는다.
    let cos: DmTensor<bf16, Chip, crate::device::layout::Cluster, Slice, m![Ds]> = cos.dma_gather_scaled(rope_offset);
    let sin: DmTensor<bf16, Chip, crate::device::layout::Cluster, Slice, m![Ds]> = sin.dma_gather_scaled(rope_offset);
    // 두 행을 같은 cluster의 연속 DM에 모아 하나의 HBM scratch로 왕복한다.
    let mut tables: DmTensor<bf16, Chip, crate::device::layout::Cluster, Slice, m![Gs, Ds]> = DmTensor::new();
    ctx.main
        .begin(cos.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .collect::<m![Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit_view(tables.view_mut().tile::<m![Gs], 1, m![Gs = 1 #{!} 2, Ds]>(0));
    ctx.main
        .begin(sin.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .collect::<m![Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit_view(tables.view_mut().tile::<m![Gs], 1, m![Gs = 1 #{!} 2, Ds]>(1));
    // q_out의 첫 head(1,024 bytes)는 최종 Q 출력 전에만 임시로 쓴다.
    // 두 table 행을 모두 DM으로 읽은 뒤 기존 호출부가 q_out 전체를 덮어쓴다.
    tables.view().to_hbm_view(
        &mut ctx.tdma,
        scratch.view_mut().tile::<m![Ns], 1, m![Ns = 1 #{!} 8, Gs, Ds]>(0),
    );
    let tables: DmTensor<bf16, Chip, m![Dummy2], m![1 # 64, Dummy8 / 2], m![Gs, Ds]> = scratch
        .view()
        .tile::<m![Ns], 1, m![Ns = 1 # 8, Gs, Ds]>(0)
        .to_dm(&mut ctx.tdma);
    // 두 cluster와 네 head마다 두 행이 실제 복제돼 있으므로 spatial 축 이름만 바꾼다.
    let tables: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]> = unsafe { tables.reshape() };
    // tile의 실제 바이트 offset으로 두 행을 선택하며 singleton Gs는 fetch에서 제외한다.
    let cos = tables.view().tile::<m![Gs], 1, m![Gs = 1 # 2, Ds]>(0);
    let sin = tables.view().tile::<m![Gs], 1, m![Gs = 1 # 2, Ds]>(1);

    let cos_vrf: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = ctx
        .sub
        .begin(cos)
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let sin_vrf: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = ctx
        .sub
        .begin(sin)
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


// E200: RoPE sin 곱과 cos 항 덧셈을 같은 Main에 둔다. FMA와 중간 BF16 변경은 없다.
pub(crate) fn apply_rope_sin_add_e200(
    ctx: &mut Context,
    q: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Gs, Ds]>,
    k: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Ds]>,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
    scratch: &mut HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
) -> (
    DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Ds]>,
) {
    // 공개 byte offset으로 각 table의 한 행을 단일 cluster에서 읽는다.
    let cos: DmTensor<bf16, Chip, crate::device::layout::Cluster, Slice, m![Ds]> = cos.dma_gather_scaled(rope_offset);
    let sin: DmTensor<bf16, Chip, crate::device::layout::Cluster, Slice, m![Ds]> = sin.dma_gather_scaled(rope_offset);
    // 두 행을 같은 cluster의 연속 DM에 모아 하나의 HBM scratch로 왕복한다.
    let mut tables: DmTensor<bf16, Chip, crate::device::layout::Cluster, Slice, m![Gs, Ds]> = DmTensor::new();
    ctx.main
        .begin(cos.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .collect::<m![Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit_view(tables.view_mut().tile::<m![Gs], 1, m![Gs = 1 #{!} 2, Ds]>(0));
    ctx.main
        .begin(sin.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .collect::<m![Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit_view(tables.view_mut().tile::<m![Gs], 1, m![Gs = 1 #{!} 2, Ds]>(1));
    // q_out의 첫 head(1,024 bytes)는 최종 Q 출력 전에만 임시로 쓴다.
    // 두 table 행을 모두 DM으로 읽은 뒤 기존 호출부가 q_out 전체를 덮어쓴다.
    tables.view().to_hbm_view(
        &mut ctx.tdma,
        scratch.view_mut().tile::<m![Ns], 1, m![Ns = 1 #{!} 8, Gs, Ds]>(0),
    );
    let tables: DmTensor<bf16, Chip, m![Dummy2], m![1 # 64, Dummy8 / 2], m![Gs, Ds]> = scratch
        .view()
        .tile::<m![Ns], 1, m![Ns = 1 # 8, Gs, Ds]>(0)
        .to_dm(&mut ctx.tdma);
    // 두 cluster와 네 head마다 두 행이 실제 복제돼 있으므로 spatial 축 이름만 바꾼다.
    let tables: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]> = unsafe { tables.reshape() };
    // tile의 실제 바이트 offset으로 두 행을 선택하며 singleton Gs는 fetch에서 제외한다.
    let cos = tables.view().tile::<m![Gs], 1, m![Gs = 1 # 2, Ds]>(0);
    let sin = tables.view().tile::<m![Gs], 1, m![Gs = 1 # 2, Ds]>(1);

    let cos_vrf: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = ctx
        .sub
        .begin(cos)
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let sin_vrf: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = ctx
        .sub
        .begin(sin)
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

    let q_cos_vrf: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]> = ctx
        .sub
        .begin(q_cos.view())
        .fetch::<m![Gs, Ds / 8], m![Ds % 8]>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let result_q: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]> = ctx
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
        .vector_clip(ClipBinaryOpF32::Add, &q_cos_vrf)
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

    let k_cos_vrf: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = ctx
        .sub
        .begin(k_cos.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let result_k: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = ctx
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
        .vector_clip(ClipBinaryOpF32::Add, &k_cos_vrf)
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
