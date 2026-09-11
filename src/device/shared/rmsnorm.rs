
use furiosa_opt_std::prelude::*;

use crate::axes::{Dummy8, H};

use crate::{Chip, EPS};

const H_F32: f32 = H::SIZE as f32;

pub(crate) fn normalize<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    type ReducingSlices = m![1 # 32, H / 480];

    let x: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = x.to_dm(&mut ctx.tdma);

    let mean_square: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let reduced_mean_square: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 32, Dummy8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .main
        .begin(reduced_mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = unsafe { rms.reshape() };

    let weight_dm: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();

    let rms_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let normalized: DmTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .commit_trim::<m![H % 8]>()
        .commit();

    ctx.main
        .begin(normalized.view())
        .fetch::<m![1], m![H % 480]>()
        .switch::<Slice, m![H / 480]>(SwitchConfig::Broadcast1 { slice1: 8, slice0: 1 })
        .collect::<m![H / 8], m![H % 8]>()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

// Stage 1 O projection 후처리: RMSNorm과 residual add의 slice 배치를 유지한다.
pub(crate) fn normalize_add<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    residual: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, m![1 # 32, H / 480], m![H % 480]> {
    type ReducingSlices = m![1 # 32, H / 480];

    // Projection의 HBM 결과를 norm의 8개 slice에 바로 읽어 중간 DM 재분산을 제거한다.
    let x: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = x.to_dm(&mut ctx.tdma);

    let mean_square: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // Inter-slice 합산, EPS, sqrt는 합치되 VRF는 검증된 DM→Sub 경로로 채운다.
    let rms: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 32, Dummy8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, EPS)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // reduction이8개 slice에 같은 scalar를 복제했으므로 기존 H/480 배치로 선택한다.
    let rms: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = unsafe { rms.reshape() };

    let weight_dm: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();

    let rms_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let normalized: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        // RMSNorm의 bf16 반올림을 residual 덧셈 전에 유지한다.
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    // 정규화 결과를 한 slice로 모았다가 다시 나누지 않고 같은 8개 slice에서 더한다.
    let residual: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = residual.to_dm(&mut ctx.tdma);
    let residual_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .sub
        .begin(residual.view())
        .fetch::<m![1], m![H % 480]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();
    let output: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(normalized.view())
        .fetch::<m![1], m![H % 480]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &residual_vrf)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    // 각 slice의 480개 BF16 결과를 그대로 반환하여 HBM에 직접 쓴다.
    output
}

// QKV 입력의 기존 8-slice RMSNorm과 bf16 반올림을 유지하고 직접 broadcast한다.
pub(crate) fn normalize_broadcast<Cluster: M>(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, crate::device::layout::Replicated, m![H]> {
    type ReducingSlices = m![1 # 32, H / 480];

    let x: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = x.to_dm(&mut ctx.tdma);

    let mean_square: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let reduced_mean_square: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 32, Dummy8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .main
        .begin(reduced_mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // slice 간 합계와 sqrt가 8개 Dummy8 위치에 동일하게 복제됐으므로 H 분할 매핑으로 재해석한다.
    let rms: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = unsafe { rms.reshape() };

    let weight_dm: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();

    let rms_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let normalized: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    // 8개 slice의 H480 분할에서 직접 전체 H를 각 projection slice에 복제한다.
    ctx.main
        .begin(normalized.view())
        .fetch::<m![1], m![H % 480]>()
        .switch::<crate::device::layout::Replicated, m![H / 480]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit()
}

// Stage 1 FFN 후처리: RMSNorm, residual add, layer gate의 8-slice 배치를 유지한다.
pub(crate) fn normalize_add_gate<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    residual: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
    scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    type ReducingSlices = m![1 # 32, H / 480];

    let x: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = x.to_dm(&mut ctx.tdma);

    let mean_square: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // Inter-slice 합산, EPS, sqrt를 연결하되 RMS scalar의 DM→Sub→VRF 경로를 유지한다.
    let rms: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 32, Dummy8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, EPS)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // reduction이 Dummy8의 모든 slice에 같은 scalar를 복제했으므로 H / 480 행으로 선택한다.
    let rms: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = unsafe { rms.reshape() };

    let weight_dm: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();

    let rms_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let normalized: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        // RMSNorm의 bf16 반올림을 residual 덧셈 전에 유지한다.
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    // 정규화 결과를 한 slice로 모았다가 다시 나누지 않고 같은 8개 slice에서 더한다.
    let residual: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = residual.to_dm(&mut ctx.tdma);
    let residual_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .sub
        .begin(residual.view())
        .fetch::<m![1], m![H % 480]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();
    let output: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(normalized.view())
        .fetch::<m![1], m![H % 480]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &residual_vrf)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    // residual 덧셈의 bf16 반올림 뒤 같은 8개 slice에서 layer gate를 적용한다.
    let scalar: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![1 # 8]> = scalar.to_dm(&mut ctx.tdma);
    let scalar_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx.sub
        .begin(scalar.view())
        .fetch::<m![1], m![1 # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let output: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx.main
        .begin(output.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scalar_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    ctx.main
        .begin(output.view())
        .fetch::<m![1], m![H % 480]>()
        .switch::<Slice, m![H / 480]>(SwitchConfig::Broadcast1 { slice1: 8, slice0: 1 })
        .collect::<m![H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit()
}

// Cluster의 Dummy2와 겹치지 않는 FFN 전용 reduction 복제 축이다.
axes![FfnRms16 = 16];

// FFN 입력을 HBM에서 16개 연속 slice에 직접 읽고 정규화한 뒤 broadcast한다.
pub(crate) fn normalize_broadcast_ffn16<Cluster: M>(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, crate::device::layout::Replicated, m![H]> {
    type ReducingSlices = m![1 # 16, H / 240];

    let x: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 240]> = x.to_dm(&mut ctx.tdma);

    let mean_square: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let reduced_mean_square: DmTensor<f32, Chip, Cluster, m![1 # 16, FfnRms16], m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 16, FfnRms16], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cluster, m![1 # 16, FfnRms16], m![1 # 8]> = ctx
        .main
        .begin(reduced_mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // 16개 slice 모두에 복제된 RMS 값을 같은 물리 위치의 H240 분할로 읽는다.
    let rms: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = unsafe { rms.reshape() };

    let weight_dm: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 240]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 240]> = ctx
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .to_vrf();

    let rms_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let normalized: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 240]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 30], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    // 16개 slice의 H240 분할에서 직접 전체 H를 각 projection slice에 복제한다.
    ctx.main
        .begin(normalized.view())
        .fetch::<m![1], m![H % 240]>()
        .switch::<crate::device::layout::Replicated, m![H / 240]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit()
}

pub(crate) fn normalize_add_gate_sharded<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, crate::device::shared::mlp::FfnSlices, m![H % 480]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    residual: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
    scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    type ReducingSlices = m![1 # 32, H / 480];

    // global scale이 bf16로 저장한 H480×8 입력을 같은 배치에서 읽는다.

    let mean_square: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // Inter-slice 합산, EPS, sqrt를 연결하되 RMS scalar의 DM→Sub→VRF 경로를 유지한다.
    let rms: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 32, Dummy8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, EPS)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // reduction이 Dummy8의 모든 slice에 같은 scalar를 복제했으므로 H / 480 행으로 선택한다.
    let rms: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = unsafe { rms.reshape() };

    let weight_dm: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();

    let rms_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let normalized: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        // RMSNorm의 bf16 반올림을 residual 덧셈 전에 유지한다.
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    // 정규화 결과를 한 slice로 모았다가 다시 나누지 않고 같은 8개 slice에서 더한다.
    let residual: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = residual.to_dm(&mut ctx.tdma);
    let residual_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .sub
        .begin(residual.view())
        .fetch::<m![1], m![H % 480]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();
    let output: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(normalized.view())
        .fetch::<m![1], m![H % 480]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &residual_vrf)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    // residual 덧셈의 bf16 반올림 뒤 같은 8개 slice에서 layer gate를 적용한다.
    let scalar: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![1 # 8]> = scalar.to_dm(&mut ctx.tdma);
    let scalar_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx.sub
        .begin(scalar.view())
        .fetch::<m![1], m![1 # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let output: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx.main
        .begin(output.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scalar_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    ctx.main
        .begin(output.view())
        .fetch::<m![1], m![H % 480]>()
        .switch::<Slice, m![H / 480]>(SwitchConfig::Broadcast1 { slice1: 8, slice0: 1 })
        .collect::<m![H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit()
}

pub(crate) fn normalize_add_gate_sharded_unscaled<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, crate::device::shared::mlp::FfnSlices, m![H % 480]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    residual: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
    scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    type ReducingSlices = m![1 # 32, H / 480];

    // global scale은 평균제곱과 정규화 양쪽에서 f32 곱셈으로 적용한다.
    // down dot의 bf16 값과 norm/residual/layer gate의 bf16 반올림은 유지한다.
    // scale 직후의 중간 bf16 반올림만 제거하므로 별도 수치 검증이 필요하다.
    let scale_dm: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> =
        down_global_scale.to_dm(&mut ctx.tdma);
    let scale_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx.sub
        .begin(scale_dm.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let mean_square: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // Inter-slice 합산, EPS, sqrt를 연결하되 RMS scalar의 DM→Sub→VRF 경로를 유지한다.
    let rms: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 32, Dummy8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, EPS)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // reduction이 Dummy8의 모든 slice에 같은 scalar를 복제했으므로 H / 480 행으로 선택한다.
    let rms: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = unsafe { rms.reshape() };

    let weight_dm: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();

    let rms_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let normalized: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        // RMSNorm의 bf16 반올림을 residual 덧셈 전에 유지한다.
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    // 정규화 결과를 한 slice로 모았다가 다시 나누지 않고 같은 8개 slice에서 더한다.
    let residual: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = residual.to_dm(&mut ctx.tdma);
    let residual_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .sub
        .begin(residual.view())
        .fetch::<m![1], m![H % 480]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();
    let output: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(normalized.view())
        .fetch::<m![1], m![H % 480]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &residual_vrf)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    // residual 덧셈의 bf16 반올림 뒤 같은 8개 slice에서 layer gate를 적용한다.
    let scalar: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![1 # 8]> = scalar.to_dm(&mut ctx.tdma);
    let scalar_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx.sub
        .begin(scalar.view())
        .fetch::<m![1], m![1 # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let output: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx.main
        .begin(output.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scalar_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    ctx.main
        .begin(output.view())
        .fetch::<m![1], m![H % 480]>()
        .switch::<Slice, m![H / 480]>(SwitchConfig::Broadcast1 { slice1: 8, slice0: 1 })
        .collect::<m![H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit()
}

// E70 O 후처리 수치 융합을 그대로 재사용한다.
pub(crate) fn normalize_add_output_unscaled<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    channel_scale: &HbmTensor<bf16, Chip, m![H]>,
    residual: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, m![1 # 32, H / 480], m![H % 480]> {
    type ReducingSlices = m![1 # 32, H / 480];

    // Projection의 HBM 결과를 norm의 8개 slice에 바로 읽어 중간 DM 재분산을 제거한다.
    let x: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = x.to_dm(&mut ctx.tdma);

    // 채널 scale을8개 H480 slice에 읽고 두 패스에서 동일한 f32 곱을 계산한다.
    let channel_scale: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = channel_scale.to_dm(&mut ctx.tdma);
    let channel_scale_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .sub
        .begin(channel_scale.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();

    let mean_square: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        // 수치 변경: 기존 BF16(channel_scale * BF16(dot))의 두 번째 BF16 반올림을 제거한다.
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &channel_scale_vrf)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // Inter-slice 합산, EPS, sqrt는 합치되 VRF는 검증된 DM→Sub 경로로 채운다.
    let rms: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 32, Dummy8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, EPS)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // reduction이8개 slice에 같은 scalar를 복제했으므로 기존 H/480 배치로 선택한다.
    let rms: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = unsafe { rms.reshape() };

    let weight_dm: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();

    let rms_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    // 원본 residual을 같은 8개 slice에 보존하고 f32 VRF에 올린다.
    let residual: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = residual.to_dm(&mut ctx.tdma);
    let residual_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .sub
        .begin(residual.view())
        .fetch::<m![1], m![H % 480]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();
    let output: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &channel_scale_vrf)
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        // 수치 변경: 중간 RMSNorm BF16 반올림을 생략하고 f32 값에 residual을 더한다.
        .vector_clip(ClipBinaryOpF32::Add, &residual_vrf)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    // 각 slice의 480개 BF16 결과를 그대로 반환하여 HBM에 직접 쓴다.
    output
}


// E118 QKV 전용: 기존 normalize_broadcast와 동일한 연산을 broadcast 직전까지 수행.
pub(crate) fn normalize_partial_qkv_e118<Cluster: M>(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, m![1 # 32, H / 480], m![H % 480]> {
    type ReducingSlices = m![1 # 32, H / 480];

    let x: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = x.to_dm(&mut ctx.tdma);

    let mean_square: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let reduced_mean_square: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 32, Dummy8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .main
        .begin(reduced_mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // slice 간 합계와 sqrt가 8개 Dummy8 위치에 동일하게 복제됐으므로 H 분할 매핑으로 재해석한다.
    let rms: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = unsafe { rms.reshape() };

    let weight_dm: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();

    let rms_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let normalized: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    // E118: 기존 BF16 반올림을 끝낸 H480 분할을 유지하여 FP8 인코딩에 쓴다.
    normalized
}

// E148: 후처리 slice 수를16개로 늘려 각각의 H/VRF 크기를 절반으로 줄인다.
axes![O148RmsReplica16 = 16];
pub(crate) fn normalize_add_output_norm16_e148<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    channel_scale: &HbmTensor<bf16, Chip, m![H]>,
    residual: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, m![1 # 16, H / 240], m![H % 240]> {
    type ReducingSlices = m![1 # 16, H / 240];

    // Projection의 HBM 결과를 norm의 16개 slice에 바로 읽어 중간 DM 재분산을 제거한다.
    let x: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 240]> = x.to_dm(&mut ctx.tdma);

    // 채널 scale을16개 H240 slice에 읽고 두 패스에서 동일한 f32 곱을 계산한다.
    let channel_scale: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 240]> = channel_scale.to_dm(&mut ctx.tdma);
    let channel_scale_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 240]> = ctx
        .sub
        .begin(channel_scale.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .to_vrf();

    let mean_square: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        // 수치 변경: 기존 BF16(channel_scale * BF16(dot))의 두 번째 BF16 반올림을 제거한다.
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &channel_scale_vrf)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // Inter-slice 합산, EPS, sqrt는 합치되 VRF는 검증된 DM→Sub 경로로 채운다.
    let rms: DmTensor<f32, Chip, Cluster, m![1 # 16, O148RmsReplica16], m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 16, O148RmsReplica16], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, EPS)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // reduction이 16개 slice에 같은 scalar를 복제했으므로 H/240 배치로 이름만 바꾼다.
    let rms: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = unsafe { rms.reshape() };

    let weight_dm: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 240]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 240]> = ctx
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .to_vrf();

    let rms_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    // 원본 residual을 같은 16개 slice에 보존하고 f32 VRF에 올린다.
    let residual: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 240]> = residual.to_dm(&mut ctx.tdma);
    let residual_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 240]> = ctx
        .sub
        .begin(residual.view())
        .fetch::<m![1], m![H % 240]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .to_vrf();
    let output: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 240]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &channel_scale_vrf)
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 30], m![H % 8]>()
        // 수치 변경: 중간 RMSNorm BF16 반올림을 생략하고 f32 값에 residual을 더한다.
        .vector_clip(ClipBinaryOpF32::Add, &residual_vrf)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    // 각 slice의 240개 BF16 결과를 그대로 반환하여 HBM에 직접 쓴다.
    output
}

// E149: 16-slice norm에서 residual HBM을 같은16개 slice에 바로 읽는다.
pub(crate) fn normalize_add_output_norm16_direct_e149<Cluster: M>(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    channel_scale: &HbmTensor<bf16, Chip, m![H]>,
    residual: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, m![1 # 16, H / 240], m![H % 240]> {
    type ReducingSlices = m![1 # 16, H / 240];

    // Projection의 HBM 결과를 norm의 16개 slice에 바로 읽어 중간 DM 재분산을 제거한다.
    let x: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 240]> = x.to_dm(&mut ctx.tdma);

    // 채널 scale을16개 H240 slice에 읽고 두 패스에서 동일한 f32 곱을 계산한다.
    let channel_scale: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 240]> = channel_scale.to_dm(&mut ctx.tdma);
    let channel_scale_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 240]> = ctx
        .sub
        .begin(channel_scale.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .to_vrf();

    let mean_square: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        // 수치 변경: 기존 BF16(channel_scale * BF16(dot))의 두 번째 BF16 반올림을 제거한다.
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &channel_scale_vrf)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // Inter-slice 합산, EPS, sqrt는 합치되 VRF는 검증된 DM→Sub 경로로 채운다.
    let rms: DmTensor<f32, Chip, Cluster, m![1 # 16, O148RmsReplica16], m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 16, O148RmsReplica16], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, EPS)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // reduction이 16개 slice에 같은 scalar를 복제했으므로 H/240 배치로 이름만 바꾼다.
    let rms: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = unsafe { rms.reshape() };

    let weight_dm: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 240]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 240]> = ctx
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .to_vrf();

    let rms_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    // 원본 residual을 같은 16개 slice에 보존하고 f32 VRF에 올린다.
    let residual: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 240]> = residual.to_dm(&mut ctx.tdma);
    let residual_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 240]> = ctx
        .sub
        .begin(residual.view())
        .fetch::<m![1], m![H % 240]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .to_vrf();
    let output: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 240]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &channel_scale_vrf)
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 30], m![H % 8]>()
        // 수치 변경: 중간 RMSNorm BF16 반올림을 생략하고 f32 값에 residual을 더한다.
        .vector_clip(ClipBinaryOpF32::Add, &residual_vrf)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    // 각 slice의 240개 BF16 결과를 그대로 반환하여 HBM에 직접 쓴다.
    output
}

// E161: RMS 역수를 scalar에서 한 번 계산하는 O 전용 후보.
pub(crate) fn normalize_add_output_inverse_rms_e161<Cluster: M>(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    channel_scale: &HbmTensor<bf16, Chip, m![H]>,
    residual: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, m![1 # 16, H / 240], m![H % 240]> {
    type ReducingSlices = m![1 # 16, H / 240];

    // Projection의 HBM 결과를 norm의 16개 slice에 바로 읽어 중간 DM 재분산을 제거한다.
    let x: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 240]> = x.to_dm(&mut ctx.tdma);

    // 채널 scale을16개 H240 slice에 읽고 두 패스에서 동일한 f32 곱을 계산한다.
    let channel_scale: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 240]> = channel_scale.to_dm(&mut ctx.tdma);
    let channel_scale_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 240]> = ctx
        .sub
        .begin(channel_scale.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .to_vrf();

    let mean_square: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        // 수치 변경: 기존 BF16(channel_scale * BF16(dot))의 두 번째 BF16 반올림을 제거한다.
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &channel_scale_vrf)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // Inter-slice 합산, EPS, sqrt는 합치되 VRF는 검증된 DM→Sub 경로로 채운다.
    let rms: DmTensor<f32, Chip, Cluster, m![1 # 16, O148RmsReplica16], m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 16, O148RmsReplica16], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, EPS)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        // E161: 16개 실제 RMS scalar에서만 역수를 구한다. EPS와 sqrt는 그대로다.
        .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // reduction이 16개 slice에 같은 scalar를 복제했으므로 H/240 배치로 이름만 바꾼다.
    let rms: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = unsafe { rms.reshape() };

    let weight_dm: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 240]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 240]> = ctx
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .to_vrf();

    let rms_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    // 원본 residual을 같은 16개 slice에 보존하고 f32 VRF에 올린다.
    let residual: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 240]> = residual.to_dm(&mut ctx.tdma);
    let residual_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 240]> = ctx
        .sub
        .begin(residual.view())
        .fetch::<m![1], m![H % 240]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .to_vrf();
    let output: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 240]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &channel_scale_vrf)
        // rms_vrf는 1/sqrt(mean_square+EPS)이다. 원소별 FPU divide를 두 번째 Mul로 대체한다.
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &rms_vrf)
        // 별도 FMA ALU의 곱(+0)을 사용해 Mul0/Mul1을 한 pipeline에서 재사용하지 않는다.
        .vector_fp_ternary(FpTernaryOp::FmaF, (&weight_vrf, 0.0))
        .vector_widen_concat::<m![H / 8 % 30], m![H % 8]>()
        // 수치 변경: 중간 RMSNorm BF16 반올림을 생략하고 f32 값에 residual을 더한다.
        .vector_clip(ClipBinaryOpF32::Add, &residual_vrf)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    // 각 slice의 240개 BF16 결과를 그대로 반환하여 HBM에 직접 쓴다.
    output
}

// E173: E171에서 감사한 norm16을 그대로 재사용한다. EPS, BF16 round, residual, layer gate 보존.
pub(crate) fn normalize_add_gate_sharded_unscaled16_e171<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, m![1 # 16, H / 240], m![H % 240]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    residual: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
    scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    type ReducingSlices = m![1 # 16, H / 240];

    // global scale은 평균제곱과 정규화 양쪽에서 f32 곱셈으로 적용한다.
    // down dot의 bf16 값과 norm/residual/layer gate의 bf16 반올림은 유지한다.
    // scale 직후의 중간 bf16 반올림만 제거하므로 별도 수치 검증이 필요하다.
    let scale_dm: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> =
        down_global_scale.to_dm(&mut ctx.tdma);
    let scale_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx.sub
        .begin(scale_dm.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let mean_square: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // Inter-slice 합산, EPS, sqrt를 연결하되 RMS scalar의 DM→Sub→VRF 경로를 유지한다.
    let rms: DmTensor<f32, Chip, Cluster, m![1 # 16, FfnRms16], m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 16, FfnRms16], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, EPS)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // reduction이 유효한 첫16개 slice 모두에 같은 scalar를 복제했다.
    // 같은 물리16개 위치를 H/240 이름으로 읽으며 padding으로 유효 영역을 확대하지 않는다.
    let rms: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = unsafe { rms.reshape() };

    let weight_dm: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 240]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 240]> = ctx
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .to_vrf();

    let rms_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let normalized: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 240]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 30], m![H % 8]>()
        .vector_final()
        // RMSNorm의 bf16 반올림을 residual 덧셈 전에 유지한다.
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    // 정규화 결과를 한 slice로 모았다가 다시 나누지 않고 같은 16개 slice에서 더한다.
    let residual: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 240]> = residual.to_dm(&mut ctx.tdma);
    let residual_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 240]> = ctx
        .sub
        .begin(residual.view())
        .fetch::<m![1], m![H % 240]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .to_vrf();
    let output: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 240]> = ctx
        .main
        .begin(normalized.view())
        .fetch::<m![1], m![H % 240]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &residual_vrf)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    // residual 덧셈의 bf16 반올림 뒤 같은 16개 slice에서 layer gate를 적용한다.
    let scalar: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![1 # 8]> = scalar.to_dm(&mut ctx.tdma);
    let scalar_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx.sub
        .begin(scalar.view())
        .fetch::<m![1], m![1 # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let output: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 240]> = ctx.main
        .begin(output.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scalar_vrf)
        .vector_widen_concat::<m![H / 8 % 30], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    ctx.main
        .begin(output.view())
        .fetch::<m![1], m![H % 240]>()
        .switch::<Slice, m![H / 240]>(SwitchConfig::Broadcast1 { slice1: 16, slice0: 1 })
        .collect::<m![H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit()
}

// E175: 마지막 Main gather를 제거하고 분산 출력에서 residual HBM으로 직접 쓴다.
pub(crate) fn normalize_add_gate_sharded16_store_e175<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, m![1 # 16, H / 240], m![H % 240]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    residual: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
    scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
    residual_hbm: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    type ReducingSlices = m![1 # 16, H / 240];

    // global scale은 평균제곱과 정규화 양쪽에서 f32 곱셈으로 적용한다.
    // down dot의 bf16 값과 norm/residual/layer gate의 bf16 반올림은 유지한다.
    // scale 직후의 중간 bf16 반올림만 제거하므로 별도 수치 검증이 필요하다.
    let scale_dm: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> =
        down_global_scale.to_dm(&mut ctx.tdma);
    let scale_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx.sub
        .begin(scale_dm.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let mean_square: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // Inter-slice 합산, EPS, sqrt를 연결하되 RMS scalar의 DM→Sub→VRF 경로를 유지한다.
    let rms: DmTensor<f32, Chip, Cluster, m![1 # 16, FfnRms16], m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 16, FfnRms16], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, EPS)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    // reduction이 유효한 첫16개 slice 모두에 같은 scalar를 복제했다.
    // 같은 물리16개 위치를 H/240 이름으로 읽으며 padding으로 유효 영역을 확대하지 않는다.
    let rms: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = unsafe { rms.reshape() };

    let weight_dm: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 240]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 240]> = ctx
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .to_vrf();

    let rms_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let normalized: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 240]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 30], m![H % 8]>()
        .vector_final()
        // RMSNorm의 bf16 반올림을 residual 덧셈 전에 유지한다.
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    // 정규화 결과를 한 slice로 모았다가 다시 나누지 않고 같은 16개 slice에서 더한다.
    let residual: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 240]> = residual.to_dm(&mut ctx.tdma);
    let residual_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 240]> = ctx
        .sub
        .begin(residual.view())
        .fetch::<m![1], m![H % 240]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .to_vrf();
    let output: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 240]> = ctx
        .main
        .begin(normalized.view())
        .fetch::<m![1], m![H % 240]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &residual_vrf)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    // residual 덧셈의 bf16 반올림 뒤 같은 16개 slice에서 layer gate를 적용한다.
    let scalar: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![1 # 8]> = scalar.to_dm(&mut ctx.tdma);
    let scalar_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx.sub
        .begin(scalar.view())
        .fetch::<m![1], m![1 # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let output: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 240]> = ctx.main
        .begin(output.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scalar_vrf)
        .vector_widen_concat::<m![H / 8 % 30], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    // 실제16개 slice가 각각 H240 구간을 소유한다. 전역 H=3840을 한 번씩 쓰며
    // 마지막 layer gate의 BF16 commit 결과를 그대로 전송한다. 추가 cast/reshape는 없다.
    output.view().to_hbm_view(&mut ctx.tdma, residual_hbm.view_mut());
}
