
use furiosa_opt_std::prelude::*;

use crate::axes::{Ds, Gs, Ns};
use crate::device::layout::Slice;

// 네 head씩 두 cluster에서 독립적으로 정규화한다. Ds reduction과 반올림은 유지한다.
type Cluster = m![Ns / 4];
use crate::{Chip, EPS};

const DS_F32: f32 = Ds::SIZE as f32;

pub(crate) fn normalize_query<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Gs, Ds]>,
    rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Gs, Ds]> {
    let mean_square: DmTensor<f32, Chip, Cluster, Slice, m![Ns % 4, Gs]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![Ns % 4, Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ns % 4, Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns % 4, Gs, Ds / 4], m![Ds % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Ds, m![Ns % 4, Gs], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .transpose::<m![Ns % 4], m![Gs % 2 # 8]>()
        .commit_trim::<m![Gs % 2]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cluster, Slice, m![Ns % 4, Gs]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![Ns % 4, Gs]>()
        .collect::<m![1], m![Ns % 4, Gs]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns % 4 / 2], m![Ns % 2, Gs]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_concat::<m![1], m![Ns % 4, Gs]>()
        .vector_final()
        .commit_trim::<m![Ns % 4, Gs]>()
        .commit();

    let weight_vrf = load_norm_weight::<Cluster, Slice>(ctx, rms_weight);

    let rms_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![Ns % 4, Gs]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![Ns % 4, Gs]>()
        .collect::<m![1], m![Ns % 4, Gs]>()
        .to_vrf();

    ctx.main
        .begin(x.view())
        .fetch::<m![Ns % 4, Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ns % 4, Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns % 4, Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![Ns % 4, Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

fn load_norm_weight<Cluster: M, Slice: M>(
    ctx: &mut Context,
    rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> VrfTensor<f32, Chip, Cluster, Slice, m![Ds]> {
    let weight_dm: DmTensor<bf16, Chip, Cluster, Slice, m![Ds]> = rms_weight.to_dm(&mut ctx.tdma);

    ctx.sub
        .begin(weight_dm.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf()
}

fn root_mean_square<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Ds]>,
) -> VrfTensor<f32, Chip, Cluster, Slice, m![Ns % 4 # 8]> {
    let mean_square: DmTensor<f32, Chip, Cluster, Slice, m![Ns % 4]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![Ns % 4, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ns % 4, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns % 4, Ds / 4], m![Ds % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Ds, m![Ns % 4], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .transpose::<m![Ns % 4 / 2], m![Ns % 2 # 8]>()
        .commit_trim::<m![Ns % 2]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cluster, Slice, m![Ns % 4]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![Ns % 4 # 8]>()
        .collect::<m![1], m![Ns % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![Ns % 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![Ns % 4 # 8]>()
        .vector_final()
        .commit_trim::<m![Ns % 4]>()
        .commit();

    ctx.sub
        .begin(rms.view())
        .fetch::<m![1], m![Ns % 4 # 8]>()
        .collect::<m![1], m![Ns % 4 # 8]>()
        .to_vrf()
}

fn scale_by_rms_and_weight<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Ds]>,
    rms_vrf: &VrfTensor<f32, Chip, Cluster, Slice, m![Ns % 4 # 8]>,
    weight_vrf: &VrfTensor<f32, Chip, Cluster, Slice, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Ds]> {
    ctx.main
        .begin(x.view())
        .fetch::<m![Ns % 4, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ns % 4, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns % 4, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), weight_vrf)
        .vector_widen_concat::<m![Ns % 4, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

fn scale_by_rms<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Ds]>,
    rms_vrf: &VrfTensor<f32, Chip, Cluster, Slice, m![Ns % 4 # 8]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Ds]> {
    ctx.main
        .begin(x.view())
        .fetch::<m![Ns % 4, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ns % 4, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns % 4, Ds / 4], m![Ds % 4]>()
        .vector_fp_div(rms_vrf)
        .vector_widen_concat::<m![Ns % 4, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

pub(crate) fn normalize_key(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Ds]>,
    rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Ds]> {
    let rms_vrf = root_mean_square(ctx, x);
    let weight_vrf = load_norm_weight::<Cluster, Slice>(ctx, rms_weight);

    scale_by_rms_and_weight(ctx, x, &rms_vrf, &weight_vrf)
}

pub(crate) fn normalize_value(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Ds]> {
    let rms_vrf = root_mean_square(ctx, x);

    scale_by_rms(ctx, x, &rms_vrf)
}

// V의 각 head를 독립 slice에서 정규화한다. 각 head 내부 Ds 합산 순서는 같다.
pub(crate) fn normalize_value_heads(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, crate::device::sliding::projection::ValueHeadSlices, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, crate::device::sliding::projection::ValueHeadSlices, m![Ds]> {
    type HeadSlices = crate::device::sliding::projection::ValueHeadSlices;
    let mean_square: DmTensor<f32, Chip, Cluster, HeadSlices, m![1 # 8]> = ctx.main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DS_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cluster, HeadSlices, m![1 # 8]> = ctx.main
        .begin(mean_square.view())
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
    let rms: VrfTensor<f32, Chip, Cluster, HeadSlices, m![1 # 8]> = ctx.sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    ctx.main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_div(&rms)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}


// E209: Contraction 뒤의 Vector에서 head RMS의 sqrt까지 계산한다.
pub(crate) fn normalize_query_dot_e209<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Gs, Ds]>,
    rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Gs, Ds]> {
    // E209: head마다 같은 BF16 x를 TRF와 stream에서 읽어 모든 Ds256의 제곱합을 계산한다.
    // head 축은 양쪽 operand의 같은 논리 축이며 다른 head의 값과 곱하지 않는다.
    let square_trf: TrfTensor<bf16, Chip, Cluster, Slice, m![1], m![Ns % 4, Gs, Ds]> = ctx.sub
        .begin(x.view()).fetch::<m![Ns % 4, Gs, Ds / 16], m![Ds % 16]>()
        .collect::<m![Ns % 4, Gs, Ds / 16], m![Ds % 16]>().to_trf();
    let rms: DmTensor<f32, Chip, Cluster, Slice, m![Ns % 4, Gs]> = ctx.main
        .begin(x.view()).fetch::<m![Ns % 4, Gs, Ds / 16], m![Ds % 16]>()
        .collect::<m![Ns % 4, Gs, Ds / 16], m![Ds % 16]>()
        .contract_outer::<m![Ns % 4, Gs, Ds / 32], m![Ds % 32], _, _, _>(&square_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ns % 4, Gs]>()
        .contract_lane::<m![Ns % 4, Gs], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0 / DS_F32)
        .vector_fp_binary(FpBinaryOp::AddF, EPS)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .transpose::<m![Ns % 4], m![Gs % 2 # 8]>()
        .commit_trim::<m![Gs % 2]>()
        .commit();



    let weight_vrf = load_norm_weight::<Cluster, Slice>(ctx, rms_weight);

    let rms_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![Ns % 4, Gs]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![Ns % 4, Gs]>()
        .collect::<m![1], m![Ns % 4, Gs]>()
        .to_vrf();

    ctx.main
        .begin(x.view())
        .fetch::<m![Ns % 4, Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ns % 4, Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns % 4, Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![Ns % 4, Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

fn root_mean_square_dot_e209<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Ds]>,
) -> VrfTensor<f32, Chip, Cluster, Slice, m![Ns % 4 # 8]> {
    // E209: head마다 같은 BF16 x를 TRF와 stream에서 읽어 모든 Ds256의 제곱합을 계산한다.
    // head 축은 양쪽 operand의 같은 논리 축이며 다른 head의 값과 곱하지 않는다.
    let square_trf: TrfTensor<bf16, Chip, Cluster, Slice, m![1], m![Ns % 4, Ds]> = ctx.sub
        .begin(x.view()).fetch::<m![Ns % 4, Ds / 16], m![Ds % 16]>()
        .collect::<m![Ns % 4, Ds / 16], m![Ds % 16]>().to_trf();
    let rms: DmTensor<f32, Chip, Cluster, Slice, m![Ns % 4]> = ctx.main
        .begin(x.view()).fetch::<m![Ns % 4, Ds / 16], m![Ds % 16]>()
        .collect::<m![Ns % 4, Ds / 16], m![Ds % 16]>()
        .contract_outer::<m![Ns % 4, Ds / 32], m![Ds % 32], _, _, _>(&square_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ns % 4]>()
        .contract_lane::<m![Ns % 4], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0 / DS_F32)
        .vector_fp_binary(FpBinaryOp::AddF, EPS)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .transpose::<m![Ns % 4 / 2], m![Ns % 2 # 8]>()
        .commit_trim::<m![Ns % 2]>()
        .commit();



    ctx.sub
        .begin(rms.view())
        .fetch::<m![1], m![Ns % 4 # 8]>()
        .collect::<m![1], m![Ns % 4 # 8]>()
        .to_vrf()
}

pub(crate) fn normalize_value_dot_e209(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, crate::device::sliding::projection::ValueHeadSlices, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, crate::device::sliding::projection::ValueHeadSlices, m![Ds]> {
    type HeadSlices = crate::device::sliding::projection::ValueHeadSlices;
    // E209: head마다 같은 BF16 x를 TRF와 stream에서 읽어 모든 Ds256의 제곱합을 계산한다.
    // head 축은 양쪽 operand의 같은 논리 축이며 다른 head의 값과 곱하지 않는다.
    let square_trf: TrfTensor<bf16, Chip, Cluster, HeadSlices, m![1], m![Ds]> = ctx.sub
        .begin(x.view()).fetch::<m![Ds / 16], m![Ds % 16]>()
        .collect::<m![Ds / 16], m![Ds % 16]>().to_trf();
    let rms: DmTensor<f32, Chip, Cluster, HeadSlices, m![1 # 8]> = ctx.main
        .begin(x.view()).fetch::<m![Ds / 16], m![Ds % 16]>()
        .collect::<m![Ds / 16], m![Ds % 16]>()
        .contract_outer::<m![Ds / 32], m![Ds % 32], _, _, _>(&square_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![1]>()
        .contract_lane::<m![1], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0 / DS_F32)
        .vector_fp_binary(FpBinaryOp::AddF, EPS)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: VrfTensor<f32, Chip, Cluster, HeadSlices, m![1 # 8]> = ctx.sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    ctx.main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_div(&rms)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

pub(crate) fn normalize_key_dot_e209(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Ds]>,
    rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Ds]> {
    let rms_vrf = root_mean_square_dot_e209(ctx, x);
    let weight_vrf = load_norm_weight::<Cluster, Slice>(ctx, rms_weight);

    scale_by_rms_and_weight(ctx, x, &rms_vrf, &weight_vrf)
}


// E211: Contraction 뒤의 Vector에서 head RMS의 sqrt까지 계산한다.
pub(crate) fn normalize_query_dot_e211<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Gs, Ds]>,
    rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Gs, Ds]> {
    // E211: head마다 같은 BF16 x를 TRF와 stream에서 읽어 모든 Ds256의 제곱합을 계산한다.
    // head 축은 양쪽 operand의 같은 논리 축이며 다른 head의 값과 곱하지 않는다.
    let square_trf: TrfTensor<bf16, Chip, Cluster, Slice, m![1], m![Ns % 4, Gs, Ds]> = ctx.sub
        .begin(x.view()).fetch::<m![Ns % 4, Gs, Ds / 16], m![Ds % 16]>()
        .collect::<m![Ns % 4, Gs, Ds / 16], m![Ds % 16]>().to_trf();
    let rms: DmTensor<f32, Chip, Cluster, Slice, m![Ns % 4, Gs]> = ctx.main
        .begin(x.view()).fetch::<m![Ns % 4, Gs, Ds / 16], m![Ds % 16]>()
        .collect::<m![Ns % 4, Gs, Ds / 16], m![Ds % 16]>()
        .contract_outer::<m![Ns % 4, Gs, Ds / 32], m![Ds % 32], _, _, _>(&square_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ns % 4, Gs]>()
        .contract_lane::<m![Ns % 4, Gs], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0 / DS_F32)
        .vector_fp_binary(FpBinaryOp::AddF, EPS)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .transpose::<m![Ns % 4], m![Gs % 2 # 8]>()
        .commit_trim::<m![Gs % 2]>()
        .commit();



    let weight_vrf = load_norm_weight::<Cluster, Slice>(ctx, rms_weight);

    let rms_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![Ns % 4, Gs]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![Ns % 4, Gs]>()
        .collect::<m![1], m![Ns % 4, Gs]>()
        .to_vrf();

    ctx.main
        .begin(x.view())
        .fetch::<m![Ns % 4, Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ns % 4, Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns % 4, Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight_vrf)
        .vector_widen_concat::<m![Ns % 4, Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

fn root_mean_square_dot_e211<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Ds]>,
) -> VrfTensor<f32, Chip, Cluster, Slice, m![Ns % 4 # 8]> {
    // E211: head마다 같은 BF16 x를 TRF와 stream에서 읽어 모든 Ds256의 제곱합을 계산한다.
    // head 축은 양쪽 operand의 같은 논리 축이며 다른 head의 값과 곱하지 않는다.
    let square_trf: TrfTensor<bf16, Chip, Cluster, Slice, m![1], m![Ns % 4, Ds]> = ctx.sub
        .begin(x.view()).fetch::<m![Ns % 4, Ds / 16], m![Ds % 16]>()
        .collect::<m![Ns % 4, Ds / 16], m![Ds % 16]>().to_trf();
    let rms: DmTensor<f32, Chip, Cluster, Slice, m![Ns % 4]> = ctx.main
        .begin(x.view()).fetch::<m![Ns % 4, Ds / 16], m![Ds % 16]>()
        .collect::<m![Ns % 4, Ds / 16], m![Ds % 16]>()
        .contract_outer::<m![Ns % 4, Ds / 32], m![Ds % 32], _, _, _>(&square_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ns % 4]>()
        .contract_lane::<m![Ns % 4], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0 / DS_F32)
        .vector_fp_binary(FpBinaryOp::AddF, EPS)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .transpose::<m![Ns % 4 / 2], m![Ns % 2 # 8]>()
        .commit_trim::<m![Ns % 2]>()
        .commit();



    ctx.sub
        .begin(rms.view())
        .fetch::<m![1], m![Ns % 4 # 8]>()
        .collect::<m![1], m![Ns % 4 # 8]>()
        .to_vrf()
}

pub(crate) fn normalize_value_dot_e211(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, crate::device::sliding::projection::ValueHeadSlices, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, crate::device::sliding::projection::ValueHeadSlices, m![Ds]> {
    type HeadSlices = crate::device::sliding::projection::ValueHeadSlices;
    // E211: head마다 같은 BF16 x를 TRF와 stream에서 읽어 모든 Ds256의 제곱합을 계산한다.
    // head 축은 양쪽 operand의 같은 논리 축이며 다른 head의 값과 곱하지 않는다.
    let square_trf: TrfTensor<bf16, Chip, Cluster, HeadSlices, m![1], m![Ds]> = ctx.sub
        .begin(x.view()).fetch::<m![Ds / 16], m![Ds % 16]>()
        .collect::<m![Ds / 16], m![Ds % 16]>().to_trf();
    let rms: DmTensor<f32, Chip, Cluster, HeadSlices, m![1 # 8]> = ctx.main
        .begin(x.view()).fetch::<m![Ds / 16], m![Ds % 16]>()
        .collect::<m![Ds / 16], m![Ds % 16]>()
        .contract_outer::<m![Ds / 32], m![Ds % 32], _, _, _>(&square_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![1]>()
        .contract_lane::<m![1], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0 / DS_F32)
        .vector_fp_binary(FpBinaryOp::AddF, EPS)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: VrfTensor<f32, Chip, Cluster, HeadSlices, m![1 # 8]> = ctx.sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    ctx.main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &rms)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

pub(crate) fn normalize_key_dot_e211(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Ds]>,
    rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Ds]> {
    let rms_vrf = root_mean_square_dot_e211(ctx, x);
    let weight_vrf = load_norm_weight::<Cluster, Slice>(ctx, rms_weight);

    scale_by_inverse_rms_e211(ctx, x, &rms_vrf, &weight_vrf)
}


fn scale_by_inverse_rms_e211<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Ds]>,
    rms_vrf: &VrfTensor<f32, Chip, Cluster, Slice, m![Ns % 4 # 8]>,
    weight_vrf: &VrfTensor<f32, Chip, Cluster, Slice, m![Ds]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns % 4, Ds]> {
    ctx.main
        .begin(x.view())
        .fetch::<m![Ns % 4, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ns % 4, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns % 4, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), weight_vrf)
        .vector_widen_concat::<m![Ns % 4, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}
