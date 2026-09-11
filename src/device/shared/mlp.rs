
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Dummy2, H, L};
use crate::device::layout::{Cluster, Slice};

const INVSQRT2: f32 = 0.70710678118f32;

type UpGateClusters = m![L / 7680];
pub(crate) type UpGateRows = m![L / 60 % 128, 1 # 2];
pub(crate) type UpGateRowsByColumns = m![L / 60 % 128, H / 1920];
pub(crate) type UpGateRowsPaired = m![L / 120 % 64, 1 # 4];

pub(crate) fn project_up_and_gate(
    ctx: &mut Context,
    x_trf: &TrfTensor<bf16, Chip, UpGateClusters, UpGateRowsByColumns, m![1], m![H % 1920]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
) -> (
    DmTensor<bf16, Chip, UpGateClusters, UpGateRows, m![L % 60]>,
    DmTensor<bf16, Chip, UpGateClusters, UpGateRows, m![L % 60]>,
) {
    const ROWS_PER_SLICE: usize = 60;
    const ROWS_PER_PASS: usize = 12;
    const PASSES: usize = ROWS_PER_SLICE / ROWS_PER_PASS;

    let mut up: DmTensor<bf16, Chip, UpGateClusters, UpGateRows, m![L % 60]> = DmTensor::new();

    // F4를 정확한 F8 값으로 복원한 뒤, 각 16원소 dot에 block scale을 적용한다.
    let up_weight_packed: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60, H % 1920]> =
        up_weight_packed.to_dm(&mut ctx.tdma);
    // FP4를 FP8에 정확히 복원한다. 블록 scale은 16원소 부분합 뒤에 적용한다.
    let up_weight_packed: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60, H % 1920]> = ctx.main
        .begin(up_weight_packed.view())
        .fetch::<m![L % 60], m![H % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![L % 60, H / 32 % 60], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();

    let up_weight_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60, H / 16 % 120]> =
        up_weight_scale.to_dm(&mut ctx.tdma);

    for i in 0..PASSES {
        let up_weight_scale_vrf: VrfTensor<f32, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60 = 12, H / 16 % 120]> = ctx
            .sub
            .begin(
                up_weight_scale
                    .view()
                    .tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H / 16 % 120]>(12 * i),
            )
            .fetch::<m![L % 60 = 12], m![H / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![L % 60 = 12, H / 128 % 15], m![H / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(up_weight_packed.view().tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H % 1920]>(12 * i))
            .fetch::<m![L % 60 = 12, H / 32 % 60], m![H % 32]>()
            .fetch_table_lookup::<bf16>()
            .collect::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
            .contract_outer::<m![L % 60 = 12, H / 32 % 60], m![H % 32], _, _, _>(&x_trf)
            .contract_packet::<m![H / 16 % 2]>()
            .contract_time::<m![L % 60 = 12, H / 32 % 60]>()
            .contract_lane::<m![L % 60 = 12, H / 16 % 120], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &up_weight_scale_vrf)
            .vector_intra_slice_reduce::<H, m![L % 60 = 12], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<UpGateRows, m![L % 60 = 12]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![L % 60 = 12 / 4], m![L % 60 = 12 % 4 # 16]>()
            .commit_trim::<m![L % 60 = 12 % 4]>()
            .commit_view(up.view_mut().tile::<m![L % 60], 12, m![L % 60 = 12 #{!} 60]>(12 * i));
    }

    let mut gate: DmTensor<bf16, Chip, UpGateClusters, UpGateRows, m![L % 60]> = DmTensor::new();

    let gate_weight_packed: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60, H % 1920]> =
        gate_weight_packed.to_dm(&mut ctx.tdma);
    // FP4를 FP8에 정확히 복원한다. 블록 scale은 16원소 부분합 뒤에 적용한다.
    let gate_weight_packed: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60, H % 1920]> = ctx.main
        .begin(gate_weight_packed.view())
        .fetch::<m![L % 60], m![H % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![L % 60, H / 32 % 60], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();

    let gate_weight_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60, H / 16 % 120]> =
        gate_weight_scale.to_dm(&mut ctx.tdma);

    for i in 0..PASSES {
        let gate_weight_scale_vrf: VrfTensor<f32, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60 = 12, H / 16 % 120]> = ctx
            .sub
            .begin(
                gate_weight_scale
                    .view()
                    .tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H / 16 % 120]>(12 * i),
            )
            .fetch::<m![L % 60 = 12], m![H / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![L % 60 = 12, H / 128 % 15], m![H / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(gate_weight_packed.view().tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H % 1920]>(12 * i))
            .fetch::<m![L % 60 = 12, H / 32 % 60], m![H % 32]>()
            .fetch_table_lookup::<bf16>()
            .collect::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
            .contract_outer::<m![L % 60 = 12, H / 32 % 60], m![H % 32], _, _, _>(&x_trf)
            .contract_packet::<m![H / 16 % 2]>()
            .contract_time::<m![L % 60 = 12, H / 32 % 60]>()
            .contract_lane::<m![L % 60 = 12, H / 16 % 120], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gate_weight_scale_vrf)
            .vector_intra_slice_reduce::<H, m![L % 60 = 12], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<UpGateRows, m![L % 60 = 12]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![L % 60 = 12 / 4], m![L % 60 = 12 % 4 # 16]>()
            .commit_trim::<m![L % 60 = 12 % 4]>()
            .commit_view(gate.view_mut().tile::<m![L % 60], 12, m![L % 60 = 12 #{!} 60]>(12 * i));
    }

    (up, gate)
}

pub(crate) fn feedforward(
    ctx: &mut Context,
    x: DmTensor<bf16, Chip, m![Dummy2], UpGateRowsByColumns, m![H % 1920]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    // 두 cluster에 같은 정규화 입력을 놓았으므로 cluster 이름만 출력 L 블록으로 재해석한다.
    let x: DmTensor<bf16, Chip, UpGateClusters, UpGateRowsByColumns, m![H % 1920]> = unsafe { x.reshape() };
    let x_trf: TrfTensor<bf16, Chip, UpGateClusters, UpGateRowsByColumns, m![1], m![H % 1920]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![H / 16 % 120], m![H % 16]>()
        .collect::<m![H / 16 % 120], m![H % 16]>()
        .to_trf();

    let (up, gate) = project_up_and_gate(
        ctx,
        &x_trf,
        up_weight_packed,
        gate_weight_packed,
        up_weight_scale,
        gate_weight_scale,
    );
    let x = geglu(ctx, up, gate, up_global_scale, gate_global_scale);
    let x: HbmTensor<bf16, Chip, m![L]> = x.to_hbm(&mut ctx.tdma);
    let x = x.to_dm(&mut ctx.tdma);
    let down = project_down(ctx, &x, down_weight_packed, down_weight_scale);

    let down_global_scale: DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> =
        down_global_scale.to_dm(&mut ctx.tdma);
    let down_global_scale_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = ctx
        .sub
        .begin(down_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let down: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = ctx
        .main
        .begin(down.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_global_scale_vrf)
        .vector_widen_concat::<m![H / 8], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    down
}

pub(crate) fn geglu(
    ctx: &mut Context,
    up: DmTensor<bf16, Chip, UpGateClusters, UpGateRows, m![L % 60]>,
    gate: DmTensor<bf16, Chip, UpGateClusters, UpGateRows, m![L % 60]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, UpGateClusters, UpGateRowsPaired, m![L % 120]> {
    let up: DmTensor<bf16, Chip, UpGateClusters, UpGateRowsPaired, m![L % 120]> = up.to_dm(&mut ctx.tdma);

    let up_global_scale: DmTensor<f32, Chip, UpGateClusters, UpGateRowsPaired, m![1 # 8]> =
        up_global_scale.to_dm(&mut ctx.tdma);
    let up_global_scale_vrf: VrfTensor<f32, Chip, UpGateClusters, UpGateRowsPaired, m![1 # 8]> = ctx
        .sub
        .begin(up_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let up: DmTensor<bf16, Chip, UpGateClusters, UpGateRowsPaired, m![L % 120]> = ctx
        .main
        .begin(up.view())
        .fetch::<m![1], m![L % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &up_global_scale_vrf)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit();

    let gate_global_scale: DmTensor<f32, Chip, UpGateClusters, UpGateRowsPaired, m![1 # 8]> =
        gate_global_scale.to_dm(&mut ctx.tdma);
    let gate_global_scale_vrf: VrfTensor<f32, Chip, UpGateClusters, UpGateRowsPaired, m![1 # 8]> = ctx
        .sub
        .begin(gate_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let gate: DmTensor<bf16, Chip, UpGateClusters, UpGateRowsPaired, m![L % 120]> = gate.to_dm(&mut ctx.tdma);

    let gate: DmTensor<bf16, Chip, UpGateClusters, UpGateRowsPaired, m![L % 120]> = ctx
        .main
        .begin(gate.view())
        .fetch::<m![1], m![L % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gate_global_scale_vrf)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit();

    let gelu: DmTensor<f32, Chip, UpGateClusters, UpGateRowsPaired, m![L % 120]> = ctx
        .sub
        .begin(gate.view())
        .fetch::<m![1], m![L % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), INVSQRT2)
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .commit_trim::<m![L % 8]>()
        .commit();

    let gelu_vrf: VrfTensor<f32, Chip, UpGateClusters, UpGateRowsPaired, m![L % 120]> = ctx
        .sub
        .begin(gelu.view())
        .fetch::<m![L / 8 % 15], m![L % 8]>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .to_vrf();

    ctx.main
        .begin(up.view())
        .fetch::<m![L / 8 % 15], m![L % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gelu_vrf)
        .vector_fp_div(2f32)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit()
}

type DownClusters = m![H / 1920];
pub(crate) type DownRows = m![H / 60 % 32, 1 # 8];
pub(crate) type DownRowsByColumns = m![H / 60 % 32, L / 1920];

pub(crate) fn project_down(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![L % 1920]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    let x_trf: TrfTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![1], m![L % 1920]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![L / 16 % 120], m![L % 16]>()
        .collect::<m![L / 16 % 120], m![L % 16]>()
        .to_trf();

    const ROWS_PER_SLICE: usize = 60;
    const ROWS_PER_PASS: usize = 12;
    const PASSES: usize = ROWS_PER_SLICE / ROWS_PER_PASS;

    let mut down: DmTensor<bf16, Chip, DownClusters, DownRows, m![H % 60]> = DmTensor::new();

    // L을 8개 slice에 먼저 분할한다. 복원·scale 적용도 contraction과 같은
    // 512개 slice에서 수행하여 bf16 행 전체의 재배치와 타일 복사를 없앤다.
    let down_weight_scale: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![H % 60, L / 16 % 120]> =
        down_weight_scale.to_dm(&mut ctx.tdma);

    let down_weight_packed: DmTensor<f4e2m1, Chip, DownClusters, DownRowsByColumns, m![H % 60, L % 1920]> =
        down_weight_packed.to_dm(&mut ctx.tdma);
    let down_weight_packed: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![H % 60, L % 1920]> = ctx.main
        .begin(down_weight_packed.view())
        .fetch::<m![H % 60], m![L % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![H % 60, L / 32 % 60], m![L % 32]>()
        .commit_trim::<m![L % 32]>()
        .commit();

    for i in 0..PASSES {
        let down_weight_scale_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]> = ctx
            .sub
            .begin(
                down_weight_scale
                    .view()
                    .tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L / 16 % 120]>(12 * i),
            )
            .fetch::<m![H % 60 = 12], m![L / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 128 % 15], m![L / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(down_weight_packed.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L % 1920]>(12 * i))
            .fetch::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .fetch_table_lookup::<bf16>()
            .collect::<m![H % 60 = 12, L / 16 % 120], m![L % 16]>()
            .contract_outer::<m![H % 60 = 12, L / 32 % 60], m![L % 32], _, _, _>(&x_trf)
            .contract_packet::<m![L / 16 % 2]>()
            .contract_time::<m![H % 60 = 12, L / 32 % 60]>()
            .contract_lane::<m![H % 60 = 12, L / 16 % 120], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_weight_scale_vrf)
            .vector_intra_slice_reduce::<L, m![H % 60 = 12], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<DownRows, m![H % 60 = 12]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![H % 60 = 12 / 4], m![H % 60 = 12 % 4 # 16]>()
            .commit_trim::<m![H % 60 = 12 % 4]>()
            .commit_view(down.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(12 * i));
    }

    let down: HbmTensor<bf16, Chip, m![H]> = down.to_hbm(&mut ctx.tdma);
    down.to_dm(&mut ctx.tdma)
}
