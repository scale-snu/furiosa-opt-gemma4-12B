
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

    let mut up: DmTensor<bf16, Chip, UpGateClusters, UpGateRows, m![L % 60]> = DmTensor::new();

    // F4 복원 결과를 별도 DM 행렬로 만들지 않고 scale 연산의 fetch에 연결한다.


    // Scale source의 전체 행을 30개씩 연속 로드하고 row partition을 column partition으로 교환한다.
    let up_scale_rows: DmTensor<f8e4m3, Chip, UpGateClusters, m![L / 60 % 128, L % 60 / 30], m![L % 30, H / 16]> =
        up_weight_scale.to_dm(&mut ctx.tdma);
    let up_weight_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60, H / 16 % 120 # 128]> = ctx.main
        .begin(up_scale_rows.view())
        .fetch::<m![L % 30, H / 1920], m![H / 16 % 120]>()
        .switch::<UpGateRowsByColumns, m![L % 30, L % 60 / 30]>(SwitchConfig::InterTranspose { slice1: 2, slice0: 1, time0: 1 })
        .collect::<m![L % 30, L % 60 / 30, H / 16 % 120 # 128 / 32], m![H / 16 % 120 # 128 % 32]>()
        .commit_trim::<m![H / 16 % 120 # 128 % 32]>()
        .commit();

    let mut up_buffer0: DmTensor<f4e2m1, Chip, UpGateClusters, m![L / 60 % 128, L % 60 = 36 / 18], m![L % 60 = 36 % 18, H]> = DmTensor::new();
    let mut up_buffer1: DmTensor<f4e2m1, Chip, UpGateClusters, m![L / 60 % 128, L % 60 = 24 / 12], m![L % 60 = 24 % 12, H]> = DmTensor::new();
    let mut up_decoded: DmTensor<bf16, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60, H % 1920]> = DmTensor::new();
    up_weight_packed.view().tile::<m![L % 60], 36, m![L / 60, L % 60 = 36 # 60, H]>(0).to_dm_view(&mut ctx.tdma, up_buffer0.view_mut());
    up_weight_packed.view().tile::<m![L % 60], 24, m![L / 60, L % 60 = 24 # 60, H]>(36).to_dm_view(&mut ctx.tdma, up_buffer1.view_mut());
    // FP4 nibble packing을 유지한 채 row/column을 교환하여 switch 전송량을 줄인다.
    let up_split0: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60 = 36, H % 1920]> = ctx.main
        .begin(up_buffer0.view())
        .fetch::<m![L % 60 = 36 % 18, H / 1920], m![H % 1920]>()
        .switch::<UpGateRowsByColumns, m![L % 60 = 36 % 18, L % 60 = 36 / 18]>(SwitchConfig::InterTranspose { slice1: 2, slice0: 1, time0: 1 })
        .collect::<m![L % 60 = 36 % 18, L % 60 = 36 / 18, H / 64 % 30], m![H % 64]>()
        .commit_trim::<m![H % 64]>()
        .commit();
    ctx.main
        .begin(up_split0.view())
        .fetch::<m![L % 60 = 36], m![H % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 36, H / 8 % 240], m![H % 8]>()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit_view(up_decoded.view_mut().tile::<m![L % 60], 36, m![L % 60 = 36 #{!} 60, H % 1920]>(0));

        let up_weight_scale_vrf: VrfTensor<f32, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60 = 12, H / 16 % 120]> = ctx
            .sub
            .begin(
                up_weight_scale
                    .view()
                    .tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H / 16 % 120 # 128]>(0),
            )
            .fetch::<m![L % 60 = 12], m![H / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![L % 60 = 12, H / 128 % 15], m![H / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(up_decoded.view().tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H % 1920]>(0))
            .fetch::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
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
            .commit_view(up.view_mut().tile::<m![L % 60], 12, m![L % 60 = 12 #{!} 60]>(0));

        let up_weight_scale_vrf: VrfTensor<f32, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60 = 12, H / 16 % 120]> = ctx
            .sub
            .begin(
                up_weight_scale
                    .view()
                    .tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H / 16 % 120 # 128]>(12),
            )
            .fetch::<m![L % 60 = 12], m![H / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![L % 60 = 12, H / 128 % 15], m![H / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(up_decoded.view().tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H % 1920]>(12))
            .fetch::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
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
            .commit_view(up.view_mut().tile::<m![L % 60], 12, m![L % 60 = 12 #{!} 60]>(12));

        let up_weight_scale_vrf: VrfTensor<f32, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60 = 12, H / 16 % 120]> = ctx
            .sub
            .begin(
                up_weight_scale
                    .view()
                    .tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H / 16 % 120 # 128]>(24),
            )
            .fetch::<m![L % 60 = 12], m![H / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![L % 60 = 12, H / 128 % 15], m![H / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(up_decoded.view().tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H % 1920]>(24))
            .fetch::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
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
            .commit_view(up.view_mut().tile::<m![L % 60], 12, m![L % 60 = 12 #{!} 60]>(24));
    // FP4 nibble packing을 유지한 채 row/column을 교환하여 switch 전송량을 줄인다.
    let up_split1: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60 = 24, H % 1920]> = ctx.main
        .begin(up_buffer1.view())
        .fetch::<m![L % 60 = 24 % 12, H / 1920], m![H % 1920]>()
        .switch::<UpGateRowsByColumns, m![L % 60 = 24 % 12, L % 60 = 24 / 12]>(SwitchConfig::InterTranspose { slice1: 2, slice0: 1, time0: 1 })
        .collect::<m![L % 60 = 24 % 12, L % 60 = 24 / 12, H / 64 % 30], m![H % 64]>()
        .commit_trim::<m![H % 64]>()
        .commit();
    ctx.main
        .begin(up_split1.view())
        .fetch::<m![L % 60 = 24], m![H % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 24, H / 8 % 240], m![H % 8]>()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit_view(up_decoded.view_mut().tile::<m![L % 60], 24, m![L % 60 = 24 #{!} 60, H % 1920]>(36));

        let up_weight_scale_vrf: VrfTensor<f32, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60 = 12, H / 16 % 120]> = ctx
            .sub
            .begin(
                up_weight_scale
                    .view()
                    .tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H / 16 % 120 # 128]>(36),
            )
            .fetch::<m![L % 60 = 12], m![H / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![L % 60 = 12, H / 128 % 15], m![H / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(up_decoded.view().tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H % 1920]>(36))
            .fetch::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
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
            .commit_view(up.view_mut().tile::<m![L % 60], 12, m![L % 60 = 12 #{!} 60]>(36));

        let up_weight_scale_vrf: VrfTensor<f32, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60 = 12, H / 16 % 120]> = ctx
            .sub
            .begin(
                up_weight_scale
                    .view()
                    .tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H / 16 % 120 # 128]>(48),
            )
            .fetch::<m![L % 60 = 12], m![H / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![L % 60 = 12, H / 128 % 15], m![H / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(up_decoded.view().tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H % 1920]>(48))
            .fetch::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
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
            .commit_view(up.view_mut().tile::<m![L % 60], 12, m![L % 60 = 12 #{!} 60]>(48));


    let mut gate: DmTensor<bf16, Chip, UpGateClusters, UpGateRows, m![L % 60]> = DmTensor::new();



    // Scale source의 전체 행을 30개씩 연속 로드하고 row partition을 column partition으로 교환한다.
    let gate_scale_rows: DmTensor<f8e4m3, Chip, UpGateClusters, m![L / 60 % 128, L % 60 / 30], m![L % 30, H / 16]> =
        gate_weight_scale.to_dm(&mut ctx.tdma);
    let gate_weight_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60, H / 16 % 120 # 128]> = ctx.main
        .begin(gate_scale_rows.view())
        .fetch::<m![L % 30, H / 1920], m![H / 16 % 120]>()
        .switch::<UpGateRowsByColumns, m![L % 30, L % 60 / 30]>(SwitchConfig::InterTranspose { slice1: 2, slice0: 1, time0: 1 })
        .collect::<m![L % 30, L % 60 / 30, H / 16 % 120 # 128 / 32], m![H / 16 % 120 # 128 % 32]>()
        .commit_trim::<m![H / 16 % 120 # 128 % 32]>()
        .commit();

    let mut gate_buffer0: DmTensor<f4e2m1, Chip, UpGateClusters, m![L / 60 % 128, L % 60 = 36 / 18], m![L % 60 = 36 % 18, H]> = DmTensor::new();
    let mut gate_buffer1: DmTensor<f4e2m1, Chip, UpGateClusters, m![L / 60 % 128, L % 60 = 24 / 12], m![L % 60 = 24 % 12, H]> = DmTensor::new();
    let mut gate_decoded: DmTensor<bf16, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60, H % 1920]> = DmTensor::new();
    gate_weight_packed.view().tile::<m![L % 60], 36, m![L / 60, L % 60 = 36 # 60, H]>(0).to_dm_view(&mut ctx.tdma, gate_buffer0.view_mut());
    gate_weight_packed.view().tile::<m![L % 60], 24, m![L / 60, L % 60 = 24 # 60, H]>(36).to_dm_view(&mut ctx.tdma, gate_buffer1.view_mut());
    // FP4 nibble packing을 유지한 채 row/column을 교환하여 switch 전송량을 줄인다.
    let gate_split0: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60 = 36, H % 1920]> = ctx.main
        .begin(gate_buffer0.view())
        .fetch::<m![L % 60 = 36 % 18, H / 1920], m![H % 1920]>()
        .switch::<UpGateRowsByColumns, m![L % 60 = 36 % 18, L % 60 = 36 / 18]>(SwitchConfig::InterTranspose { slice1: 2, slice0: 1, time0: 1 })
        .collect::<m![L % 60 = 36 % 18, L % 60 = 36 / 18, H / 64 % 30], m![H % 64]>()
        .commit_trim::<m![H % 64]>()
        .commit();
    ctx.main
        .begin(gate_split0.view())
        .fetch::<m![L % 60 = 36], m![H % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 36, H / 8 % 240], m![H % 8]>()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit_view(gate_decoded.view_mut().tile::<m![L % 60], 36, m![L % 60 = 36 #{!} 60, H % 1920]>(0));

        let gate_weight_scale_vrf: VrfTensor<f32, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60 = 12, H / 16 % 120]> = ctx
            .sub
            .begin(
                gate_weight_scale
                    .view()
                    .tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H / 16 % 120 # 128]>(0),
            )
            .fetch::<m![L % 60 = 12], m![H / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![L % 60 = 12, H / 128 % 15], m![H / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(gate_decoded.view().tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H % 1920]>(0))
            .fetch::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
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
            .commit_view(gate.view_mut().tile::<m![L % 60], 12, m![L % 60 = 12 #{!} 60]>(0));

        let gate_weight_scale_vrf: VrfTensor<f32, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60 = 12, H / 16 % 120]> = ctx
            .sub
            .begin(
                gate_weight_scale
                    .view()
                    .tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H / 16 % 120 # 128]>(12),
            )
            .fetch::<m![L % 60 = 12], m![H / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![L % 60 = 12, H / 128 % 15], m![H / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(gate_decoded.view().tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H % 1920]>(12))
            .fetch::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
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
            .commit_view(gate.view_mut().tile::<m![L % 60], 12, m![L % 60 = 12 #{!} 60]>(12));

        let gate_weight_scale_vrf: VrfTensor<f32, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60 = 12, H / 16 % 120]> = ctx
            .sub
            .begin(
                gate_weight_scale
                    .view()
                    .tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H / 16 % 120 # 128]>(24),
            )
            .fetch::<m![L % 60 = 12], m![H / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![L % 60 = 12, H / 128 % 15], m![H / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(gate_decoded.view().tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H % 1920]>(24))
            .fetch::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
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
            .commit_view(gate.view_mut().tile::<m![L % 60], 12, m![L % 60 = 12 #{!} 60]>(24));
    // FP4 nibble packing을 유지한 채 row/column을 교환하여 switch 전송량을 줄인다.
    let gate_split1: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60 = 24, H % 1920]> = ctx.main
        .begin(gate_buffer1.view())
        .fetch::<m![L % 60 = 24 % 12, H / 1920], m![H % 1920]>()
        .switch::<UpGateRowsByColumns, m![L % 60 = 24 % 12, L % 60 = 24 / 12]>(SwitchConfig::InterTranspose { slice1: 2, slice0: 1, time0: 1 })
        .collect::<m![L % 60 = 24 % 12, L % 60 = 24 / 12, H / 64 % 30], m![H % 64]>()
        .commit_trim::<m![H % 64]>()
        .commit();
    ctx.main
        .begin(gate_split1.view())
        .fetch::<m![L % 60 = 24], m![H % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 24, H / 8 % 240], m![H % 8]>()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit_view(gate_decoded.view_mut().tile::<m![L % 60], 24, m![L % 60 = 24 #{!} 60, H % 1920]>(36));

        let gate_weight_scale_vrf: VrfTensor<f32, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60 = 12, H / 16 % 120]> = ctx
            .sub
            .begin(
                gate_weight_scale
                    .view()
                    .tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H / 16 % 120 # 128]>(36),
            )
            .fetch::<m![L % 60 = 12], m![H / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![L % 60 = 12, H / 128 % 15], m![H / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(gate_decoded.view().tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H % 1920]>(36))
            .fetch::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
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
            .commit_view(gate.view_mut().tile::<m![L % 60], 12, m![L % 60 = 12 #{!} 60]>(36));

        let gate_weight_scale_vrf: VrfTensor<f32, Chip, UpGateClusters, UpGateRowsByColumns, m![L % 60 = 12, H / 16 % 120]> = ctx
            .sub
            .begin(
                gate_weight_scale
                    .view()
                    .tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H / 16 % 120 # 128]>(48),
            )
            .fetch::<m![L % 60 = 12], m![H / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![L % 60 = 12, H / 128 % 15], m![H / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(gate_decoded.view().tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H % 1920]>(48))
            .fetch::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
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
            .commit_view(gate.view_mut().tile::<m![L % 60], 12, m![L % 60 = 12 #{!} 60]>(48));


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

// Stage 1 FFN의 global scale과 tail 정규화가 공유하는 8-slice 배치.
pub(crate) type FfnSlices = m![1 # 32, H / 480];

pub(crate) fn feedforward_sharded(
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
) -> DmTensor<bf16, Chip, Cluster, FfnSlices, m![H % 480]> {
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
    // GeGLU의 64개 sparse slice×120 elements를 cluster별 연속 7680 elements로 모은다.
    let x: DmTensor<bf16, Chip, UpGateClusters, Slice, m![L % 7680]> = x.to_dm(&mut ctx.tdma);
    let x: HbmTensor<bf16, Chip, m![L]> = x.to_hbm(&mut ctx.tdma);
    let x = x.to_dm(&mut ctx.tdma);
    let down = project_down_hbm(ctx, &x, down_weight_packed, down_weight_scale);

    // HBM 결과를 H480×8에 직접 읽고 global scale의 bf16 commit 뒤 tail까지 유지한다.
    let down: DmTensor<bf16, Chip, Cluster, FfnSlices, m![H % 480]> = down.to_dm(&mut ctx.tdma);

    let down_global_scale: DmTensor<f32, Chip, Cluster, FfnSlices, m![1 # 8]> =
        down_global_scale.to_dm(&mut ctx.tdma);
    let down_global_scale_vrf: VrfTensor<f32, Chip, Cluster, FfnSlices, m![1 # 8]> = ctx
        .sub
        .begin(down_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let down: DmTensor<bf16, Chip, Cluster, FfnSlices, m![H % 480]> = ctx
        .main
        .begin(down.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_global_scale_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    down
}

pub(crate) fn feedforward_sharded_unscaled(
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
) -> DmTensor<bf16, Chip, Cluster, FfnSlices, m![H % 480]> {
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
    // GeGLU의 64개 sparse slice×120 elements를 cluster별 연속 7680 elements로 모은다.
    let x: DmTensor<bf16, Chip, UpGateClusters, Slice, m![L % 7680]> = x.to_dm(&mut ctx.tdma);
    let x: HbmTensor<bf16, Chip, m![L]> = x.to_hbm(&mut ctx.tdma);
    let x = x.to_dm(&mut ctx.tdma);
    let down = project_down_hbm(ctx, &x, down_weight_packed, down_weight_scale);

    // down dot의 bf16 결과를 보존한다. 행렬 global scale은 post-FFN norm에서 적용한다.
    down.to_dm(&mut ctx.tdma)
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
    // 기존 DM 반환 계약은 HBM core 뒤 동일한 to_dm 경로로 보존한다.
    project_down_hbm(ctx, x, down_weight_packed, down_weight_scale).to_dm(&mut ctx.tdma)
}

pub(crate) fn project_down_hbm(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![L % 1920]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
) -> HbmTensor<bf16, Chip, m![H]> {
    let x_trf: TrfTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![1], m![L % 1920]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![L / 16 % 120], m![L % 16]>()
        .collect::<m![L / 16 % 120], m![L % 16]>()
        .to_trf();


    let mut down: DmTensor<bf16, Chip, DownClusters, DownRows, m![H % 60]> = DmTensor::new();

    // L을 8개 slice에 먼저 분할한다. 복원·scale 적용도 contraction과 같은
    // 512개 slice에서 수행하여 bf16 행 전체의 재배치와 타일 복사를 없앤다.
    // Scale source의 전체 행을 15개씩 연속 로드하고 row partition을 column partition으로 교환한다.
    let down_scale_rows: DmTensor<f8e4m3, Chip, DownClusters, m![H / 60 % 32, H % 60 / 15, 1 # 2], m![H % 15, L / 16]> =
        down_weight_scale.to_dm(&mut ctx.tdma);
    let down_weight_scale: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![H % 60, 1 # 2, L / 16 % 120 # 128]> = ctx.main
        .begin(down_scale_rows.view())
        .fetch::<m![H % 15, L / 1920], m![L / 16 % 120]>()
        .switch::<DownRowsByColumns, m![H % 15, H % 60 / 15, 1 # 2]>(SwitchConfig::InterTranspose { slice1: 8, slice0: 1, time0: 1 })
        .collect::<m![H % 15, H % 60 / 15, 1 # 2, L / 16 % 120 # 128 / 32], m![L / 16 % 120 # 128 % 32]>()
        .commit_trim::<m![L / 16 % 120 # 128 % 32]>()
        .commit();



    let mut down_buffer0: DmTensor<f4e2m1, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 36, L % 1920]> = DmTensor::new();
    let mut down_buffer1: DmTensor<f4e2m1, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 24, L % 1920]> = DmTensor::new();
    let mut down_decoded: DmTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![H % 60, L % 1920]> = DmTensor::new();
    down_weight_packed.view().tile::<m![H % 60], 36, m![H / 60, H % 60 = 36 # 60, L]>(0).to_dm_view(&mut ctx.tdma, down_buffer0.view_mut());
    down_weight_packed.view().tile::<m![H % 60], 24, m![H / 60, H % 60 = 24 # 60, L]>(36).to_dm_view(&mut ctx.tdma, down_buffer1.view_mut());
    ctx.main
        .begin(down_buffer0.view())
        .fetch::<m![H % 60 = 36], m![L % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![H % 60 = 36, L / 8 % 240], m![L % 8]>()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit_view(down_decoded.view_mut().tile::<m![H % 60], 36, m![H % 60 = 36 #{!} 60, L % 1920]>(0));

        let down_weight_scale_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]> = ctx
            .sub
            .begin(
                down_weight_scale
                    .view()
                    .tile::<m![H % 60], 12, m![H % 60 = 12 # 60, 1 # 2, L / 16 % 120 # 128]>(0),
            )
            .fetch::<m![H % 60 = 12], m![L / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 128 % 15], m![L / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(down_decoded.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L % 1920]>(0))
            .fetch::<m![H % 60 = 12, L / 16 % 120], m![L % 16]>()
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
            .commit_view(down.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(0));

        let down_weight_scale_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]> = ctx
            .sub
            .begin(
                down_weight_scale
                    .view()
                    .tile::<m![H % 60], 12, m![H % 60 = 12 # 60, 1 # 2, L / 16 % 120 # 128]>(12),
            )
            .fetch::<m![H % 60 = 12], m![L / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 128 % 15], m![L / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(down_decoded.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L % 1920]>(12))
            .fetch::<m![H % 60 = 12, L / 16 % 120], m![L % 16]>()
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
            .commit_view(down.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(12));

        let down_weight_scale_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]> = ctx
            .sub
            .begin(
                down_weight_scale
                    .view()
                    .tile::<m![H % 60], 12, m![H % 60 = 12 # 60, 1 # 2, L / 16 % 120 # 128]>(24),
            )
            .fetch::<m![H % 60 = 12], m![L / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 128 % 15], m![L / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(down_decoded.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L % 1920]>(24))
            .fetch::<m![H % 60 = 12, L / 16 % 120], m![L % 16]>()
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
            .commit_view(down.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(24));
    ctx.main
        .begin(down_buffer1.view())
        .fetch::<m![H % 60 = 24], m![L % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![H % 60 = 24, L / 8 % 240], m![L % 8]>()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit_view(down_decoded.view_mut().tile::<m![H % 60], 24, m![H % 60 = 24 #{!} 60, L % 1920]>(36));

        let down_weight_scale_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]> = ctx
            .sub
            .begin(
                down_weight_scale
                    .view()
                    .tile::<m![H % 60], 12, m![H % 60 = 12 # 60, 1 # 2, L / 16 % 120 # 128]>(36),
            )
            .fetch::<m![H % 60 = 12], m![L / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 128 % 15], m![L / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(down_decoded.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L % 1920]>(36))
            .fetch::<m![H % 60 = 12, L / 16 % 120], m![L % 16]>()
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
            .commit_view(down.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(36));

        let down_weight_scale_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]> = ctx
            .sub
            .begin(
                down_weight_scale
                    .view()
                    .tile::<m![H % 60], 12, m![H % 60 = 12 # 60, 1 # 2, L / 16 % 120 # 128]>(48),
            )
            .fetch::<m![H % 60 = 12], m![L / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 128 % 15], m![L / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(down_decoded.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L % 1920]>(48))
            .fetch::<m![H % 60 = 12, L / 16 % 120], m![L % 16]>()
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
            .commit_view(down.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(48));


    // 같은 cluster의 32개 sparse slice를 한 slice의 연속 1920개 값으로 모은다.
    let down: DmTensor<bf16, Chip, DownClusters, Slice, m![H % 1920]> = down.to_dm(&mut ctx.tdma);
    let down: HbmTensor<bf16, Chip, m![H]> = down.to_hbm(&mut ctx.tdma);
    down
}


// E139: down만 native FP8 두 항 경로를 사용하며 기존 helper는 그대로 둔다.
axes![E139Term = 2];

pub(crate) fn project_down_native_hbm_e139(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![L % 1920]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
) -> HbmTensor<bf16, Chip, m![H]> {
    // 입력 L1920마다 동적 scale을 계산하고 두 FP8 항을 실제 DM에 저장한다.
    // FP4 weight는 FP8에서 정확하다. 입력 BF16→hi/lo 근사만 새로운 수치 변화다.
    let input_scale: DmTensor<f32, Chip, DownClusters, DownRowsByColumns, m![1 # 8]> = ctx.main
        .begin(x.view()).fetch::<m![L / 16 % 120], m![L % 16]>().fetch_cast::<f32>()
        .collect::<m![L / 8 % 240], m![L % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffff).vector_reinterpret::<f32>()
        .vector_narrow_split::<m![L / 4 % 480], m![L % 4]>()
        .vector_intra_slice_reduce::<L, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_fp_div(256.0).vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1e-30 / 256.0)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    let input_scale_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![1 # 8]> = ctx.sub
        .begin(input_scale.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let mut terms: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![E139Term, L % 1920]> = DmTensor::new();
    ctx.main.begin(x.view()).fetch::<m![L / 16 % 120], m![L % 16]>().fetch_cast::<f32>()
        .collect::<m![L / 8 % 240], m![L % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 480], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &input_scale_vrf)
        .vector_widen_concat::<m![L / 8 % 240], m![L % 8]>().vector_final()
        .cast::<f8e4m3, m![L % 8 # 32]>().commit_trim::<m![L % 8]>()
        .commit_view(terms.view_mut().tile::<m![E139Term], 1, m![E139Term = 1 #{!} 2, L % 1920]>(0));
    let mut lo_buffer: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![L % 1920]> = DmTensor::new();
    // 같은 L1920 scale을 사용하고 hi VRF의 생존량만 480개로 제한한다.
    let hi_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![L % 1920 = 480]> = ctx.sub
        .begin(terms.view().tile::<m![E139Term], 1, m![E139Term = 1 # 2, L % 1920]>(0)
            .tile::<m![L % 1920], 480, m![E139Term = 1 # 2, L % 1920 = 480 # 1920]>(0))
        .fetch::<m![1], m![L % 1920 = 480]>().fetch_cast::<f32>()
        .collect::<m![L % 1920 = 480 / 8], m![L % 8]>().to_vrf();
    ctx.main.begin(x.view().tile::<m![L % 1920], 480, m![L % 1920 = 480 # 1920]>(0))
        .fetch::<m![1], m![L % 1920 = 480]>().fetch_cast::<f32>()
        .collect::<m![L % 1920 = 480 / 8], m![L % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 1920 = 480 / 4], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &input_scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
        .vector_widen_concat::<m![L % 1920 = 480 / 8], m![L % 8]>().vector_final()
        .cast::<f8e4m3, m![L % 8 # 32]>().commit_trim::<m![L % 8]>()
        .commit_view(lo_buffer.view_mut().tile::<m![L % 1920], 480, m![L % 1920 = 480 #{!} 1920]>(0));
    // 같은 L1920 scale을 사용하고 hi VRF의 생존량만 480개로 제한한다.
    let hi_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![L % 1920 = 480]> = ctx.sub
        .begin(terms.view().tile::<m![E139Term], 1, m![E139Term = 1 # 2, L % 1920]>(0)
            .tile::<m![L % 1920], 480, m![E139Term = 1 # 2, L % 1920 = 480 # 1920]>(480))
        .fetch::<m![1], m![L % 1920 = 480]>().fetch_cast::<f32>()
        .collect::<m![L % 1920 = 480 / 8], m![L % 8]>().to_vrf();
    ctx.main.begin(x.view().tile::<m![L % 1920], 480, m![L % 1920 = 480 # 1920]>(480))
        .fetch::<m![1], m![L % 1920 = 480]>().fetch_cast::<f32>()
        .collect::<m![L % 1920 = 480 / 8], m![L % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 1920 = 480 / 4], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &input_scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
        .vector_widen_concat::<m![L % 1920 = 480 / 8], m![L % 8]>().vector_final()
        .cast::<f8e4m3, m![L % 8 # 32]>().commit_trim::<m![L % 8]>()
        .commit_view(lo_buffer.view_mut().tile::<m![L % 1920], 480, m![L % 1920 = 480 #{!} 1920]>(480));
    // 같은 L1920 scale을 사용하고 hi VRF의 생존량만 480개로 제한한다.
    let hi_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![L % 1920 = 480]> = ctx.sub
        .begin(terms.view().tile::<m![E139Term], 1, m![E139Term = 1 # 2, L % 1920]>(0)
            .tile::<m![L % 1920], 480, m![E139Term = 1 # 2, L % 1920 = 480 # 1920]>(960))
        .fetch::<m![1], m![L % 1920 = 480]>().fetch_cast::<f32>()
        .collect::<m![L % 1920 = 480 / 8], m![L % 8]>().to_vrf();
    ctx.main.begin(x.view().tile::<m![L % 1920], 480, m![L % 1920 = 480 # 1920]>(960))
        .fetch::<m![1], m![L % 1920 = 480]>().fetch_cast::<f32>()
        .collect::<m![L % 1920 = 480 / 8], m![L % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 1920 = 480 / 4], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &input_scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
        .vector_widen_concat::<m![L % 1920 = 480 / 8], m![L % 8]>().vector_final()
        .cast::<f8e4m3, m![L % 8 # 32]>().commit_trim::<m![L % 8]>()
        .commit_view(lo_buffer.view_mut().tile::<m![L % 1920], 480, m![L % 1920 = 480 #{!} 1920]>(960));
    // 같은 L1920 scale을 사용하고 hi VRF의 생존량만 480개로 제한한다.
    let hi_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![L % 1920 = 480]> = ctx.sub
        .begin(terms.view().tile::<m![E139Term], 1, m![E139Term = 1 # 2, L % 1920]>(0)
            .tile::<m![L % 1920], 480, m![E139Term = 1 # 2, L % 1920 = 480 # 1920]>(1440))
        .fetch::<m![1], m![L % 1920 = 480]>().fetch_cast::<f32>()
        .collect::<m![L % 1920 = 480 / 8], m![L % 8]>().to_vrf();
    ctx.main.begin(x.view().tile::<m![L % 1920], 480, m![L % 1920 = 480 # 1920]>(1440))
        .fetch::<m![1], m![L % 1920 = 480]>().fetch_cast::<f32>()
        .collect::<m![L % 1920 = 480 / 8], m![L % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 1920 = 480 / 4], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &input_scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
        .vector_widen_concat::<m![L % 1920 = 480 / 8], m![L % 8]>().vector_final()
        .cast::<f8e4m3, m![L % 8 # 32]>().commit_trim::<m![L % 8]>()
        .commit_view(lo_buffer.view_mut().tile::<m![L % 1920], 480, m![L % 1920 = 480 #{!} 1920]>(1440));
    // 완성한 lo를 실제 term1로 전송하여 packed operand의 값을 보장한다.
    lo_buffer.view().to_dm_view(&mut ctx.tdma,
        terms.view_mut().tile::<m![E139Term], 1, m![E139Term = 1 #{!} 2, L % 1920]>(1));
    let x_trf: TrfTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![1], m![E139Term, L % 1920]> = ctx.sub
        .begin(terms.view()).fetch::<m![E139Term, L / 32 % 60], m![L % 32]>()
        .collect::<m![E139Term, L / 32 % 60], m![L % 32]>().to_trf();

    let mut down: DmTensor<bf16, Chip, DownClusters, DownRows, m![H % 60]> = DmTensor::new();

    // L을 8개 slice에 먼저 분할한다. 복원·scale 적용도 contraction과 같은
    // 512개 slice에서 수행하여 bf16 행 전체의 재배치와 타일 복사를 없앤다.
    // Scale source의 전체 행을 15개씩 연속 로드하고 row partition을 column partition으로 교환한다.
    let down_scale_rows: DmTensor<f8e4m3, Chip, DownClusters, m![H / 60 % 32, H % 60 / 15, 1 # 2], m![H % 15, L / 16]> =
        down_weight_scale.to_dm(&mut ctx.tdma);
    let down_weight_scale: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![H % 60, 1 # 2, L / 16 % 120 # 128]> = ctx.main
        .begin(down_scale_rows.view())
        .fetch::<m![H % 15, L / 1920], m![L / 16 % 120]>()
        .switch::<DownRowsByColumns, m![H % 15, H % 60 / 15, 1 # 2]>(SwitchConfig::InterTranspose { slice1: 8, slice0: 1, time0: 1 })
        .collect::<m![H % 15, H % 60 / 15, 1 # 2, L / 16 % 120 # 128 / 32], m![L / 16 % 120 # 128 % 32]>()
        .commit_trim::<m![L / 16 % 120 # 128 % 32]>()
        .commit();



    let mut down_buffer0: DmTensor<f4e2m1, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 36, L % 1920]> = DmTensor::new();
    let mut down_buffer1: DmTensor<f4e2m1, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 24, L % 1920]> = DmTensor::new();
    let mut down_decoded: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![H % 60, L % 1920]> = DmTensor::new();
    down_weight_packed.view().tile::<m![H % 60], 36, m![H / 60, H % 60 = 36 # 60, L]>(0).to_dm_view(&mut ctx.tdma, down_buffer0.view_mut());
    down_weight_packed.view().tile::<m![H % 60], 24, m![H / 60, H % 60 = 24 # 60, L]>(36).to_dm_view(&mut ctx.tdma, down_buffer1.view_mut());
    ctx.main
        .begin(down_buffer0.view())
        .fetch::<m![H % 60 = 36], m![L % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![H % 60 = 36, L / 32 % 60], m![L % 32]>()
        .commit_trim::<m![L % 32]>()
        .commit_view(down_decoded.view_mut().tile::<m![H % 60], 36, m![H % 60 = 36 #{!} 60, L % 1920]>(0));

        let down_weight_scale_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]> = ctx
            .sub
            .begin(
                down_weight_scale
                    .view()
                    .tile::<m![H % 60], 12, m![H % 60 = 12 # 60, 1 # 2, L / 16 % 120 # 128]>(0),
            )
            .fetch::<m![H % 60 = 12], m![L / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 128 % 15], m![L / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(down_decoded.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L % 1920]>(0))
            .fetch::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .collect::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .contract_outer::<m![H % 60 = 12, L / 32 % 60, E139Term], m![L % 32], _, _, _>(&x_trf)
            .contract_packet::<m![L / 16 % 2]>()
            .contract_time::<m![H % 60 = 12, L / 32 % 60]>()
            .contract_lane::<m![H % 60 = 12, L / 16 % 120], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &input_scale_vrf)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &down_weight_scale_vrf)
            .vector_intra_slice_reduce::<L, m![H % 60 = 12], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<DownRows, m![H % 60 = 12]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![H % 60 = 12 / 4], m![H % 60 = 12 % 4 # 16]>()
            .commit_trim::<m![H % 60 = 12 % 4]>()
            .commit_view(down.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(0));

        let down_weight_scale_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]> = ctx
            .sub
            .begin(
                down_weight_scale
                    .view()
                    .tile::<m![H % 60], 12, m![H % 60 = 12 # 60, 1 # 2, L / 16 % 120 # 128]>(12),
            )
            .fetch::<m![H % 60 = 12], m![L / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 128 % 15], m![L / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(down_decoded.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L % 1920]>(12))
            .fetch::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .collect::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .contract_outer::<m![H % 60 = 12, L / 32 % 60, E139Term], m![L % 32], _, _, _>(&x_trf)
            .contract_packet::<m![L / 16 % 2]>()
            .contract_time::<m![H % 60 = 12, L / 32 % 60]>()
            .contract_lane::<m![H % 60 = 12, L / 16 % 120], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &input_scale_vrf)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &down_weight_scale_vrf)
            .vector_intra_slice_reduce::<L, m![H % 60 = 12], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<DownRows, m![H % 60 = 12]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![H % 60 = 12 / 4], m![H % 60 = 12 % 4 # 16]>()
            .commit_trim::<m![H % 60 = 12 % 4]>()
            .commit_view(down.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(12));

        let down_weight_scale_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]> = ctx
            .sub
            .begin(
                down_weight_scale
                    .view()
                    .tile::<m![H % 60], 12, m![H % 60 = 12 # 60, 1 # 2, L / 16 % 120 # 128]>(24),
            )
            .fetch::<m![H % 60 = 12], m![L / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 128 % 15], m![L / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(down_decoded.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L % 1920]>(24))
            .fetch::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .collect::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .contract_outer::<m![H % 60 = 12, L / 32 % 60, E139Term], m![L % 32], _, _, _>(&x_trf)
            .contract_packet::<m![L / 16 % 2]>()
            .contract_time::<m![H % 60 = 12, L / 32 % 60]>()
            .contract_lane::<m![H % 60 = 12, L / 16 % 120], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &input_scale_vrf)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &down_weight_scale_vrf)
            .vector_intra_slice_reduce::<L, m![H % 60 = 12], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<DownRows, m![H % 60 = 12]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![H % 60 = 12 / 4], m![H % 60 = 12 % 4 # 16]>()
            .commit_trim::<m![H % 60 = 12 % 4]>()
            .commit_view(down.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(24));
    ctx.main
        .begin(down_buffer1.view())
        .fetch::<m![H % 60 = 24], m![L % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![H % 60 = 24, L / 32 % 60], m![L % 32]>()
        .commit_trim::<m![L % 32]>()
        .commit_view(down_decoded.view_mut().tile::<m![H % 60], 24, m![H % 60 = 24 #{!} 60, L % 1920]>(36));

        let down_weight_scale_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]> = ctx
            .sub
            .begin(
                down_weight_scale
                    .view()
                    .tile::<m![H % 60], 12, m![H % 60 = 12 # 60, 1 # 2, L / 16 % 120 # 128]>(36),
            )
            .fetch::<m![H % 60 = 12], m![L / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 128 % 15], m![L / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(down_decoded.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L % 1920]>(36))
            .fetch::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .collect::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .contract_outer::<m![H % 60 = 12, L / 32 % 60, E139Term], m![L % 32], _, _, _>(&x_trf)
            .contract_packet::<m![L / 16 % 2]>()
            .contract_time::<m![H % 60 = 12, L / 32 % 60]>()
            .contract_lane::<m![H % 60 = 12, L / 16 % 120], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &input_scale_vrf)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &down_weight_scale_vrf)
            .vector_intra_slice_reduce::<L, m![H % 60 = 12], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<DownRows, m![H % 60 = 12]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![H % 60 = 12 / 4], m![H % 60 = 12 % 4 # 16]>()
            .commit_trim::<m![H % 60 = 12 % 4]>()
            .commit_view(down.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(36));

        let down_weight_scale_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]> = ctx
            .sub
            .begin(
                down_weight_scale
                    .view()
                    .tile::<m![H % 60], 12, m![H % 60 = 12 # 60, 1 # 2, L / 16 % 120 # 128]>(48),
            )
            .fetch::<m![H % 60 = 12], m![L / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 128 % 15], m![L / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(down_decoded.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L % 1920]>(48))
            .fetch::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .collect::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .contract_outer::<m![H % 60 = 12, L / 32 % 60, E139Term], m![L % 32], _, _, _>(&x_trf)
            .contract_packet::<m![L / 16 % 2]>()
            .contract_time::<m![H % 60 = 12, L / 32 % 60]>()
            .contract_lane::<m![H % 60 = 12, L / 16 % 120], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &input_scale_vrf)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &down_weight_scale_vrf)
            .vector_intra_slice_reduce::<L, m![H % 60 = 12], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<DownRows, m![H % 60 = 12]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![H % 60 = 12 / 4], m![H % 60 = 12 % 4 # 16]>()
            .commit_trim::<m![H % 60 = 12 % 4]>()
            .commit_view(down.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(48));


    // 같은 cluster의 32개 sparse slice를 한 slice의 연속 1920개 값으로 모은다.
    let down: DmTensor<bf16, Chip, DownClusters, Slice, m![H % 1920]> = down.to_dm(&mut ctx.tdma);
    let down: HbmTensor<bf16, Chip, m![H]> = down.to_hbm(&mut ctx.tdma);
    down
}

pub(crate) fn feedforward_native_down_e139(
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
) -> DmTensor<bf16, Chip, Cluster, FfnSlices, m![H % 480]> {
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
    // GeGLU의 64개 sparse slice×120 elements를 cluster별 연속 7680 elements로 모은다.
    let x: DmTensor<bf16, Chip, UpGateClusters, Slice, m![L % 7680]> = x.to_dm(&mut ctx.tdma);
    let x: HbmTensor<bf16, Chip, m![L]> = x.to_hbm(&mut ctx.tdma);
    let x = x.to_dm(&mut ctx.tdma);
    let down = project_down_native_hbm_e139(ctx, &x, down_weight_packed, down_weight_scale);

    // down dot의 bf16 결과를 보존한다. 행렬 global scale은 post-FFN norm에서 적용한다.
    down.to_dm(&mut ctx.tdma)
}

axes![E159Part = 8];

pub(crate) fn project_down_native_hbm_e159(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![L % 1920]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
) -> HbmTensor<bf16, Chip, m![H]> {
    // 입력 L1920마다 동적 scale을 계산하고 두 FP8 항을 실제 DM에 저장한다.
    // FP4 weight는 FP8에서 정확하다. 입력 BF16→hi/lo 근사만 새로운 수치 변화다.
    let input_scale: DmTensor<f32, Chip, DownClusters, DownRowsByColumns, m![1 # 8]> = ctx.main
        .begin(x.view()).fetch::<m![L / 16 % 120], m![L % 16]>().fetch_cast::<f32>()
        .collect::<m![L / 8 % 240], m![L % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffff).vector_reinterpret::<f32>()
        .vector_narrow_split::<m![L / 4 % 480], m![L % 4]>()
        .vector_intra_slice_reduce::<L, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_fp_div(256.0).vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1e-30 / 256.0)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    let input_scale_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![1 # 8]> = ctx.sub
        .begin(input_scale.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let mut terms: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![E139Term, L % 1920]> = DmTensor::new();
    ctx.main.begin(x.view()).fetch::<m![L / 16 % 120], m![L % 16]>().fetch_cast::<f32>()
        .collect::<m![L / 8 % 240], m![L % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 480], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &input_scale_vrf)
        .vector_widen_concat::<m![L / 8 % 240], m![L % 8]>().vector_final()
        .cast::<f8e4m3, m![L % 8 # 32]>().commit_trim::<m![L % 8]>()
        .commit_view(terms.view_mut().tile::<m![E139Term], 1, m![E139Term = 1 #{!} 2, L % 1920]>(0));
    // 같은 3840B owner를 재명명한다. term0의 hi1920은 Part0..3에 이미 존재한다.
    // Part4..7은 아직 쓰지 않은 lo 구간이며, 아래 단일 IndexWrite로만 채운다.
    let mut parts: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![E159Part, L % 480]> = unsafe { terms.reshape() };
    // 같은 L1920 scale을 사용하고 hi VRF의 생존량만 480개로 제한한다.
    let hi_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![L % 480]> = ctx.sub
        .begin(parts.view().tile::<m![E159Part], 1, m![E159Part = 1 # 8, L % 480]>(0))
        .fetch::<m![1], m![L % 480]>().fetch_cast::<f32>()
        .collect::<m![L / 8 % 60], m![L % 8]>().to_vrf();
    ctx.main.begin(x.view().tile::<m![L / 480 % 4], 1, m![L / 480 % 4 = 1 # 4, L % 480]>(0))
        .fetch::<m![1], m![L % 480]>().fetch_cast::<f32>()
        .collect::<m![L / 8 % 60], m![L % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 120], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &input_scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
        .vector_widen_concat::<m![L / 8 % 60], m![L % 8]>().vector_final()
        .cast::<f8e4m3, m![L % 8 # 32]>().commit_trim::<m![L % 8]>()
        .commit_view(parts.view_mut().tile::<m![E159Part], 1, m![E159Part = 1 #{!} 8, L % 480]>(4));
    // 같은 L1920 scale을 사용하고 hi VRF의 생존량만 480개로 제한한다.
    let hi_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![L % 480]> = ctx.sub
        .begin(parts.view().tile::<m![E159Part], 1, m![E159Part = 1 # 8, L % 480]>(1))
        .fetch::<m![1], m![L % 480]>().fetch_cast::<f32>()
        .collect::<m![L / 8 % 60], m![L % 8]>().to_vrf();
    ctx.main.begin(x.view().tile::<m![L / 480 % 4], 1, m![L / 480 % 4 = 1 # 4, L % 480]>(1))
        .fetch::<m![1], m![L % 480]>().fetch_cast::<f32>()
        .collect::<m![L / 8 % 60], m![L % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 120], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &input_scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
        .vector_widen_concat::<m![L / 8 % 60], m![L % 8]>().vector_final()
        .cast::<f8e4m3, m![L % 8 # 32]>().commit_trim::<m![L % 8]>()
        .commit_view(parts.view_mut().tile::<m![E159Part], 1, m![E159Part = 1 #{!} 8, L % 480]>(5));
    // 같은 L1920 scale을 사용하고 hi VRF의 생존량만 480개로 제한한다.
    let hi_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![L % 480]> = ctx.sub
        .begin(parts.view().tile::<m![E159Part], 1, m![E159Part = 1 # 8, L % 480]>(2))
        .fetch::<m![1], m![L % 480]>().fetch_cast::<f32>()
        .collect::<m![L / 8 % 60], m![L % 8]>().to_vrf();
    ctx.main.begin(x.view().tile::<m![L / 480 % 4], 1, m![L / 480 % 4 = 1 # 4, L % 480]>(2))
        .fetch::<m![1], m![L % 480]>().fetch_cast::<f32>()
        .collect::<m![L / 8 % 60], m![L % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 120], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &input_scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
        .vector_widen_concat::<m![L / 8 % 60], m![L % 8]>().vector_final()
        .cast::<f8e4m3, m![L % 8 # 32]>().commit_trim::<m![L % 8]>()
        .commit_view(parts.view_mut().tile::<m![E159Part], 1, m![E159Part = 1 #{!} 8, L % 480]>(6));
    // 같은 L1920 scale을 사용하고 hi VRF의 생존량만 480개로 제한한다.
    let hi_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![L % 480]> = ctx.sub
        .begin(parts.view().tile::<m![E159Part], 1, m![E159Part = 1 # 8, L % 480]>(3))
        .fetch::<m![1], m![L % 480]>().fetch_cast::<f32>()
        .collect::<m![L / 8 % 60], m![L % 8]>().to_vrf();
    ctx.main.begin(x.view().tile::<m![L / 480 % 4], 1, m![L / 480 % 4 = 1 # 4, L % 480]>(3))
        .fetch::<m![1], m![L % 480]>().fetch_cast::<f32>()
        .collect::<m![L / 8 % 60], m![L % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 120], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &input_scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
        .vector_widen_concat::<m![L / 8 % 60], m![L % 8]>().vector_final()
        .cast::<f8e4m3, m![L % 8 # 32]>().commit_trim::<m![L % 8]>()
        .commit_view(parts.view_mut().tile::<m![E159Part], 1, m![E159Part = 1 #{!} 8, L % 480]>(7));
    // Part0..3=hi, Part4..7=lo이므로 term-major 두 L1920과 같은 byte 순서다.
    // 기존 별도 lo_buffer와 term1 DMA는 없으며 모든 lo 쓰기를 마친 owner만 읽는다.
    let terms: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![E139Term, L % 1920]> = unsafe { parts.reshape() };
    let x_trf: TrfTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![1], m![E139Term, L % 1920]> = ctx.sub
        .begin(terms.view()).fetch::<m![E139Term, L / 32 % 60], m![L % 32]>()
        .collect::<m![E139Term, L / 32 % 60], m![L % 32]>().to_trf();

    let mut down: DmTensor<bf16, Chip, DownClusters, DownRows, m![H % 60]> = DmTensor::new();

    // L을 8개 slice에 먼저 분할한다. 복원·scale 적용도 contraction과 같은
    // 512개 slice에서 수행하여 bf16 행 전체의 재배치와 타일 복사를 없앤다.
    // Scale source의 전체 행을 15개씩 연속 로드하고 row partition을 column partition으로 교환한다.
    let down_scale_rows: DmTensor<f8e4m3, Chip, DownClusters, m![H / 60 % 32, H % 60 / 15, 1 # 2], m![H % 15, L / 16]> =
        down_weight_scale.to_dm(&mut ctx.tdma);
    let down_weight_scale: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![H % 60, 1 # 2, L / 16 % 120 # 128]> = ctx.main
        .begin(down_scale_rows.view())
        .fetch::<m![H % 15, L / 1920], m![L / 16 % 120]>()
        .switch::<DownRowsByColumns, m![H % 15, H % 60 / 15, 1 # 2]>(SwitchConfig::InterTranspose { slice1: 8, slice0: 1, time0: 1 })
        .collect::<m![H % 15, H % 60 / 15, 1 # 2, L / 16 % 120 # 128 / 32], m![L / 16 % 120 # 128 % 32]>()
        .commit_trim::<m![L / 16 % 120 # 128 % 32]>()
        .commit();



    let mut down_buffer0: DmTensor<f4e2m1, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 36, L % 1920]> = DmTensor::new();
    let mut down_buffer1: DmTensor<f4e2m1, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 24, L % 1920]> = DmTensor::new();
    let mut down_decoded: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![H % 60, L % 1920]> = DmTensor::new();
    down_weight_packed.view().tile::<m![H % 60], 36, m![H / 60, H % 60 = 36 # 60, L]>(0).to_dm_view(&mut ctx.tdma, down_buffer0.view_mut());
    down_weight_packed.view().tile::<m![H % 60], 24, m![H / 60, H % 60 = 24 # 60, L]>(36).to_dm_view(&mut ctx.tdma, down_buffer1.view_mut());
    ctx.main
        .begin(down_buffer0.view())
        .fetch::<m![H % 60 = 36], m![L % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![H % 60 = 36, L / 32 % 60], m![L % 32]>()
        .commit_trim::<m![L % 32]>()
        .commit_view(down_decoded.view_mut().tile::<m![H % 60], 36, m![H % 60 = 36 #{!} 60, L % 1920]>(0));

        let down_weight_scale_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]> = ctx
            .sub
            .begin(
                down_weight_scale
                    .view()
                    .tile::<m![H % 60], 12, m![H % 60 = 12 # 60, 1 # 2, L / 16 % 120 # 128]>(0),
            )
            .fetch::<m![H % 60 = 12], m![L / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 128 % 15], m![L / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(down_decoded.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L % 1920]>(0))
            .fetch::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .collect::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .contract_outer::<m![H % 60 = 12, L / 32 % 60, E139Term], m![L % 32], _, _, _>(&x_trf)
            .contract_packet::<m![L / 16 % 2]>()
            .contract_time::<m![H % 60 = 12, L / 32 % 60]>()
            .contract_lane::<m![H % 60 = 12, L / 16 % 120], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &input_scale_vrf)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &down_weight_scale_vrf)
            .vector_intra_slice_reduce::<L, m![H % 60 = 12], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<DownRows, m![H % 60 = 12]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![H % 60 = 12 / 4], m![H % 60 = 12 % 4 # 16]>()
            .commit_trim::<m![H % 60 = 12 % 4]>()
            .commit_view(down.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(0));

        let down_weight_scale_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]> = ctx
            .sub
            .begin(
                down_weight_scale
                    .view()
                    .tile::<m![H % 60], 12, m![H % 60 = 12 # 60, 1 # 2, L / 16 % 120 # 128]>(12),
            )
            .fetch::<m![H % 60 = 12], m![L / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 128 % 15], m![L / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(down_decoded.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L % 1920]>(12))
            .fetch::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .collect::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .contract_outer::<m![H % 60 = 12, L / 32 % 60, E139Term], m![L % 32], _, _, _>(&x_trf)
            .contract_packet::<m![L / 16 % 2]>()
            .contract_time::<m![H % 60 = 12, L / 32 % 60]>()
            .contract_lane::<m![H % 60 = 12, L / 16 % 120], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &input_scale_vrf)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &down_weight_scale_vrf)
            .vector_intra_slice_reduce::<L, m![H % 60 = 12], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<DownRows, m![H % 60 = 12]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![H % 60 = 12 / 4], m![H % 60 = 12 % 4 # 16]>()
            .commit_trim::<m![H % 60 = 12 % 4]>()
            .commit_view(down.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(12));

        let down_weight_scale_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]> = ctx
            .sub
            .begin(
                down_weight_scale
                    .view()
                    .tile::<m![H % 60], 12, m![H % 60 = 12 # 60, 1 # 2, L / 16 % 120 # 128]>(24),
            )
            .fetch::<m![H % 60 = 12], m![L / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 128 % 15], m![L / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(down_decoded.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L % 1920]>(24))
            .fetch::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .collect::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .contract_outer::<m![H % 60 = 12, L / 32 % 60, E139Term], m![L % 32], _, _, _>(&x_trf)
            .contract_packet::<m![L / 16 % 2]>()
            .contract_time::<m![H % 60 = 12, L / 32 % 60]>()
            .contract_lane::<m![H % 60 = 12, L / 16 % 120], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &input_scale_vrf)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &down_weight_scale_vrf)
            .vector_intra_slice_reduce::<L, m![H % 60 = 12], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<DownRows, m![H % 60 = 12]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![H % 60 = 12 / 4], m![H % 60 = 12 % 4 # 16]>()
            .commit_trim::<m![H % 60 = 12 % 4]>()
            .commit_view(down.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(24));
    ctx.main
        .begin(down_buffer1.view())
        .fetch::<m![H % 60 = 24], m![L % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![H % 60 = 24, L / 32 % 60], m![L % 32]>()
        .commit_trim::<m![L % 32]>()
        .commit_view(down_decoded.view_mut().tile::<m![H % 60], 24, m![H % 60 = 24 #{!} 60, L % 1920]>(36));

        let down_weight_scale_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]> = ctx
            .sub
            .begin(
                down_weight_scale
                    .view()
                    .tile::<m![H % 60], 12, m![H % 60 = 12 # 60, 1 # 2, L / 16 % 120 # 128]>(36),
            )
            .fetch::<m![H % 60 = 12], m![L / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 128 % 15], m![L / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(down_decoded.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L % 1920]>(36))
            .fetch::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .collect::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .contract_outer::<m![H % 60 = 12, L / 32 % 60, E139Term], m![L % 32], _, _, _>(&x_trf)
            .contract_packet::<m![L / 16 % 2]>()
            .contract_time::<m![H % 60 = 12, L / 32 % 60]>()
            .contract_lane::<m![H % 60 = 12, L / 16 % 120], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &input_scale_vrf)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &down_weight_scale_vrf)
            .vector_intra_slice_reduce::<L, m![H % 60 = 12], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<DownRows, m![H % 60 = 12]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![H % 60 = 12 / 4], m![H % 60 = 12 % 4 # 16]>()
            .commit_trim::<m![H % 60 = 12 % 4]>()
            .commit_view(down.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(36));

        let down_weight_scale_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]> = ctx
            .sub
            .begin(
                down_weight_scale
                    .view()
                    .tile::<m![H % 60], 12, m![H % 60 = 12 # 60, 1 # 2, L / 16 % 120 # 128]>(48),
            )
            .fetch::<m![H % 60 = 12], m![L / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 128 % 15], m![L / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(down_decoded.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L % 1920]>(48))
            .fetch::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .collect::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .contract_outer::<m![H % 60 = 12, L / 32 % 60, E139Term], m![L % 32], _, _, _>(&x_trf)
            .contract_packet::<m![L / 16 % 2]>()
            .contract_time::<m![H % 60 = 12, L / 32 % 60]>()
            .contract_lane::<m![H % 60 = 12, L / 16 % 120], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &input_scale_vrf)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &down_weight_scale_vrf)
            .vector_intra_slice_reduce::<L, m![H % 60 = 12], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<DownRows, m![H % 60 = 12]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![H % 60 = 12 / 4], m![H % 60 = 12 % 4 # 16]>()
            .commit_trim::<m![H % 60 = 12 % 4]>()
            .commit_view(down.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(48));


    // 같은 cluster의 32개 sparse slice를 한 slice의 연속 1920개 값으로 모은다.
    let down: DmTensor<bf16, Chip, DownClusters, Slice, m![H % 1920]> = down.to_dm(&mut ctx.tdma);
    let down: HbmTensor<bf16, Chip, m![H]> = down.to_hbm(&mut ctx.tdma);
    down
}

pub(crate) fn feedforward_native_down_e159(
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
) -> DmTensor<bf16, Chip, Cluster, FfnSlices, m![H % 480]> {
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
    // GeGLU의 64개 sparse slice×120 elements를 cluster별 연속 7680 elements로 모은다.
    let x: DmTensor<bf16, Chip, UpGateClusters, Slice, m![L % 7680]> = x.to_dm(&mut ctx.tdma);
    let x: HbmTensor<bf16, Chip, m![L]> = x.to_hbm(&mut ctx.tdma);
    let x = x.to_dm(&mut ctx.tdma);
    let down = project_down_native_hbm_e159(ctx, &x, down_weight_packed, down_weight_scale);

    // down dot의 bf16 결과를 보존한다. 행렬 global scale은 post-FFN norm에서 적용한다.
    down.to_dm(&mut ctx.tdma)
}

// E165: 동일 2×32×8 물리 geometry에서 출력 H480 블록만 cluster에 교차 배치한다.
// 입력 L1920은 HBM DMA가 새 두 cluster/32row에 실제 복제한다. reshape로 확대하지 않는다.
type E165DownClusters = m![H / 480 % 2];
type E165DownRows = m![H / 960, H / 60 % 8, 1 # 8];
type E165DownRowsByColumns = m![H / 960, H / 60 % 8, L / 1920];

pub(crate) fn project_down_native_hbm_e165(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, E165DownClusters, E165DownRowsByColumns, m![L % 1920]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
) -> HbmTensor<bf16, Chip, m![H]> {
    // 입력 L1920마다 동적 scale을 계산하고 두 FP8 항을 실제 DM에 저장한다.
    // FP4 weight는 FP8에서 정확하다. 입력 BF16→hi/lo 근사만 새로운 수치 변화다.
    let input_scale: DmTensor<f32, Chip, E165DownClusters, E165DownRowsByColumns, m![1 # 8]> = ctx.main
        .begin(x.view()).fetch::<m![L / 16 % 120], m![L % 16]>().fetch_cast::<f32>()
        .collect::<m![L / 8 % 240], m![L % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>().vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffff).vector_reinterpret::<f32>()
        .vector_narrow_split::<m![L / 4 % 480], m![L % 4]>()
        .vector_intra_slice_reduce::<L, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_fp_div(256.0).vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1e-30 / 256.0)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    let input_scale_vrf: VrfTensor<f32, Chip, E165DownClusters, E165DownRowsByColumns, m![1 # 8]> = ctx.sub
        .begin(input_scale.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();
    let mut terms: DmTensor<f8e4m3, Chip, E165DownClusters, E165DownRowsByColumns, m![E139Term, L % 1920]> = DmTensor::new();
    ctx.main.begin(x.view()).fetch::<m![L / 16 % 120], m![L % 16]>().fetch_cast::<f32>()
        .collect::<m![L / 8 % 240], m![L % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 480], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &input_scale_vrf)
        .vector_widen_concat::<m![L / 8 % 240], m![L % 8]>().vector_final()
        .cast::<f8e4m3, m![L % 8 # 32]>().commit_trim::<m![L % 8]>()
        .commit_view(terms.view_mut().tile::<m![E139Term], 1, m![E139Term = 1 #{!} 2, L % 1920]>(0));
    // 같은 3840B owner를 재명명한다. term0의 hi1920은 Part0..3에 이미 존재한다.
    // Part4..7은 아직 쓰지 않은 lo 구간이며, 아래 단일 IndexWrite로만 채운다.
    let mut parts: DmTensor<f8e4m3, Chip, E165DownClusters, E165DownRowsByColumns, m![E159Part, L % 480]> = unsafe { terms.reshape() };
    // 같은 L1920 scale을 사용하고 hi VRF의 생존량만 480개로 제한한다.
    let hi_vrf: VrfTensor<f32, Chip, E165DownClusters, E165DownRowsByColumns, m![L % 480]> = ctx.sub
        .begin(parts.view().tile::<m![E159Part], 1, m![E159Part = 1 # 8, L % 480]>(0))
        .fetch::<m![1], m![L % 480]>().fetch_cast::<f32>()
        .collect::<m![L / 8 % 60], m![L % 8]>().to_vrf();
    ctx.main.begin(x.view().tile::<m![L / 480 % 4], 1, m![L / 480 % 4 = 1 # 4, L % 480]>(0))
        .fetch::<m![1], m![L % 480]>().fetch_cast::<f32>()
        .collect::<m![L / 8 % 60], m![L % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 120], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &input_scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
        .vector_widen_concat::<m![L / 8 % 60], m![L % 8]>().vector_final()
        .cast::<f8e4m3, m![L % 8 # 32]>().commit_trim::<m![L % 8]>()
        .commit_view(parts.view_mut().tile::<m![E159Part], 1, m![E159Part = 1 #{!} 8, L % 480]>(4));
    // 같은 L1920 scale을 사용하고 hi VRF의 생존량만 480개로 제한한다.
    let hi_vrf: VrfTensor<f32, Chip, E165DownClusters, E165DownRowsByColumns, m![L % 480]> = ctx.sub
        .begin(parts.view().tile::<m![E159Part], 1, m![E159Part = 1 # 8, L % 480]>(1))
        .fetch::<m![1], m![L % 480]>().fetch_cast::<f32>()
        .collect::<m![L / 8 % 60], m![L % 8]>().to_vrf();
    ctx.main.begin(x.view().tile::<m![L / 480 % 4], 1, m![L / 480 % 4 = 1 # 4, L % 480]>(1))
        .fetch::<m![1], m![L % 480]>().fetch_cast::<f32>()
        .collect::<m![L / 8 % 60], m![L % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 120], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &input_scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
        .vector_widen_concat::<m![L / 8 % 60], m![L % 8]>().vector_final()
        .cast::<f8e4m3, m![L % 8 # 32]>().commit_trim::<m![L % 8]>()
        .commit_view(parts.view_mut().tile::<m![E159Part], 1, m![E159Part = 1 #{!} 8, L % 480]>(5));
    // 같은 L1920 scale을 사용하고 hi VRF의 생존량만 480개로 제한한다.
    let hi_vrf: VrfTensor<f32, Chip, E165DownClusters, E165DownRowsByColumns, m![L % 480]> = ctx.sub
        .begin(parts.view().tile::<m![E159Part], 1, m![E159Part = 1 # 8, L % 480]>(2))
        .fetch::<m![1], m![L % 480]>().fetch_cast::<f32>()
        .collect::<m![L / 8 % 60], m![L % 8]>().to_vrf();
    ctx.main.begin(x.view().tile::<m![L / 480 % 4], 1, m![L / 480 % 4 = 1 # 4, L % 480]>(2))
        .fetch::<m![1], m![L % 480]>().fetch_cast::<f32>()
        .collect::<m![L / 8 % 60], m![L % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 120], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &input_scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
        .vector_widen_concat::<m![L / 8 % 60], m![L % 8]>().vector_final()
        .cast::<f8e4m3, m![L % 8 # 32]>().commit_trim::<m![L % 8]>()
        .commit_view(parts.view_mut().tile::<m![E159Part], 1, m![E159Part = 1 #{!} 8, L % 480]>(6));
    // 같은 L1920 scale을 사용하고 hi VRF의 생존량만 480개로 제한한다.
    let hi_vrf: VrfTensor<f32, Chip, E165DownClusters, E165DownRowsByColumns, m![L % 480]> = ctx.sub
        .begin(parts.view().tile::<m![E159Part], 1, m![E159Part = 1 # 8, L % 480]>(3))
        .fetch::<m![1], m![L % 480]>().fetch_cast::<f32>()
        .collect::<m![L / 8 % 60], m![L % 8]>().to_vrf();
    ctx.main.begin(x.view().tile::<m![L / 480 % 4], 1, m![L / 480 % 4 = 1 # 4, L % 480]>(3))
        .fetch::<m![1], m![L % 480]>().fetch_cast::<f32>()
        .collect::<m![L / 8 % 60], m![L % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 120], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &input_scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
        .vector_widen_concat::<m![L / 8 % 60], m![L % 8]>().vector_final()
        .cast::<f8e4m3, m![L % 8 # 32]>().commit_trim::<m![L % 8]>()
        .commit_view(parts.view_mut().tile::<m![E159Part], 1, m![E159Part = 1 #{!} 8, L % 480]>(7));
    // Part0..3=hi, Part4..7=lo이므로 term-major 두 L1920과 같은 byte 순서다.
    // 기존 별도 lo_buffer와 term1 DMA는 없으며 모든 lo 쓰기를 마친 owner만 읽는다.
    let terms: DmTensor<f8e4m3, Chip, E165DownClusters, E165DownRowsByColumns, m![E139Term, L % 1920]> = unsafe { parts.reshape() };
    let x_trf: TrfTensor<f8e4m3, Chip, E165DownClusters, E165DownRowsByColumns, m![1], m![E139Term, L % 1920]> = ctx.sub
        .begin(terms.view()).fetch::<m![E139Term, L / 32 % 60], m![L % 32]>()
        .collect::<m![E139Term, L / 32 % 60], m![L % 32]>().to_trf();

    let mut down: DmTensor<bf16, Chip, E165DownClusters, E165DownRows, m![H % 60]> = DmTensor::new();

    // L을 8개 slice에 먼저 분할한다. 복원·scale 적용도 contraction과 같은
    // 512개 slice에서 수행하여 bf16 행 전체의 재배치와 타일 복사를 없앤다.
    // Scale source의 전체 행을 15개씩 연속 로드하고 row partition을 column partition으로 교환한다.
    let down_scale_rows: DmTensor<f8e4m3, Chip, E165DownClusters, m![H / 960, H / 60 % 8, H % 60 / 15, 1 # 2], m![H % 15, L / 16]> =
        down_weight_scale.to_dm(&mut ctx.tdma);
    let down_weight_scale: DmTensor<f8e4m3, Chip, E165DownClusters, E165DownRowsByColumns, m![H % 60, 1 # 2, L / 16 % 120 # 128]> = ctx.main
        .begin(down_scale_rows.view())
        .fetch::<m![H % 15, L / 1920], m![L / 16 % 120]>()
        .switch::<E165DownRowsByColumns, m![H % 15, H % 60 / 15, 1 # 2]>(SwitchConfig::InterTranspose { slice1: 8, slice0: 1, time0: 1 })
        .collect::<m![H % 15, H % 60 / 15, 1 # 2, L / 16 % 120 # 128 / 32], m![L / 16 % 120 # 128 % 32]>()
        .commit_trim::<m![L / 16 % 120 # 128 % 32]>()
        .commit();



    let mut down_buffer0: DmTensor<f4e2m1, Chip, E165DownClusters, E165DownRowsByColumns, m![H % 60 = 36, L % 1920]> = DmTensor::new();
    let mut down_buffer1: DmTensor<f4e2m1, Chip, E165DownClusters, E165DownRowsByColumns, m![H % 60 = 24, L % 1920]> = DmTensor::new();
    let mut down_decoded: DmTensor<f8e4m3, Chip, E165DownClusters, E165DownRowsByColumns, m![H % 60, L % 1920]> = DmTensor::new();
    down_weight_packed.view().tile::<m![H % 60], 36, m![H / 60, H % 60 = 36 # 60, L]>(0).to_dm_view(&mut ctx.tdma, down_buffer0.view_mut());
    down_weight_packed.view().tile::<m![H % 60], 24, m![H / 60, H % 60 = 24 # 60, L]>(36).to_dm_view(&mut ctx.tdma, down_buffer1.view_mut());
    ctx.main
        .begin(down_buffer0.view())
        .fetch::<m![H % 60 = 36], m![L % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![H % 60 = 36, L / 32 % 60], m![L % 32]>()
        .commit_trim::<m![L % 32]>()
        .commit_view(down_decoded.view_mut().tile::<m![H % 60], 36, m![H % 60 = 36 #{!} 60, L % 1920]>(0));

        let down_weight_scale_vrf: VrfTensor<f32, Chip, E165DownClusters, E165DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]> = ctx
            .sub
            .begin(
                down_weight_scale
                    .view()
                    .tile::<m![H % 60], 12, m![H % 60 = 12 # 60, 1 # 2, L / 16 % 120 # 128]>(0),
            )
            .fetch::<m![H % 60 = 12], m![L / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 128 % 15], m![L / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(down_decoded.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L % 1920]>(0))
            .fetch::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .collect::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .contract_outer::<m![H % 60 = 12, L / 32 % 60, E139Term], m![L % 32], _, _, _>(&x_trf)
            .contract_packet::<m![L / 16 % 2]>()
            .contract_time::<m![H % 60 = 12, L / 32 % 60]>()
            .contract_lane::<m![H % 60 = 12, L / 16 % 120], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &input_scale_vrf)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &down_weight_scale_vrf)
            .vector_intra_slice_reduce::<L, m![H % 60 = 12], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<E165DownRows, m![H % 60 = 12]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![H % 60 = 12 / 4], m![H % 60 = 12 % 4 # 16]>()
            .commit_trim::<m![H % 60 = 12 % 4]>()
            .commit_view(down.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(0));

        let down_weight_scale_vrf: VrfTensor<f32, Chip, E165DownClusters, E165DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]> = ctx
            .sub
            .begin(
                down_weight_scale
                    .view()
                    .tile::<m![H % 60], 12, m![H % 60 = 12 # 60, 1 # 2, L / 16 % 120 # 128]>(12),
            )
            .fetch::<m![H % 60 = 12], m![L / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 128 % 15], m![L / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(down_decoded.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L % 1920]>(12))
            .fetch::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .collect::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .contract_outer::<m![H % 60 = 12, L / 32 % 60, E139Term], m![L % 32], _, _, _>(&x_trf)
            .contract_packet::<m![L / 16 % 2]>()
            .contract_time::<m![H % 60 = 12, L / 32 % 60]>()
            .contract_lane::<m![H % 60 = 12, L / 16 % 120], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &input_scale_vrf)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &down_weight_scale_vrf)
            .vector_intra_slice_reduce::<L, m![H % 60 = 12], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<E165DownRows, m![H % 60 = 12]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![H % 60 = 12 / 4], m![H % 60 = 12 % 4 # 16]>()
            .commit_trim::<m![H % 60 = 12 % 4]>()
            .commit_view(down.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(12));

        let down_weight_scale_vrf: VrfTensor<f32, Chip, E165DownClusters, E165DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]> = ctx
            .sub
            .begin(
                down_weight_scale
                    .view()
                    .tile::<m![H % 60], 12, m![H % 60 = 12 # 60, 1 # 2, L / 16 % 120 # 128]>(24),
            )
            .fetch::<m![H % 60 = 12], m![L / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 128 % 15], m![L / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(down_decoded.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L % 1920]>(24))
            .fetch::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .collect::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .contract_outer::<m![H % 60 = 12, L / 32 % 60, E139Term], m![L % 32], _, _, _>(&x_trf)
            .contract_packet::<m![L / 16 % 2]>()
            .contract_time::<m![H % 60 = 12, L / 32 % 60]>()
            .contract_lane::<m![H % 60 = 12, L / 16 % 120], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &input_scale_vrf)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &down_weight_scale_vrf)
            .vector_intra_slice_reduce::<L, m![H % 60 = 12], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<E165DownRows, m![H % 60 = 12]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![H % 60 = 12 / 4], m![H % 60 = 12 % 4 # 16]>()
            .commit_trim::<m![H % 60 = 12 % 4]>()
            .commit_view(down.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(24));
    ctx.main
        .begin(down_buffer1.view())
        .fetch::<m![H % 60 = 24], m![L % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![H % 60 = 24, L / 32 % 60], m![L % 32]>()
        .commit_trim::<m![L % 32]>()
        .commit_view(down_decoded.view_mut().tile::<m![H % 60], 24, m![H % 60 = 24 #{!} 60, L % 1920]>(36));

        let down_weight_scale_vrf: VrfTensor<f32, Chip, E165DownClusters, E165DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]> = ctx
            .sub
            .begin(
                down_weight_scale
                    .view()
                    .tile::<m![H % 60], 12, m![H % 60 = 12 # 60, 1 # 2, L / 16 % 120 # 128]>(36),
            )
            .fetch::<m![H % 60 = 12], m![L / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 128 % 15], m![L / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(down_decoded.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L % 1920]>(36))
            .fetch::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .collect::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .contract_outer::<m![H % 60 = 12, L / 32 % 60, E139Term], m![L % 32], _, _, _>(&x_trf)
            .contract_packet::<m![L / 16 % 2]>()
            .contract_time::<m![H % 60 = 12, L / 32 % 60]>()
            .contract_lane::<m![H % 60 = 12, L / 16 % 120], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &input_scale_vrf)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &down_weight_scale_vrf)
            .vector_intra_slice_reduce::<L, m![H % 60 = 12], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<E165DownRows, m![H % 60 = 12]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![H % 60 = 12 / 4], m![H % 60 = 12 % 4 # 16]>()
            .commit_trim::<m![H % 60 = 12 % 4]>()
            .commit_view(down.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(36));

        let down_weight_scale_vrf: VrfTensor<f32, Chip, E165DownClusters, E165DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]> = ctx
            .sub
            .begin(
                down_weight_scale
                    .view()
                    .tile::<m![H % 60], 12, m![H % 60 = 12 # 60, 1 # 2, L / 16 % 120 # 128]>(48),
            )
            .fetch::<m![H % 60 = 12], m![L / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 128 % 15], m![L / 16 % 8]>()
            .to_vrf();

        ctx.main
            .begin(down_decoded.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L % 1920]>(48))
            .fetch::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .collect::<m![H % 60 = 12, L / 32 % 60], m![L % 32]>()
            .contract_outer::<m![H % 60 = 12, L / 32 % 60, E139Term], m![L % 32], _, _, _>(&x_trf)
            .contract_packet::<m![L / 16 % 2]>()
            .contract_time::<m![H % 60 = 12, L / 32 % 60]>()
            .contract_lane::<m![H % 60 = 12, L / 16 % 120], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &input_scale_vrf)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &down_weight_scale_vrf)
            .vector_intra_slice_reduce::<L, m![H % 60 = 12], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<E165DownRows, m![H % 60 = 12]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![H % 60 = 12 / 4], m![H % 60 = 12 % 4 # 16]>()
            .commit_trim::<m![H % 60 = 12 % 4]>()
            .commit_view(down.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(48));


    // H = 960*outer + 480*cluster + 60*row + local이다.
    // 32개 sparse slice를 각 cluster의 실제 4×H480 순서로 DMA gather한다.
    // 뒤 HBM DMA가 전역 H 좌표에 맞춰 네 개 480행 구간을 쓴다.
    let down: DmTensor<bf16, Chip, E165DownClusters, Slice, m![H / 960, H % 480]> = down.to_dm(&mut ctx.tdma);
    let down: HbmTensor<bf16, Chip, m![H]> = down.to_hbm(&mut ctx.tdma);
    down
}

pub(crate) fn feedforward_native_down_e165(
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
) -> DmTensor<bf16, Chip, Cluster, FfnSlices, m![H % 480]> {
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
    // GeGLU의 64개 sparse slice×120 elements를 cluster별 연속 7680 elements로 모은다.
    let x: DmTensor<bf16, Chip, UpGateClusters, Slice, m![L % 7680]> = x.to_dm(&mut ctx.tdma);
    let x: HbmTensor<bf16, Chip, m![L]> = x.to_hbm(&mut ctx.tdma);
    let x = x.to_dm(&mut ctx.tdma);
    let down = project_down_native_hbm_e165(ctx, &x, down_weight_packed, down_weight_scale);

    // down dot의 bf16 결과를 보존한다. 행렬 global scale은 post-FFN norm에서 적용한다.
    down.to_dm(&mut ctx.tdma)
}

// E173: E165 projection을 유지하고 down 결과를 실제 H240×16 slice로 읽는다.
pub(crate) fn feedforward_native_down_tail16_e173(
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
) -> DmTensor<bf16, Chip, Cluster, m![1 # 16, H / 240], m![H % 240]> {
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
    // GeGLU의 64개 sparse slice×120 elements를 cluster별 연속 7680 elements로 모은다.
    let x: DmTensor<bf16, Chip, UpGateClusters, Slice, m![L % 7680]> = x.to_dm(&mut ctx.tdma);
    let x: HbmTensor<bf16, Chip, m![L]> = x.to_hbm(&mut ctx.tdma);
    let x = x.to_dm(&mut ctx.tdma);
    let down = project_down_native_hbm_e165(ctx, &x, down_weight_packed, down_weight_scale);

    // down dot의 bf16 결과를 보존한다. 행렬 global scale은 post-FFN norm에서 적용한다.
    down.to_dm(&mut ctx.tdma)
}


// E183: 2 cluster × 256 slice × 30 rows. 각 slice가 H3840 전체를 처리한다.
// 출력 F32 부분합을 2개씩 기록하고, 산술 없이 별도 pass에서 기존 BF16 반올림을 수행한다.
pub(crate) type E183Rows = m![L / 30 % 256];

pub(crate) fn project_up_gate_full_hidden_e183(
    ctx: &mut Context,
    x_trf: &TrfTensor<bf16, Chip, UpGateClusters, E183Rows, m![1], m![H]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
) -> (
    DmTensor<f32, Chip, UpGateClusters, E183Rows, m![L % 30]>,
    DmTensor<f32, Chip, UpGateClusters, E183Rows, m![L % 30]>,
) {
    let mut up: DmTensor<f32, Chip, UpGateClusters, E183Rows, m![L % 30]> = DmTensor::new();
    let up_scale: DmTensor<f8e4m3, Chip, UpGateClusters, E183Rows, m![L % 30, H / 16]> = up_weight_scale.to_dm(&mut ctx.tdma);
    let mut up_packed0: DmTensor<f4e2m1, Chip, UpGateClusters, E183Rows, m![L % 30 = 18, H]> = DmTensor::new();
    let mut up_packed1: DmTensor<f4e2m1, Chip, UpGateClusters, E183Rows, m![L % 30 = 12, H]> = DmTensor::new();
    let mut up_decoded: DmTensor<bf16, Chip, UpGateClusters, E183Rows, m![L % 30, H]> = DmTensor::new();
    up_weight_packed.view().tile::<m![L % 30], 18, m![L / 30, L % 30 = 18 # 30, H]>(0).to_dm_view(&mut ctx.tdma, up_packed0.view_mut());
    up_weight_packed.view().tile::<m![L % 30], 12, m![L / 30, L % 30 = 12 # 30, H]>(18).to_dm_view(&mut ctx.tdma, up_packed1.view_mut());
    ctx.main.begin(up_packed0.view())
        .fetch::<m![L % 30 = 18], m![H]>().fetch_table_lookup::<f8e4m3>().fetch_cast::<f32>()
        .collect::<m![L % 30 = 18, H / 8], m![H % 8]>()
        .cast::<bf16, m![H % 8 # 16]>().commit_trim::<m![H % 8]>()
        .commit_view(up_decoded.view_mut().tile::<m![L % 30], 18, m![L % 30 = 18 #{!} 30, H]>(0));
    let scale_vrf: VrfTensor<f32, Chip, UpGateClusters, E183Rows, m![L % 30 = 6, H / 16]> = ctx.sub
        .begin(up_scale.view().tile::<m![L % 30], 6, m![L % 30 = 6 # 30, H / 16]>(0))
        .fetch::<m![L % 30 = 6], m![H / 16]>().fetch_cast::<f32>()
        .collect::<m![L % 30 = 6, H / 128], m![H / 16 % 8]>().to_vrf();
    ctx.main.begin(up_decoded.view().tile::<m![L % 30], 6, m![L % 30 = 6 # 30, H]>(0))
        .fetch::<m![L % 30 = 6, H / 16], m![H % 16]>()
        .collect::<m![L % 30 = 6, H / 16], m![H % 16]>()
        .contract_outer::<m![L % 30 = 6, H / 32], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![H / 16 % 2]>()
        .contract_time::<m![L % 30 = 6, H / 32]>()
        .contract_lane::<m![L % 30 = 6, H / 16], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_intra_slice_reduce::<H, m![L % 30 = 6], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .transpose::<m![L % 30 = 6 / 2], m![L % 30 = 6 % 2 # 8]>()
        .commit_trim::<m![L % 30 = 6 % 2]>()
        .commit_view(up.view_mut().tile::<m![L % 30], 6, m![L % 30 = 6 #{!} 30]>(0));
    let scale_vrf: VrfTensor<f32, Chip, UpGateClusters, E183Rows, m![L % 30 = 6, H / 16]> = ctx.sub
        .begin(up_scale.view().tile::<m![L % 30], 6, m![L % 30 = 6 # 30, H / 16]>(6))
        .fetch::<m![L % 30 = 6], m![H / 16]>().fetch_cast::<f32>()
        .collect::<m![L % 30 = 6, H / 128], m![H / 16 % 8]>().to_vrf();
    ctx.main.begin(up_decoded.view().tile::<m![L % 30], 6, m![L % 30 = 6 # 30, H]>(6))
        .fetch::<m![L % 30 = 6, H / 16], m![H % 16]>()
        .collect::<m![L % 30 = 6, H / 16], m![H % 16]>()
        .contract_outer::<m![L % 30 = 6, H / 32], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![H / 16 % 2]>()
        .contract_time::<m![L % 30 = 6, H / 32]>()
        .contract_lane::<m![L % 30 = 6, H / 16], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_intra_slice_reduce::<H, m![L % 30 = 6], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .transpose::<m![L % 30 = 6 / 2], m![L % 30 = 6 % 2 # 8]>()
        .commit_trim::<m![L % 30 = 6 % 2]>()
        .commit_view(up.view_mut().tile::<m![L % 30], 6, m![L % 30 = 6 #{!} 30]>(6));
    let scale_vrf: VrfTensor<f32, Chip, UpGateClusters, E183Rows, m![L % 30 = 6, H / 16]> = ctx.sub
        .begin(up_scale.view().tile::<m![L % 30], 6, m![L % 30 = 6 # 30, H / 16]>(12))
        .fetch::<m![L % 30 = 6], m![H / 16]>().fetch_cast::<f32>()
        .collect::<m![L % 30 = 6, H / 128], m![H / 16 % 8]>().to_vrf();
    ctx.main.begin(up_decoded.view().tile::<m![L % 30], 6, m![L % 30 = 6 # 30, H]>(12))
        .fetch::<m![L % 30 = 6, H / 16], m![H % 16]>()
        .collect::<m![L % 30 = 6, H / 16], m![H % 16]>()
        .contract_outer::<m![L % 30 = 6, H / 32], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![H / 16 % 2]>()
        .contract_time::<m![L % 30 = 6, H / 32]>()
        .contract_lane::<m![L % 30 = 6, H / 16], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_intra_slice_reduce::<H, m![L % 30 = 6], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .transpose::<m![L % 30 = 6 / 2], m![L % 30 = 6 % 2 # 8]>()
        .commit_trim::<m![L % 30 = 6 % 2]>()
        .commit_view(up.view_mut().tile::<m![L % 30], 6, m![L % 30 = 6 #{!} 30]>(12));
    ctx.main.begin(up_packed1.view())
        .fetch::<m![L % 30 = 12], m![H]>().fetch_table_lookup::<f8e4m3>().fetch_cast::<f32>()
        .collect::<m![L % 30 = 12, H / 8], m![H % 8]>()
        .cast::<bf16, m![H % 8 # 16]>().commit_trim::<m![H % 8]>()
        .commit_view(up_decoded.view_mut().tile::<m![L % 30], 12, m![L % 30 = 12 #{!} 30, H]>(18));
    let scale_vrf: VrfTensor<f32, Chip, UpGateClusters, E183Rows, m![L % 30 = 6, H / 16]> = ctx.sub
        .begin(up_scale.view().tile::<m![L % 30], 6, m![L % 30 = 6 # 30, H / 16]>(18))
        .fetch::<m![L % 30 = 6], m![H / 16]>().fetch_cast::<f32>()
        .collect::<m![L % 30 = 6, H / 128], m![H / 16 % 8]>().to_vrf();
    ctx.main.begin(up_decoded.view().tile::<m![L % 30], 6, m![L % 30 = 6 # 30, H]>(18))
        .fetch::<m![L % 30 = 6, H / 16], m![H % 16]>()
        .collect::<m![L % 30 = 6, H / 16], m![H % 16]>()
        .contract_outer::<m![L % 30 = 6, H / 32], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![H / 16 % 2]>()
        .contract_time::<m![L % 30 = 6, H / 32]>()
        .contract_lane::<m![L % 30 = 6, H / 16], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_intra_slice_reduce::<H, m![L % 30 = 6], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .transpose::<m![L % 30 = 6 / 2], m![L % 30 = 6 % 2 # 8]>()
        .commit_trim::<m![L % 30 = 6 % 2]>()
        .commit_view(up.view_mut().tile::<m![L % 30], 6, m![L % 30 = 6 #{!} 30]>(18));
    let scale_vrf: VrfTensor<f32, Chip, UpGateClusters, E183Rows, m![L % 30 = 6, H / 16]> = ctx.sub
        .begin(up_scale.view().tile::<m![L % 30], 6, m![L % 30 = 6 # 30, H / 16]>(24))
        .fetch::<m![L % 30 = 6], m![H / 16]>().fetch_cast::<f32>()
        .collect::<m![L % 30 = 6, H / 128], m![H / 16 % 8]>().to_vrf();
    ctx.main.begin(up_decoded.view().tile::<m![L % 30], 6, m![L % 30 = 6 # 30, H]>(24))
        .fetch::<m![L % 30 = 6, H / 16], m![H % 16]>()
        .collect::<m![L % 30 = 6, H / 16], m![H % 16]>()
        .contract_outer::<m![L % 30 = 6, H / 32], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![H / 16 % 2]>()
        .contract_time::<m![L % 30 = 6, H / 32]>()
        .contract_lane::<m![L % 30 = 6, H / 16], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_intra_slice_reduce::<H, m![L % 30 = 6], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .transpose::<m![L % 30 = 6 / 2], m![L % 30 = 6 % 2 # 8]>()
        .commit_trim::<m![L % 30 = 6 % 2]>()
        .commit_view(up.view_mut().tile::<m![L % 30], 6, m![L % 30 = 6 #{!} 30]>(24));
    let mut gate: DmTensor<f32, Chip, UpGateClusters, E183Rows, m![L % 30]> = DmTensor::new();
    let gate_scale: DmTensor<f8e4m3, Chip, UpGateClusters, E183Rows, m![L % 30, H / 16]> = gate_weight_scale.to_dm(&mut ctx.tdma);
    let mut gate_packed0: DmTensor<f4e2m1, Chip, UpGateClusters, E183Rows, m![L % 30 = 18, H]> = DmTensor::new();
    let mut gate_packed1: DmTensor<f4e2m1, Chip, UpGateClusters, E183Rows, m![L % 30 = 12, H]> = DmTensor::new();
    let mut gate_decoded: DmTensor<bf16, Chip, UpGateClusters, E183Rows, m![L % 30, H]> = DmTensor::new();
    gate_weight_packed.view().tile::<m![L % 30], 18, m![L / 30, L % 30 = 18 # 30, H]>(0).to_dm_view(&mut ctx.tdma, gate_packed0.view_mut());
    gate_weight_packed.view().tile::<m![L % 30], 12, m![L / 30, L % 30 = 12 # 30, H]>(18).to_dm_view(&mut ctx.tdma, gate_packed1.view_mut());
    ctx.main.begin(gate_packed0.view())
        .fetch::<m![L % 30 = 18], m![H]>().fetch_table_lookup::<f8e4m3>().fetch_cast::<f32>()
        .collect::<m![L % 30 = 18, H / 8], m![H % 8]>()
        .cast::<bf16, m![H % 8 # 16]>().commit_trim::<m![H % 8]>()
        .commit_view(gate_decoded.view_mut().tile::<m![L % 30], 18, m![L % 30 = 18 #{!} 30, H]>(0));
    let scale_vrf: VrfTensor<f32, Chip, UpGateClusters, E183Rows, m![L % 30 = 6, H / 16]> = ctx.sub
        .begin(gate_scale.view().tile::<m![L % 30], 6, m![L % 30 = 6 # 30, H / 16]>(0))
        .fetch::<m![L % 30 = 6], m![H / 16]>().fetch_cast::<f32>()
        .collect::<m![L % 30 = 6, H / 128], m![H / 16 % 8]>().to_vrf();
    ctx.main.begin(gate_decoded.view().tile::<m![L % 30], 6, m![L % 30 = 6 # 30, H]>(0))
        .fetch::<m![L % 30 = 6, H / 16], m![H % 16]>()
        .collect::<m![L % 30 = 6, H / 16], m![H % 16]>()
        .contract_outer::<m![L % 30 = 6, H / 32], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![H / 16 % 2]>()
        .contract_time::<m![L % 30 = 6, H / 32]>()
        .contract_lane::<m![L % 30 = 6, H / 16], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_intra_slice_reduce::<H, m![L % 30 = 6], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .transpose::<m![L % 30 = 6 / 2], m![L % 30 = 6 % 2 # 8]>()
        .commit_trim::<m![L % 30 = 6 % 2]>()
        .commit_view(gate.view_mut().tile::<m![L % 30], 6, m![L % 30 = 6 #{!} 30]>(0));
    let scale_vrf: VrfTensor<f32, Chip, UpGateClusters, E183Rows, m![L % 30 = 6, H / 16]> = ctx.sub
        .begin(gate_scale.view().tile::<m![L % 30], 6, m![L % 30 = 6 # 30, H / 16]>(6))
        .fetch::<m![L % 30 = 6], m![H / 16]>().fetch_cast::<f32>()
        .collect::<m![L % 30 = 6, H / 128], m![H / 16 % 8]>().to_vrf();
    ctx.main.begin(gate_decoded.view().tile::<m![L % 30], 6, m![L % 30 = 6 # 30, H]>(6))
        .fetch::<m![L % 30 = 6, H / 16], m![H % 16]>()
        .collect::<m![L % 30 = 6, H / 16], m![H % 16]>()
        .contract_outer::<m![L % 30 = 6, H / 32], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![H / 16 % 2]>()
        .contract_time::<m![L % 30 = 6, H / 32]>()
        .contract_lane::<m![L % 30 = 6, H / 16], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_intra_slice_reduce::<H, m![L % 30 = 6], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .transpose::<m![L % 30 = 6 / 2], m![L % 30 = 6 % 2 # 8]>()
        .commit_trim::<m![L % 30 = 6 % 2]>()
        .commit_view(gate.view_mut().tile::<m![L % 30], 6, m![L % 30 = 6 #{!} 30]>(6));
    let scale_vrf: VrfTensor<f32, Chip, UpGateClusters, E183Rows, m![L % 30 = 6, H / 16]> = ctx.sub
        .begin(gate_scale.view().tile::<m![L % 30], 6, m![L % 30 = 6 # 30, H / 16]>(12))
        .fetch::<m![L % 30 = 6], m![H / 16]>().fetch_cast::<f32>()
        .collect::<m![L % 30 = 6, H / 128], m![H / 16 % 8]>().to_vrf();
    ctx.main.begin(gate_decoded.view().tile::<m![L % 30], 6, m![L % 30 = 6 # 30, H]>(12))
        .fetch::<m![L % 30 = 6, H / 16], m![H % 16]>()
        .collect::<m![L % 30 = 6, H / 16], m![H % 16]>()
        .contract_outer::<m![L % 30 = 6, H / 32], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![H / 16 % 2]>()
        .contract_time::<m![L % 30 = 6, H / 32]>()
        .contract_lane::<m![L % 30 = 6, H / 16], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_intra_slice_reduce::<H, m![L % 30 = 6], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .transpose::<m![L % 30 = 6 / 2], m![L % 30 = 6 % 2 # 8]>()
        .commit_trim::<m![L % 30 = 6 % 2]>()
        .commit_view(gate.view_mut().tile::<m![L % 30], 6, m![L % 30 = 6 #{!} 30]>(12));
    ctx.main.begin(gate_packed1.view())
        .fetch::<m![L % 30 = 12], m![H]>().fetch_table_lookup::<f8e4m3>().fetch_cast::<f32>()
        .collect::<m![L % 30 = 12, H / 8], m![H % 8]>()
        .cast::<bf16, m![H % 8 # 16]>().commit_trim::<m![H % 8]>()
        .commit_view(gate_decoded.view_mut().tile::<m![L % 30], 12, m![L % 30 = 12 #{!} 30, H]>(18));
    let scale_vrf: VrfTensor<f32, Chip, UpGateClusters, E183Rows, m![L % 30 = 6, H / 16]> = ctx.sub
        .begin(gate_scale.view().tile::<m![L % 30], 6, m![L % 30 = 6 # 30, H / 16]>(18))
        .fetch::<m![L % 30 = 6], m![H / 16]>().fetch_cast::<f32>()
        .collect::<m![L % 30 = 6, H / 128], m![H / 16 % 8]>().to_vrf();
    ctx.main.begin(gate_decoded.view().tile::<m![L % 30], 6, m![L % 30 = 6 # 30, H]>(18))
        .fetch::<m![L % 30 = 6, H / 16], m![H % 16]>()
        .collect::<m![L % 30 = 6, H / 16], m![H % 16]>()
        .contract_outer::<m![L % 30 = 6, H / 32], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![H / 16 % 2]>()
        .contract_time::<m![L % 30 = 6, H / 32]>()
        .contract_lane::<m![L % 30 = 6, H / 16], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_intra_slice_reduce::<H, m![L % 30 = 6], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .transpose::<m![L % 30 = 6 / 2], m![L % 30 = 6 % 2 # 8]>()
        .commit_trim::<m![L % 30 = 6 % 2]>()
        .commit_view(gate.view_mut().tile::<m![L % 30], 6, m![L % 30 = 6 #{!} 30]>(18));
    let scale_vrf: VrfTensor<f32, Chip, UpGateClusters, E183Rows, m![L % 30 = 6, H / 16]> = ctx.sub
        .begin(gate_scale.view().tile::<m![L % 30], 6, m![L % 30 = 6 # 30, H / 16]>(24))
        .fetch::<m![L % 30 = 6], m![H / 16]>().fetch_cast::<f32>()
        .collect::<m![L % 30 = 6, H / 128], m![H / 16 % 8]>().to_vrf();
    ctx.main.begin(gate_decoded.view().tile::<m![L % 30], 6, m![L % 30 = 6 # 30, H]>(24))
        .fetch::<m![L % 30 = 6, H / 16], m![H % 16]>()
        .collect::<m![L % 30 = 6, H / 16], m![H % 16]>()
        .contract_outer::<m![L % 30 = 6, H / 32], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![H / 16 % 2]>()
        .contract_time::<m![L % 30 = 6, H / 32]>()
        .contract_lane::<m![L % 30 = 6, H / 16], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init().vector_intra_slice_tag(TagMode::Zero).vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_intra_slice_reduce::<H, m![L % 30 = 6], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .transpose::<m![L % 30 = 6 / 2], m![L % 30 = 6 % 2 # 8]>()
        .commit_trim::<m![L % 30 = 6 % 2]>()
        .commit_view(gate.view_mut().tile::<m![L % 30], 6, m![L % 30 = 6 #{!} 30]>(24));
    (up, gate)
}

pub(crate) fn feedforward_full_hidden_e183(
    ctx: &mut Context,
    x: DmTensor<bf16, Chip, m![Dummy2], crate::device::layout::Replicated, m![H]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, m![1 # 16, H / 240], m![H % 240]> {
    // 입력 H3840은 이미 두 cluster의 모든 slice에 실제 복제되어 있다.
    // 그 복제 위치를 256개 출력 행 블록의 owner로 이름 붙인다.
    let x: DmTensor<bf16, Chip, UpGateClusters, E183Rows, m![H]> = unsafe { x.reshape() };
    let x_trf: TrfTensor<bf16, Chip, UpGateClusters, E183Rows, m![1], m![H]> = ctx.sub
        .begin(x.view()).fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>().to_trf();
    let (up, gate) = project_up_gate_full_hidden_e183(ctx, &x_trf,
        up_weight_packed, gate_weight_packed, up_weight_scale, gate_weight_scale);
    // 패딩 없는 F32 30행을 인접 slice와 모은 뒤 기존 BF16 반올림을 수행한다.
    let up: DmTensor<f32, Chip, UpGateClusters, UpGateRows, m![L % 60]> = up.to_dm(&mut ctx.tdma);
    let up: DmTensor<bf16, Chip, UpGateClusters, UpGateRows, m![L % 60]> = ctx.main
        .begin(up.view()).fetch::<m![L % 60 / 4], m![L % 4]>()
        .collect::<m![L % 60 / 4], m![L % 4 # 8]>()
        .cast::<bf16, m![L % 4 # 16]>().commit_trim::<m![L % 4]>().commit();
    let gate: DmTensor<f32, Chip, UpGateClusters, UpGateRows, m![L % 60]> = gate.to_dm(&mut ctx.tdma);
    let gate: DmTensor<bf16, Chip, UpGateClusters, UpGateRows, m![L % 60]> = ctx.main
        .begin(gate.view()).fetch::<m![L % 60 / 4], m![L % 4]>()
        .collect::<m![L % 60 / 4], m![L % 4 # 8]>()
        .cast::<bf16, m![L % 4 # 16]>().commit_trim::<m![L % 4]>().commit();

    let x = geglu(ctx, up, gate, up_global_scale, gate_global_scale);
    // GeGLU의 64개 sparse slice×120 elements를 cluster별 연속 7680 elements로 모은다.
    let x: DmTensor<bf16, Chip, UpGateClusters, Slice, m![L % 7680]> = x.to_dm(&mut ctx.tdma);
    let x: HbmTensor<bf16, Chip, m![L]> = x.to_hbm(&mut ctx.tdma);
    let x = x.to_dm(&mut ctx.tdma);
    let down = project_down_native_hbm_e165(ctx, &x, down_weight_packed, down_weight_scale);

    // down dot의 bf16 결과를 보존한다. 행렬 global scale은 post-FFN norm에서 적용한다.
    down.to_dm(&mut ctx.tdma)
}
