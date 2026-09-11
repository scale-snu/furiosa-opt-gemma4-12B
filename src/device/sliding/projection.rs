
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Ds, Dummy2, Gs, H, Ns, Ps, Qs};
use crate::device::layout::{Cluster, Replicated, Slice};

pub(crate) fn project_query(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, m![Dummy2], Replicated, m![H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
) -> DmTensor<bf16, Chip, m![Ns / 4], Slice, m![Ns % 4, Gs, Ds]> {
    type QueryClusters = m![Qs / 2048];
    type QueryRows = m![Qs / 8 % 256];

    // 두 cluster, 모든 slice에 같은 H가 이미 복제돼 있어 출력 행 매핑으로 선택할 수 있다.
    let x: DmTensorView<'_, bf16, Chip, QueryClusters, QueryRows, m![H]> = unsafe { x.view().reshape() };
    let x_trf: TrfTensor<bf16, Chip, QueryClusters, QueryRows, m![1], m![H]> = ctx
        .sub
        .begin(x)
        .fetch::<m![1], m![H]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();

    let weight_f8: DmTensor<f8e4m3, Chip, QueryClusters, QueryRows, m![Qs % 8, H]> = weight.to_dm(&mut ctx.tdma);
    let contraction: DmTensor<bf16, Chip, QueryClusters, QueryRows, m![Qs % 8]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Qs % 8, H / 16], m![H % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Qs % 8, H / 16], m![H % 16]>()
        .contract_outer::<m![Qs % 8, H / 32], m![H % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs % 8]>()
        .contract_lane::<m![Qs % 8], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 4 % 2], m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>()
        .commit();

    let weight_scale: DmTensor<bf16, Chip, QueryClusters, QueryRows, m![Qs % 8]> = weight_scale.to_dm(&mut ctx.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, QueryClusters, QueryRows, m![Qs % 8]> = ctx
        .sub
        .begin(weight_scale.view())
        .fetch::<m![1], m![Qs % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Qs % 8]>()
        .to_vrf();

    let scaled: DmTensor<bf16, Chip, QueryClusters, QueryRows, m![Qs % 8]> = ctx
        .main
        .begin(contraction.view())
        .fetch::<m![1], m![Qs % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 2], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_widen_concat::<m![1], m![Qs % 8]>()
        .vector_final()
        .cast::<bf16, m![Qs % 8 # 16]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();

    // 짧은 slice별 출력을 먼저 모아 HBM에 8-element 조각을 반복해서 쓰지 않는다.
    let gathered: DmTensor<bf16, Chip, QueryClusters, Slice, m![Qs % 2048]> = ctx
        .main
        .begin(scaled.view())
        .fetch::<m![1], m![Qs % 8 # 16]>()
        .switch::<Slice, m![Qs / 8 % 256]>(SwitchConfig::Broadcast1 { slice1: 256, slice0: 1 })
        .collect::<m![Qs / 8 % 256], m![Qs % 8 # 16]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();
    // Qs의 cluster당 2,048개 값은 Ns의 4개 head × Gs × Ds와 같은 물리 순서다.
    // 두 cluster의 분할을 유지하여 후처리까지 HBM 왕복 없이 진행한다.
    unsafe { gathered.reshape() }
}

type KvClusters = m![Ps / 1024];
type KvRows = m![Ps / 4 % 256];

fn project_one_kv_matrix(
    ctx: &mut Context,
    x_trf: &TrfTensor<bf16, Chip, KvClusters, KvRows, m![1], m![H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> DmTensor<bf16, Chip, KvClusters, Slice, m![Ps % 1024]> {
    let weight_f8: DmTensor<f8e4m3, Chip, KvClusters, KvRows, m![Ps % 4, H]> = weight.to_dm(&mut ctx.tdma);
    let contraction: DmTensor<bf16, Chip, KvClusters, KvRows, m![Ps % 4]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Ps % 4, H / 16], m![H % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Ps % 4, H / 16], m![H % 16]>()
        .contract_outer::<m![Ps % 4, H / 32], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ps % 4]>()
        .contract_lane::<m![Ps % 4], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![1], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();

    let weight_scale: DmTensor<bf16, Chip, KvClusters, KvRows, m![Ps % 4]> = weight_scale.to_dm(&mut ctx.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, KvClusters, KvRows, m![Ps % 4 # 8]> = ctx
        .sub
        .begin(weight_scale.view())
        .fetch::<m![1], m![Ps % 4 # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Ps % 4 # 8]>()
        .to_vrf();

    let scaled: DmTensor<bf16, Chip, KvClusters, KvRows, m![Ps % 4]> = ctx
        .main
        .begin(contraction.view())
        .fetch::<m![1], m![Ps % 4 # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Ps % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![Ps % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_widen_pad::<m![Ps % 4 # 8]>()
        .vector_final()
        .cast::<bf16, m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();

    let gathered: DmTensor<bf16, Chip, KvClusters, Slice, m![Ps % 1024]> = ctx
        .main
        .begin(scaled.view())
        .fetch::<m![1], m![Ps % 4 # 16]>()
        .switch::<Slice, m![Ps / 4 % 256]>(SwitchConfig::Broadcast1 { slice1: 256, slice0: 1 })
        .collect::<m![Ps / 4 % 256], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();
    gathered
}

pub(crate) fn project_key_value(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, m![Dummy2], Replicated, m![H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, m![Ns / 4], Slice, m![Ns % 4, Ds]>,
    DmTensor<bf16, Chip, m![Ns / 4], Slice, m![Ns % 4, Ds]>,
) {
    // H가 양쪽 cluster와 모든 slice에 복제돼 있으므로 각 K/V 행이 같은 입력을 선택한다.
    let x: DmTensorView<'_, bf16, Chip, KvClusters, KvRows, m![H]> = unsafe { x.view().reshape() };
    let x_trf: TrfTensor<bf16, Chip, KvClusters, KvRows, m![1], m![H]> = ctx
        .sub
        .begin(x)
        .fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();

    let k: DmTensor<bf16, Chip, KvClusters, Slice, m![Ps % 1024]> = project_one_kv_matrix(ctx, &x_trf, k_weight, k_weight_scale);
    let v: DmTensor<bf16, Chip, KvClusters, Slice, m![Ps % 1024]> = project_one_kv_matrix(ctx, &x_trf, v_weight, v_weight_scale);

    // Ps의 cluster당 1,024개 값은 Ns의 4개 head × Ds와 동일한 연속 배치다.
    (unsafe { k.reshape() }, unsafe { v.reshape() })
}

// 두 cluster에서 행 16개 그룹 × 열 16개 그룹으로 512개 slice를 사용한다.
type OutputClusters = m![H / 1920];
type OutputRowsByColumns = m![H / 120 % 16, Qs / 256];
type HiddenRows = m![H / 120 % 16, 1 # 16];

pub(crate) fn project_output(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, OutputClusters, OutputRowsByColumns, m![Qs % 256]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    let x_trf: TrfTensor<bf16, Chip, OutputClusters, OutputRowsByColumns, m![1], m![Qs % 256]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>()
        .collect::<m![Qs / 16 % 16], m![Qs % 16]>()
        .to_trf();
    let weight: DmTensor<f8e4m3, Chip, OutputClusters, OutputRowsByColumns, m![H % 120, Qs % 256]> =
        weight.to_dm(&mut ctx.tdma);
    // 각 slice의 256-column 부분합을 f32로 구하고 같은 행의 16개 slice를 합친다.
    // FP8 복원은 정확한 bf16 값이며 행렬 전체의 합산 뒤 bf16로 반올림한다.
    let result: DmTensor<bf16, Chip, OutputClusters, HiddenRows, m![H % 120]> = ctx
        .main
        .begin(weight.view())
        .fetch::<m![H % 120, Qs / 32 % 8], m![Qs % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![H % 120, Qs / 16 % 16], m![Qs % 16]>()
        .contract_outer::<m![H % 120, Qs / 32 % 8], m![Qs % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120]>()
        .contract_lane::<m![H % 120], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<HiddenRows, m![H % 120]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H / 4 % 30], m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>()
        .commit();
    let result = apply_output_channel_scale(ctx, &result, weight_scale);
    // SDK 0.6의 cluster 간 DM 복사 동기화 제약 때문에 작은 최종 H 벡터만 HBM을 경유한다.
    let result: HbmTensor<bf16, Chip, m![H]> = result.to_hbm(&mut ctx.tdma);
    result.to_dm(&mut ctx.tdma)
}

fn apply_output_channel_scale(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, OutputClusters, HiddenRows, m![H % 120]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, OutputClusters, HiddenRows, m![H % 120]> {
    let weight_scale: DmTensor<bf16, Chip, OutputClusters, HiddenRows, m![H % 120]> = weight_scale.to_dm(&mut ctx.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, OutputClusters, HiddenRows, m![H % 120]> = ctx
        .sub
        .begin(weight_scale.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .to_vrf();

    ctx.main
        .begin(x.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_widen_concat::<m![H / 8 % 15], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}
