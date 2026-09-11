
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
pub(crate) type ValueHeadSlices = m![Ns % 4, 1 # 64];

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

fn project_one_value_matrix(
    ctx: &mut Context,
    x_trf: &TrfTensor<bf16, Chip, KvClusters, KvRows, m![1], m![H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> DmTensor<bf16, Chip, m![Ns / 4], ValueHeadSlices, m![Ds]> {
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

    // 각 row slice의 네 값은 한 head의 연속 Ds%4이다. 물리 순서만 재명명한다.
    let scaled: DmTensor<bf16, Chip, m![Ns / 4], m![Ns % 4, Ds / 4], m![Ds % 4]> = unsafe { scaled.reshape() };
    // 64개 row slice씩 모아 각 head를 독립 slice에 둔다.
    ctx.main
        .begin(scaled.view())
        .fetch::<m![1], m![Ds % 4 # 16]>()
        .switch::<ValueHeadSlices, m![Ds / 4]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
        .collect::<m![Ds / 4], m![Ds % 4 # 16]>()
        .commit_trim::<m![Ds % 4]>()
        .commit()
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
    DmTensor<bf16, Chip, m![Ns / 4], ValueHeadSlices, m![Ds]>,
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
    let v: DmTensor<bf16, Chip, m![Ns / 4], ValueHeadSlices, m![Ds]> = project_one_value_matrix(ctx, &x_trf, v_weight, v_weight_scale);

    // Ps의 cluster당 1,024개 값은 Ns의 4개 head × Ds와 동일한 연속 배치다.
    (unsafe { k.reshape() }, v)
}

// 두 cluster에서 행 16개 그룹 × 열 16개 그룹으로 512개 slice를 사용한다.
type OutputClusters = m![H / 1920];
type OutputRowsByColumns = m![H / 120 % 16, Qs / 256];
type HiddenRows = m![H / 120 % 16, 1 # 16];
axes![OutputRowGroups = 30, OutputRowBlock = 4];

pub(crate) fn project_output(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, OutputClusters, OutputRowsByColumns, m![Qs % 256]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
) -> HbmTensor<bf16, Chip, m![H]> {
    let x_trf: TrfTensor<bf16, Chip, OutputClusters, OutputRowsByColumns, m![1], m![Qs % 256]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>()
        .collect::<m![Qs / 16 % 16], m![Qs % 16]>()
        .to_trf();
    // 30개의 4행 group을 비대칭 타일로 나누되 출력 물리 순서는 120행 연속이다.
    let mut result: DmTensor<bf16, Chip, OutputClusters, HiddenRows, m![OutputRowGroups, OutputRowBlock]> = DmTensor::new();
    let weight0: DmTensor<f8e4m3, Chip, OutputClusters, OutputRowsByColumns, m![H % 120 = 92, Qs % 256]> =
        weight.view().tile::<m![H % 120], 92, m![H / 120, H % 120 = 92 # 120, Qs]>(0).to_dm(&mut ctx.tdma);
    let weight1: DmTensor<f8e4m3, Chip, OutputClusters, OutputRowsByColumns, m![H % 120 = 28, Qs % 256]> =
        weight.view().tile::<m![H % 120], 28, m![H / 120, H % 120 = 28 # 120, Qs]>(92).to_dm(&mut ctx.tdma);
    {
        // 각 slice의 연속 92×256 FP8 값에서 row=4*group+block이며 주소/공간축은 그대로다.
        let weight: DmTensorView<'_, f8e4m3, Chip, OutputClusters, OutputRowsByColumns, m![OutputRowGroups = 23, OutputRowBlock, Qs % 256]> =
            unsafe { weight0.view().reshape() };
        ctx.main
            .begin(weight)
            .fetch::<m![OutputRowGroups = 23, OutputRowBlock, Qs / 32 % 8], m![Qs % 32]>()
            .fetch_table_lookup::<bf16>()
            .collect::<m![OutputRowGroups = 23, OutputRowBlock, Qs / 16 % 16], m![Qs % 16]>()
            .contract_outer::<m![OutputRowGroups = 23, OutputRowBlock, Qs / 32 % 8], m![Qs % 32], _, _, _>(&x_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![OutputRowGroups = 23, OutputRowBlock]>()
            .contract_lane::<m![OutputRowGroups = 23, OutputRowBlock], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<HiddenRows, m![OutputRowGroups = 23, OutputRowBlock]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![OutputRowGroups = 23], m![OutputRowBlock # 16]>()
            .commit_trim::<m![OutputRowBlock]>()
            .commit_view(result.view_mut().tile::<m![OutputRowGroups], 23, m![OutputRowGroups = 23 #{!} 30, OutputRowBlock]>(0));
    }
    {
        // 각 slice의 연속 28×256 FP8 값에서 row=4*group+block이며 주소/공간축은 그대로다.
        let weight: DmTensorView<'_, f8e4m3, Chip, OutputClusters, OutputRowsByColumns, m![OutputRowGroups = 7, OutputRowBlock, Qs % 256]> =
            unsafe { weight1.view().reshape() };
        ctx.main
            .begin(weight)
            .fetch::<m![OutputRowGroups = 7, OutputRowBlock, Qs / 32 % 8], m![Qs % 32]>()
            .fetch_table_lookup::<bf16>()
            .collect::<m![OutputRowGroups = 7, OutputRowBlock, Qs / 16 % 16], m![Qs % 16]>()
            .contract_outer::<m![OutputRowGroups = 7, OutputRowBlock, Qs / 32 % 8], m![Qs % 32], _, _, _>(&x_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![OutputRowGroups = 7, OutputRowBlock]>()
            .contract_lane::<m![OutputRowGroups = 7, OutputRowBlock], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<HiddenRows, m![OutputRowGroups = 7, OutputRowBlock]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![OutputRowGroups = 7], m![OutputRowBlock # 16]>()
            .commit_trim::<m![OutputRowBlock]>()
            .commit_view(result.view_mut().tile::<m![OutputRowGroups], 7, m![OutputRowGroups = 7 #{!} 30, OutputRowBlock]>(23));
    }
    // group-major 30×4의 120개 bf16은 H%120과 같은 주소/순서다. padding이나 broadcast는 없다.
    let result: DmTensor<bf16, Chip, OutputClusters, HiddenRows, m![H % 120]> = unsafe { result.reshape() };
    let result = apply_output_channel_scale(ctx, &result, weight_scale);
    let result: DmTensor<bf16, Chip, OutputClusters, Slice, m![H % 1920]> = result.to_dm(&mut ctx.tdma);
    result.to_hbm(&mut ctx.tdma)
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

// 단일120행 후보. 다른 호출 경로의 기존 helper는 보존한다.
pub(crate) fn project_output_single120_e91_norm(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, OutputClusters, OutputRowsByColumns, m![Qs % 256]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> HbmTensor<bf16, Chip, m![H]> {
    // 각 slice의 실제 BF16 입력 256개에서 scale을 구한다. 영벡터도 같은 경로로 처리한다.
    let block_scale: DmTensor<f32, Chip, OutputClusters, OutputRowsByColumns, m![1 # 8]> = ctx.main
        .begin(x.view())
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffff)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_intra_slice_reduce::<Qs, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_fp_div(256.0)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1e-30 / 256.0)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    let scale_vrf: VrfTensor<f32, Chip, OutputClusters, OutputRowsByColumns, m![1 # 8]> = ctx.sub
        .begin(block_scale.view()).fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>().to_vrf();
    let hi: DmTensor<f8e4m3, Chip, OutputClusters, OutputRowsByColumns, m![Qs % 256]> = ctx.main
        .begin(x.view()).fetch::<m![Qs / 16 % 16], m![Qs % 16]>().fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>().vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>().commit_trim::<m![Qs % 8]>().commit();
    let hi_vrf: VrfTensor<f32, Chip, OutputClusters, OutputRowsByColumns, m![Qs % 256]> = ctx.sub
        .begin(hi.view()).fetch::<m![Qs / 32 % 8], m![Qs % 32]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>().to_vrf();
    let lo: DmTensor<f8e4m3, Chip, OutputClusters, OutputRowsByColumns, m![Qs % 256]> = ctx.main
        .begin(x.view()).fetch::<m![Qs / 16 % 16], m![Qs % 16]>().fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 16.0)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>().vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>().commit_trim::<m![Qs % 8]>().commit();
    let hi_trf: TrfTensor<f8e4m3, Chip, OutputClusters, OutputRowsByColumns, m![1], m![Qs % 256]> = ctx.sub
        .begin(hi.view()).fetch::<m![Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Qs / 32 % 8], m![Qs % 32]>().to_trf();
    let lo_trf: TrfTensor<f8e4m3, Chip, OutputClusters, OutputRowsByColumns, m![1], m![Qs % 256]> = ctx.sub
        .begin(lo.view()).fetch::<m![Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Qs / 32 % 8], m![Qs % 32]>().to_trf();
    // 120행 전체를 한 번 읽어 두 번째 weight DMA/LUT/타일 경계를 제거한다.
    let weight_all: DmTensor<f8e4m3, Chip, OutputClusters, OutputRowsByColumns, m![H % 120, Qs % 256]> =
        weight.to_dm(&mut ctx.tdma);
    let result = {
        // 각 slice의 연속 120×256 FP8 값에서 row=4*group+block이며 주소/공간축은 그대로다.
        let weight: DmTensorView<'_, f8e4m3, Chip, OutputClusters, OutputRowsByColumns, m![OutputRowGroups, OutputRowBlock, Qs % 256]> =
            unsafe { weight_all.view().reshape() };
        let dot_hi: DmTensor<f32, Chip, OutputClusters, OutputRowsByColumns, m![OutputRowGroups, OutputRowBlock]> = ctx.main
            .begin(weight)
            .fetch::<m![OutputRowGroups, OutputRowBlock, Qs / 32 % 8], m![Qs % 32]>()
            .collect::<m![OutputRowGroups, OutputRowBlock, Qs / 32 % 8], m![Qs % 32]>()
            .contract_outer::<m![OutputRowGroups, OutputRowBlock, Qs / 32 % 8], m![Qs % 32], _, _, _>(&hi_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![OutputRowGroups, OutputRowBlock]>()
            .contract_lane::<m![OutputRowGroups, OutputRowBlock], m![1 # 8]>(LaneMode::Interleaved)
            .transpose::<m![OutputRowGroups, OutputRowBlock / 2], m![OutputRowBlock % 2 # 8]>()
            .commit_trim::<m![OutputRowBlock % 2]>().commit();
        let dot_hi_vrf: VrfTensor<f32, Chip, OutputClusters, OutputRowsByColumns, m![OutputRowGroups, OutputRowBlock # 8]> = ctx.sub
            .begin(dot_hi.view()).fetch::<m![OutputRowGroups], m![OutputRowBlock # 8]>()
            .collect::<m![OutputRowGroups], m![OutputRowBlock # 8]>().to_vrf();
        let weight: DmTensorView<'_, f8e4m3, Chip, OutputClusters, OutputRowsByColumns, m![OutputRowGroups, OutputRowBlock, Qs % 256]> =
            unsafe { weight_all.view().reshape() };
        let result: DmTensor<bf16, Chip, OutputClusters, HiddenRows, m![OutputRowGroups, OutputRowBlock]> = ctx.main
            .begin(weight)
            .fetch::<m![OutputRowGroups, OutputRowBlock, Qs / 32 % 8], m![Qs % 32]>()
            .collect::<m![OutputRowGroups, OutputRowBlock, Qs / 32 % 8], m![Qs % 32]>()
            .contract_outer::<m![OutputRowGroups, OutputRowBlock, Qs / 32 % 8], m![Qs % 32], _, _, _>(&lo_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![OutputRowGroups, OutputRowBlock]>()
            .contract_lane::<m![OutputRowGroups, OutputRowBlock], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 1.0 / 16.0)
            .vector_fp_binary(FpBinaryOp::AddF, &dot_hi_vrf)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scale_vrf)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<HiddenRows, m![OutputRowGroups, OutputRowBlock]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![OutputRowGroups], m![OutputRowBlock # 16]>()
            .commit_trim::<m![OutputRowBlock]>()
            .commit();
        result
    };
    // group-major 30×4의 120개 bf16은 H%120과 같은 주소/순서다. padding이나 broadcast는 없다.
    let result: DmTensor<bf16, Chip, OutputClusters, HiddenRows, m![H % 120]> = unsafe { result.reshape() };
    let result: DmTensor<bf16, Chip, OutputClusters, Slice, m![H % 1920]> = result.to_dm(&mut ctx.tdma);
    result.to_hbm(&mut ctx.tdma)
}


// E113: 같은 normalized H를 Q/K/V가 공유한다. Term=0은 hi, Term=1은 lo다.
axes![QkvFp8Term = 2, QkvFp8Part = 4];


pub(crate) fn prepare_qkv_packed_input(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, m![Dummy2], Replicated, m![H]>,
) -> (
    DmTensor<f8e4m3, Chip, m![Dummy2], Replicated, m![QkvFp8Term, H]>,
    DmTensor<f32, Chip, m![Dummy2], Replicated, m![1 # 8]>,
) {
    // 각 실제 replica가 같은 H3840의 max를 구한다. 분할마다 scale을 바꾸지 않는다.
    let scale: DmTensor<f32, Chip, m![Dummy2], Replicated, m![1 # 8]> = ctx.main
        .begin(x.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffff)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_fp_div(256.0)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1e-30 / 256.0)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    let scale_vrf: VrfTensor<f32, Chip, m![Dummy2], Replicated, m![1 # 8]> = ctx.sub
        .begin(scale.view()).fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>().to_vrf();
    let mut parts: DmTensor<f8e4m3, Chip, m![Dummy2], Replicated, m![QkvFp8Part, H % 1920]> = DmTensor::new();
    for block in 0..2 {
        // H/1920의 논리 좌표0/1은 실제 H byte 구간을 고른다. 입력 fixture와 무관하다.
        ctx.main.begin(x.view().tile::<m![H / 1920], 1, m![H / 1920 = 1 # 2, H % 1920]>(block))
            .fetch::<m![H / 16 % 120], m![H % 16]>().fetch_cast::<f32>()
            .collect::<m![H / 8 % 240], m![H % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H / 4 % 480], m![H % 4]>()
            .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
            .vector_widen_concat::<m![H / 8 % 240], m![H % 8]>().vector_final()
            .cast::<f8e4m3, m![H % 8 # 32]>().commit_trim::<m![H % 8]>()
            .commit_view(parts.view_mut().tile::<m![QkvFp8Part], 1, m![QkvFp8Part = 1 #{!} 4, H % 1920]>(block + 0));
        // 실제 term0의 해당1920개만 읽는다. hi f32=7680B + scale32B로8192B보다작다.
        let hi_vrf: VrfTensor<f32, Chip, m![Dummy2], Replicated, m![H % 1920]> = ctx.sub
            .begin(parts.view().tile::<m![QkvFp8Part], 1, m![QkvFp8Part = 1 # 4, H % 1920]>(block))
            .fetch::<m![H / 32 % 60], m![H % 32]>().fetch_cast::<f32>()
            .collect::<m![H / 8 % 240], m![H % 8]>().to_vrf();
        ctx.main.begin(x.view().tile::<m![H / 1920], 1, m![H / 1920 = 1 # 2, H % 1920]>(block))
            .fetch::<m![H / 16 % 120], m![H % 16]>().fetch_cast::<f32>()
            .collect::<m![H / 8 % 240], m![H % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H / 4 % 480], m![H % 4]>()
            .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
            .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
            .vector_widen_concat::<m![H / 8 % 240], m![H % 8]>().vector_final()
            .cast::<f8e4m3, m![H % 8 # 32]>().commit_trim::<m![H % 8]>()
            .commit_view(parts.view_mut().tile::<m![QkvFp8Part], 1, m![QkvFp8Part = 1 #{!} 4, H % 1920]>(block + 2));
    }
    // 4개의 실제1920B 구간은 hi0,hi1,lo0,lo1이다. 연속4×1920을2×3840으로 묶는다.
    // 위치별 값/byte순서는 동일하며 새복제나padding가정은 없다.
    let terms: DmTensor<f8e4m3, Chip, m![Dummy2], Replicated, m![QkvFp8Term, H]> = unsafe { parts.reshape() };
    (terms, scale)
}

pub(crate) fn project_query_packed_e113(
    ctx: &mut Context,
    terms: &DmTensor<f8e4m3, Chip, m![Dummy2], Replicated, m![QkvFp8Term, H]>,
    scale: &DmTensor<f32, Chip, m![Dummy2], Replicated, m![1 # 8]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
) -> DmTensor<bf16, Chip, m![Ns / 4], Slice, m![Ns % 4, Gs, Ds]> {
    type QueryClusters = m![Qs / 2048];
    type QueryRows = m![Qs / 8 % 256];

    // 두 cluster, 모든 slice에 같은 H가 이미 복제돼 있어 출력 행 매핑으로 선택할 수 있다.
    let x: DmTensorView<'_, f8e4m3, Chip, QueryClusters, QueryRows, m![QkvFp8Term, H]> = unsafe { terms.view().reshape() };
    let x_trf: TrfTensor<f8e4m3, Chip, QueryClusters, QueryRows, m![1], m![QkvFp8Term, H]> = ctx
        .sub
        .begin(x)
        .fetch::<m![QkvFp8Term, H / 32], m![H % 32]>()
        .collect::<m![QkvFp8Term, H / 32], m![H % 32]>()
        .to_trf();

    // scale도 각 실제 H replica에서 동일하게 계산됐으므로 Q 행 축으로 재해석한다.
    let input_scale: DmTensorView<'_, f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = unsafe { scale.view().reshape() };
    let input_scale_vrf: VrfTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx.sub
        .begin(input_scale).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();

    let weight_f8: DmTensor<f8e4m3, Chip, QueryClusters, QueryRows, m![Qs % 8, H]> = weight.to_dm(&mut ctx.tdma);
    let contraction: DmTensor<bf16, Chip, QueryClusters, QueryRows, m![Qs % 8]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Qs % 8, H / 32], m![H % 32]>()
        .collect::<m![Qs % 8, H / 32], m![H % 32]>()
        .contract_outer::<m![Qs % 8, H / 32, QkvFp8Term], m![H % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs % 8]>()
        .contract_lane::<m![Qs % 8], m![1 # 8]>(LaneMode::Interleaved)
        // 全H×Term sum을 f32로 복원한 뒤 기존 첫 BF16 dot 경계로 돌아간다.
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &input_scale_vrf)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
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
fn project_one_kv_matrix_packed_e113(
    ctx: &mut Context,
    x_trf: &TrfTensor<f8e4m3, Chip, KvClusters, KvRows, m![1], m![QkvFp8Term, H]>,
    input_scale_vrf: &VrfTensor<f32, Chip, KvClusters, KvRows, m![1 # 8]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> DmTensor<bf16, Chip, KvClusters, Slice, m![Ps % 1024]> {
    let weight_f8: DmTensor<f8e4m3, Chip, KvClusters, KvRows, m![Ps % 4, H]> = weight.to_dm(&mut ctx.tdma);
    let contraction: DmTensor<bf16, Chip, KvClusters, KvRows, m![Ps % 4]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Ps % 4, H / 32], m![H % 32]>()
        .collect::<m![Ps % 4, H / 32], m![H % 32]>()
        .contract_outer::<m![Ps % 4, H / 32, QkvFp8Term], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ps % 4]>()
        .contract_lane::<m![Ps % 4], m![1 # 8]>(LaneMode::Interleaved)
        // 全H×Term sum을 f32로 복원한 뒤 기존 첫 BF16 dot 경계로 돌아간다.
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), input_scale_vrf)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
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
fn project_one_value_matrix_packed_e113(
    ctx: &mut Context,
    x_trf: &TrfTensor<f8e4m3, Chip, KvClusters, KvRows, m![1], m![QkvFp8Term, H]>,
    input_scale_vrf: &VrfTensor<f32, Chip, KvClusters, KvRows, m![1 # 8]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> DmTensor<bf16, Chip, m![Ns / 4], ValueHeadSlices, m![Ds]> {
    let weight_f8: DmTensor<f8e4m3, Chip, KvClusters, KvRows, m![Ps % 4, H]> = weight.to_dm(&mut ctx.tdma);
    let contraction: DmTensor<bf16, Chip, KvClusters, KvRows, m![Ps % 4]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Ps % 4, H / 32], m![H % 32]>()
        .collect::<m![Ps % 4, H / 32], m![H % 32]>()
        .contract_outer::<m![Ps % 4, H / 32, QkvFp8Term], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ps % 4]>()
        .contract_lane::<m![Ps % 4], m![1 # 8]>(LaneMode::Interleaved)
        // 全H×Term sum을 f32로 복원한 뒤 기존 첫 BF16 dot 경계로 돌아간다.
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), input_scale_vrf)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
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

    // 각 row slice의 네 값은 한 head의 연속 Ds%4이다. 물리 순서만 재명명한다.
    let scaled: DmTensor<bf16, Chip, m![Ns / 4], m![Ns % 4, Ds / 4], m![Ds % 4]> = unsafe { scaled.reshape() };
    // 64개 row slice씩 모아 각 head를 독립 slice에 둔다.
    ctx.main
        .begin(scaled.view())
        .fetch::<m![1], m![Ds % 4 # 16]>()
        .switch::<ValueHeadSlices, m![Ds / 4]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
        .collect::<m![Ds / 4], m![Ds % 4 # 16]>()
        .commit_trim::<m![Ds % 4]>()
        .commit()
}
pub(crate) fn project_key_value_packed_e113(
    ctx: &mut Context,
    terms: &DmTensor<f8e4m3, Chip, m![Dummy2], Replicated, m![QkvFp8Term, H]>,
    scale: &DmTensor<f32, Chip, m![Dummy2], Replicated, m![1 # 8]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, m![Ns / 4], Slice, m![Ns % 4, Ds]>,
    DmTensor<bf16, Chip, m![Ns / 4], ValueHeadSlices, m![Ds]>,
) {
    // H가 양쪽 cluster와 모든 slice에 복제돼 있으므로 각 K/V 행이 같은 입력을 선택한다.
    let x: DmTensorView<'_, f8e4m3, Chip, KvClusters, KvRows, m![QkvFp8Term, H]> = unsafe { terms.view().reshape() };
    let x_trf: TrfTensor<f8e4m3, Chip, KvClusters, KvRows, m![1], m![QkvFp8Term, H]> = ctx
        .sub
        .begin(x)
        .fetch::<m![QkvFp8Term, H / 32], m![H % 32]>()
        .collect::<m![QkvFp8Term, H / 32], m![H % 32]>()
        .to_trf();

    // scale도 각 실제 H replica에서 동일하게 계산됐으므로 Q 행 축으로 재해석한다.
    let input_scale: DmTensorView<'_, f32, Chip, KvClusters, KvRows, m![1 # 8]> = unsafe { scale.view().reshape() };
    let input_scale_vrf: VrfTensor<f32, Chip, KvClusters, KvRows, m![1 # 8]> = ctx.sub
        .begin(input_scale).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();

    let k: DmTensor<bf16, Chip, KvClusters, Slice, m![Ps % 1024]> = project_one_kv_matrix_packed_e113(ctx, &x_trf, &input_scale_vrf, k_weight, k_weight_scale);
    let v: DmTensor<bf16, Chip, m![Ns / 4], ValueHeadSlices, m![Ds]> = project_one_value_matrix_packed_e113(ctx, &x_trf, &input_scale_vrf, v_weight, v_weight_scale);

    // Ps의 cluster당 1,024개 값은 Ns의 4개 head × Ds와 동일한 연속 배치다.
    (unsafe { k.reshape() }, v)
}


use crate::axes::Dummy8;

// E118: 각 H480 분할에서 인코딩하고 두 F8 항을 전체H로 실제 broadcast한다.
pub(crate) fn prepare_qkv_packed_local_e118(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, m![Dummy2], m![1 # 32, H / 480], m![H % 480]>,
) -> (
    DmTensor<f8e4m3, Chip, m![Dummy2], Replicated, m![QkvFp8Term, H]>,
    DmTensor<f32, Chip, m![Dummy2], Replicated, m![1 # 8]>,
) {
    type LocalSlices = m![1 # 32, H / 480];
    // 각 8-slice group에 실제 H3840이 분할되어 있다. BF16→f32와 abs는 E113과 같다.
    let local_max: DmTensor<f32, Chip, m![Dummy2], LocalSlices, m![1 # 8]> = ctx.main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffff)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .commit_trim::<m![1 # 8]>().commit();
    // max 결합은 추가 반올림이 없다. 실제8slice inter-reduce 후 하나의 global scale을 만든다.
    let scale: DmTensor<f32, Chip, m![Dummy2], m![1 # 32, Dummy8], m![1 # 8]> = ctx.main
        .begin(local_max.view()).fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>().vector_init()
        .vector_inter_slice_reduce::<m![1 # 32, Dummy8], m![1]>(InterSliceReduceOpF32::Max)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>().vector_fp_div(256.0)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1e-30 / 256.0)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    // inter-reduce는 유효한 첫8slice에 같은scale을 쓴다. 앞1#32는 padding이다.
    let scale_local: DmTensorView<'_, f32, Chip, m![Dummy2], LocalSlices, m![1 # 8]> = unsafe { scale.view().reshape() };
    let scale_vrf: VrfTensor<f32, Chip, m![Dummy2], LocalSlices, m![1 # 8]> = ctx.sub
        .begin(scale_local).fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>().to_vrf();
    let mut local_terms: DmTensor<f8e4m3, Chip, m![Dummy2], LocalSlices, m![QkvFp8Term, H % 480]> = DmTensor::new();
    ctx.main.begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>().vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>().commit_trim::<m![H % 8]>()
        .commit_view(local_terms.view_mut().tile::<m![QkvFp8Term], 1, m![QkvFp8Term = 1 #{!} 2, H % 480]>(0));
    let hi_vrf: VrfTensor<f32, Chip, m![Dummy2], LocalSlices, m![H % 480]> = ctx.sub
        .begin(local_terms.view().tile::<m![QkvFp8Term], 1, m![QkvFp8Term = 1 # 2, H % 480]>(0))
        .fetch::<m![H / 32 % 15], m![H % 32]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>().to_vrf();
    ctx.main.begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>().vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>().commit_trim::<m![H % 8]>()
        .commit_view(local_terms.view_mut().tile::<m![QkvFp8Term], 1, m![QkvFp8Term = 1 #{!} 2, H % 480]>(1));
    // Switch OutTime=[Term,H/480]과 Packet H%480은 Term-major fullH 순서로 실제 이동한다.
    let terms: DmTensor<f8e4m3, Chip, m![Dummy2], Replicated, m![QkvFp8Term, H]> = ctx.main
        .begin(local_terms.view())
        .fetch::<m![QkvFp8Term], m![H % 480]>()
        .switch::<Replicated, m![QkvFp8Term, H / 480]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![QkvFp8Term, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>().commit();
    // E118 수정: 1#32는 복제가 아니라 padding이다. 첫8유효slice의 같은scalar 중
    // slice0만 선택하는 view는 유효 범위를 좁힌다. 이어 실제Switch로256slice를 채운다.
    let scale_one: DmTensorView<'_, f32, Chip, m![Dummy2], Slice, m![1 # 8]> = unsafe { scale.view().reshape() };
    let scale: DmTensor<f32, Chip, m![Dummy2], Replicated, m![1 # 8]> = ctx.main
        .begin(scale_one)
        .fetch::<m![1], m![1 # 8]>()
        .switch::<Replicated, m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![1], m![1 # 8]>()
        .commit_trim::<m![1 # 8]>().commit();
    (terms, scale)
}

axes![Fp8Term = 2];
pub(crate) fn project_output_interleaved_e119(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, OutputClusters, OutputRowsByColumns, m![Qs % 256]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> HbmTensor<bf16, Chip, m![H]> {
    type InterleavedClusters = m![H / 480 % 2];
    type InterleavedRowsByColumns = m![H / 960, H / 120 % 4, Qs / 256];
    type InterleavedHiddenRows = m![H / 960, H / 120 % 4, 1 # 16];
    // x는 H와 독립적으로 이미 전체 32개 행 group에 복제돼 있다.
    // 물리 (cluster,row,col,element)의 값은 x[col*256+element]이므로
    // H의 row/cluster 이름만 바꾸어도 모든 물리 위치의 실제 값이 같다.

    // 각 slice의 실제 BF16 입력 256개에서 scale을 구한다. 영벡터도 같은 경로로 처리한다.
    let block_scale: DmTensor<f32, Chip, InterleavedClusters, InterleavedRowsByColumns, m![1 # 8]> = ctx.main
        .begin(unsafe { x.view().reshape::<Chip, InterleavedClusters, InterleavedRowsByColumns, m![Qs % 256]>() })
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffff)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_intra_slice_reduce::<Qs, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_fp_div(256.0)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1e-30 / 256.0)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    let scale_vrf: VrfTensor<f32, Chip, InterleavedClusters, InterleavedRowsByColumns, m![1 # 8]> = ctx.sub
        .begin(block_scale.view()).fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>().to_vrf();
    let mut terms: DmTensor<f8e4m3, Chip, InterleavedClusters, InterleavedRowsByColumns, m![Fp8Term, Qs % 256]> = DmTensor::new();
    // E95: 실제 term0에 높은 항을 직접 commit한다.
    ctx.main
        .begin(unsafe { x.view().reshape::<Chip, InterleavedClusters, InterleavedRowsByColumns, m![Qs % 256]>() }).fetch::<m![Qs / 16 % 16], m![Qs % 16]>().fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>().vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>().commit_trim::<m![Qs % 8]>()
        .commit_view(terms.view_mut().tile::<m![Fp8Term], 1, m![Fp8Term = 1 #{!} 2, Qs % 256]>(0));
    let hi_vrf: VrfTensor<f32, Chip, InterleavedClusters, InterleavedRowsByColumns, m![Qs % 256]> = ctx.sub
        .begin(terms.view().tile::<m![Fp8Term], 1, m![Fp8Term = 1 # 2, Qs % 256]>(0))
        .fetch::<m![Qs / 32 % 8], m![Qs % 32]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>().to_vrf();
    // 낮은 항은 실제 두 번째 half에 직접 commit한다. 첫 half를 덮어쓰지 않는다.
    ctx.main
        .begin(unsafe { x.view().reshape::<Chip, InterleavedClusters, InterleavedRowsByColumns, m![Qs % 256]>() }).fetch::<m![Qs / 16 % 16], m![Qs % 16]>().fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>().vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>().commit_trim::<m![Qs % 8]>()
        .commit_view(terms.view_mut().tile::<m![Fp8Term], 1, m![Fp8Term = 1 #{!} 2, Qs % 256]>(1));
    // term0 Sub fetch가 끝나 hi_vrf를 만든 뒤 term1을 commit한다.
    // 아래 TRF fetch는 두 실제 term의 commit을 모두 기다린다. reshape/broadcast는 없다.
    let terms_trf: TrfTensor<f8e4m3, Chip, InterleavedClusters, InterleavedRowsByColumns, m![1], m![Fp8Term, Qs % 256]> = ctx.sub
        .begin(terms.view()).fetch::<m![Fp8Term, Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Fp8Term, Qs / 32 % 8], m![Qs % 32]>().to_trf();
    // 30개의 4행 group을 비대칭 타일로 나누되 출력 물리 순서는 120행 연속이다.
    let mut result: DmTensor<bf16, Chip, InterleavedClusters, InterleavedHiddenRows, m![OutputRowGroups, OutputRowBlock]> = DmTensor::new();
    let weight0: DmTensor<f8e4m3, Chip, InterleavedClusters, InterleavedRowsByColumns, m![H % 120 = 104, Qs % 256]> =
        weight.view().tile::<m![H % 120], 104, m![H / 120, H % 120 = 104 # 120, Qs]>(0).to_dm(&mut ctx.tdma);
    let weight1: DmTensor<f8e4m3, Chip, InterleavedClusters, InterleavedRowsByColumns, m![H % 120 = 16, Qs % 256]> =
        weight.view().tile::<m![H % 120], 16, m![H / 120, H % 120 = 16 # 120, Qs]>(104).to_dm(&mut ctx.tdma);
    {
        // 각 slice의 연속 104×256 FP8 값에서 row=4*group+block이며 주소/공간축은 그대로다.
        let weight: DmTensorView<'_, f8e4m3, Chip, InterleavedClusters, InterleavedRowsByColumns, m![OutputRowGroups = 26, OutputRowBlock, Qs % 256]> =
            unsafe { weight0.view().reshape() };
        ctx.main
            .begin(weight)
            .fetch::<m![OutputRowGroups = 26, OutputRowBlock, Qs / 32 % 8], m![Qs % 32]>()
            .collect::<m![OutputRowGroups = 26, OutputRowBlock, Qs / 32 % 8], m![Qs % 32]>()
            .contract_outer::<m![OutputRowGroups = 26, OutputRowBlock, Qs / 32 % 8, Fp8Term], m![Qs % 32], _, _, _>(&terms_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![OutputRowGroups = 26, OutputRowBlock]>()
            .contract_lane::<m![OutputRowGroups = 26, OutputRowBlock], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<InterleavedHiddenRows, m![OutputRowGroups = 26, OutputRowBlock]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![OutputRowGroups = 26], m![OutputRowBlock # 16]>()
            .commit_trim::<m![OutputRowBlock]>()
            .commit_view(result.view_mut().tile::<m![OutputRowGroups], 26, m![OutputRowGroups = 26 #{!} 30, OutputRowBlock]>(0));
    }
    {
        // 각 slice의 연속 16×256 FP8 값에서 row=4*group+block이며 주소/공간축은 그대로다.
        let weight: DmTensorView<'_, f8e4m3, Chip, InterleavedClusters, InterleavedRowsByColumns, m![OutputRowGroups = 4, OutputRowBlock, Qs % 256]> =
            unsafe { weight1.view().reshape() };
        ctx.main
            .begin(weight)
            .fetch::<m![OutputRowGroups = 4, OutputRowBlock, Qs / 32 % 8], m![Qs % 32]>()
            .collect::<m![OutputRowGroups = 4, OutputRowBlock, Qs / 32 % 8], m![Qs % 32]>()
            .contract_outer::<m![OutputRowGroups = 4, OutputRowBlock, Qs / 32 % 8, Fp8Term], m![Qs % 32], _, _, _>(&terms_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![OutputRowGroups = 4, OutputRowBlock]>()
            .contract_lane::<m![OutputRowGroups = 4, OutputRowBlock], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<InterleavedHiddenRows, m![OutputRowGroups = 4, OutputRowBlock]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![OutputRowGroups = 4], m![OutputRowBlock # 16]>()
            .commit_trim::<m![OutputRowBlock]>()
            .commit_view(result.view_mut().tile::<m![OutputRowGroups], 4, m![OutputRowGroups = 4 #{!} 30, OutputRowBlock]>(26));
    }
    // group-major 30×4의 120개 bf16은 H%120과 같은 주소/순서다. padding이나 broadcast는 없다.
    let result: DmTensor<bf16, Chip, InterleavedClusters, InterleavedHiddenRows, m![H % 120]> = unsafe { result.reshape() };
    // 실제 gather로 각 cluster의 (H/960, H%480) 네 480행 chunk를 모은다.
    // H480 chunk 사이 HBM stride는 to_hbm의 논리 H mapping이 보존한다.
    let result: DmTensor<bf16, Chip, InterleavedClusters, Slice, m![H / 960, H % 480]> = result.to_dm(&mut ctx.tdma);
    result.to_hbm(&mut ctx.tdma)
}

// E164: 각 120행을 두 cluster에 번갈아 배정하여 실제 DMA 분배를 비교한다.
pub(crate) fn project_output_interleaved_e164(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, OutputClusters, OutputRowsByColumns, m![Qs % 256]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> HbmTensor<bf16, Chip, m![H]> {
    type InterleavedClusters = m![H / 120 % 2];
    type InterleavedRowsByColumns = m![H / 240, Qs / 256];
    type InterleavedHiddenRows = m![H / 240, 1 # 16];
    // x는 H와 독립적으로 이미 전체 32개 행 group에 복제돼 있다.
    // 물리 (cluster,row,col,element)의 값은 x[col*256+element]이므로
    // H의 row/cluster 이름만 바꾸어도 모든 물리 위치의 실제 값이 같다.

    // 각 slice의 실제 BF16 입력 256개에서 scale을 구한다. 영벡터도 같은 경로로 처리한다.
    let block_scale: DmTensor<f32, Chip, InterleavedClusters, InterleavedRowsByColumns, m![1 # 8]> = ctx.main
        .begin(unsafe { x.view().reshape::<Chip, InterleavedClusters, InterleavedRowsByColumns, m![Qs % 256]>() })
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffff)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_intra_slice_reduce::<Qs, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_fp_div(256.0)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1e-30 / 256.0)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    let scale_vrf: VrfTensor<f32, Chip, InterleavedClusters, InterleavedRowsByColumns, m![1 # 8]> = ctx.sub
        .begin(block_scale.view()).fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>().to_vrf();
    let mut terms: DmTensor<f8e4m3, Chip, InterleavedClusters, InterleavedRowsByColumns, m![Fp8Term, Qs % 256]> = DmTensor::new();
    // E95: 실제 term0에 높은 항을 직접 commit한다.
    ctx.main
        .begin(unsafe { x.view().reshape::<Chip, InterleavedClusters, InterleavedRowsByColumns, m![Qs % 256]>() }).fetch::<m![Qs / 16 % 16], m![Qs % 16]>().fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>().vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>().commit_trim::<m![Qs % 8]>()
        .commit_view(terms.view_mut().tile::<m![Fp8Term], 1, m![Fp8Term = 1 #{!} 2, Qs % 256]>(0));
    let hi_vrf: VrfTensor<f32, Chip, InterleavedClusters, InterleavedRowsByColumns, m![Qs % 256]> = ctx.sub
        .begin(terms.view().tile::<m![Fp8Term], 1, m![Fp8Term = 1 # 2, Qs % 256]>(0))
        .fetch::<m![Qs / 32 % 8], m![Qs % 32]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>().to_vrf();
    // 낮은 항은 실제 두 번째 half에 직접 commit한다. 첫 half를 덮어쓰지 않는다.
    ctx.main
        .begin(unsafe { x.view().reshape::<Chip, InterleavedClusters, InterleavedRowsByColumns, m![Qs % 256]>() }).fetch::<m![Qs / 16 % 16], m![Qs % 16]>().fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>().vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>().commit_trim::<m![Qs % 8]>()
        .commit_view(terms.view_mut().tile::<m![Fp8Term], 1, m![Fp8Term = 1 #{!} 2, Qs % 256]>(1));
    // term0 Sub fetch가 끝나 hi_vrf를 만든 뒤 term1을 commit한다.
    // 아래 TRF fetch는 두 실제 term의 commit을 모두 기다린다. reshape/broadcast는 없다.
    let terms_trf: TrfTensor<f8e4m3, Chip, InterleavedClusters, InterleavedRowsByColumns, m![1], m![Fp8Term, Qs % 256]> = ctx.sub
        .begin(terms.view()).fetch::<m![Fp8Term, Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Fp8Term, Qs / 32 % 8], m![Qs % 32]>().to_trf();
    // 30개의 4행 group을 비대칭 타일로 나누되 출력 물리 순서는 120행 연속이다.
    let mut result: DmTensor<bf16, Chip, InterleavedClusters, InterleavedHiddenRows, m![OutputRowGroups, OutputRowBlock]> = DmTensor::new();
    let weight0: DmTensor<f8e4m3, Chip, InterleavedClusters, InterleavedRowsByColumns, m![H % 120 = 104, Qs % 256]> =
        weight.view().tile::<m![H % 120], 104, m![H / 120, H % 120 = 104 # 120, Qs]>(0).to_dm(&mut ctx.tdma);
    let weight1: DmTensor<f8e4m3, Chip, InterleavedClusters, InterleavedRowsByColumns, m![H % 120 = 16, Qs % 256]> =
        weight.view().tile::<m![H % 120], 16, m![H / 120, H % 120 = 16 # 120, Qs]>(104).to_dm(&mut ctx.tdma);
    {
        // 각 slice의 연속 104×256 FP8 값에서 row=4*group+block이며 주소/공간축은 그대로다.
        let weight: DmTensorView<'_, f8e4m3, Chip, InterleavedClusters, InterleavedRowsByColumns, m![OutputRowGroups = 26, OutputRowBlock, Qs % 256]> =
            unsafe { weight0.view().reshape() };
        ctx.main
            .begin(weight)
            .fetch::<m![OutputRowGroups = 26, OutputRowBlock, Qs / 32 % 8], m![Qs % 32]>()
            .collect::<m![OutputRowGroups = 26, OutputRowBlock, Qs / 32 % 8], m![Qs % 32]>()
            .contract_outer::<m![OutputRowGroups = 26, OutputRowBlock, Qs / 32 % 8, Fp8Term], m![Qs % 32], _, _, _>(&terms_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![OutputRowGroups = 26, OutputRowBlock]>()
            .contract_lane::<m![OutputRowGroups = 26, OutputRowBlock], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<InterleavedHiddenRows, m![OutputRowGroups = 26, OutputRowBlock]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![OutputRowGroups = 26], m![OutputRowBlock # 16]>()
            .commit_trim::<m![OutputRowBlock]>()
            .commit_view(result.view_mut().tile::<m![OutputRowGroups], 26, m![OutputRowGroups = 26 #{!} 30, OutputRowBlock]>(0));
    }
    {
        // 각 slice의 연속 16×256 FP8 값에서 row=4*group+block이며 주소/공간축은 그대로다.
        let weight: DmTensorView<'_, f8e4m3, Chip, InterleavedClusters, InterleavedRowsByColumns, m![OutputRowGroups = 4, OutputRowBlock, Qs % 256]> =
            unsafe { weight1.view().reshape() };
        ctx.main
            .begin(weight)
            .fetch::<m![OutputRowGroups = 4, OutputRowBlock, Qs / 32 % 8], m![Qs % 32]>()
            .collect::<m![OutputRowGroups = 4, OutputRowBlock, Qs / 32 % 8], m![Qs % 32]>()
            .contract_outer::<m![OutputRowGroups = 4, OutputRowBlock, Qs / 32 % 8, Fp8Term], m![Qs % 32], _, _, _>(&terms_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![OutputRowGroups = 4, OutputRowBlock]>()
            .contract_lane::<m![OutputRowGroups = 4, OutputRowBlock], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<InterleavedHiddenRows, m![OutputRowGroups = 4, OutputRowBlock]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![OutputRowGroups = 4], m![OutputRowBlock # 16]>()
            .commit_trim::<m![OutputRowBlock]>()
            .commit_view(result.view_mut().tile::<m![OutputRowGroups], 4, m![OutputRowGroups = 4 #{!} 30, OutputRowBlock]>(26));
    }
    // group-major 30×4의 120개 bf16은 H%120과 같은 주소/순서다. padding이나 broadcast는 없다.
    let result: DmTensor<bf16, Chip, InterleavedClusters, InterleavedHiddenRows, m![H % 120]> = unsafe { result.reshape() };
    // E164: 실제 gather로 각 cluster의 H120 chunk 16개를 모은다.
    // cluster=H/120%2와 element=(H/240,H%120)가 3840개 H를 중복 없이 덮는다.
    let result: DmTensor<bf16, Chip, InterleavedClusters, Slice, m![H / 240, H % 120]> = result.to_dm(&mut ctx.tdma);
    result.to_hbm(&mut ctx.tdma)
}

// E169: preserve E164 helper and arithmetic; change only the output transfer.
pub(crate) fn project_output_direct_hbm_e169(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, OutputClusters, OutputRowsByColumns, m![Qs % 256]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
) -> HbmTensor<bf16, Chip, m![H]> {
    type InterleavedClusters = m![H / 120 % 2];
    type InterleavedRowsByColumns = m![H / 240, Qs / 256];
    type InterleavedHiddenRows = m![H / 240, 1 # 16];
    // x는 H와 독립적으로 이미 전체 32개 행 group에 복제돼 있다.
    // 물리 (cluster,row,col,element)의 값은 x[col*256+element]이므로
    // H의 row/cluster 이름만 바꾸어도 모든 물리 위치의 실제 값이 같다.

    // 각 slice의 실제 BF16 입력 256개에서 scale을 구한다. 영벡터도 같은 경로로 처리한다.
    let block_scale: DmTensor<f32, Chip, InterleavedClusters, InterleavedRowsByColumns, m![1 # 8]> = ctx.main
        .begin(unsafe { x.view().reshape::<Chip, InterleavedClusters, InterleavedRowsByColumns, m![Qs % 256]>() })
        .fetch::<m![Qs / 16 % 16], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffff)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_intra_slice_reduce::<Qs, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_fp_div(256.0)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1e-30 / 256.0)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    let scale_vrf: VrfTensor<f32, Chip, InterleavedClusters, InterleavedRowsByColumns, m![1 # 8]> = ctx.sub
        .begin(block_scale.view()).fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>().to_vrf();
    let mut terms: DmTensor<f8e4m3, Chip, InterleavedClusters, InterleavedRowsByColumns, m![Fp8Term, Qs % 256]> = DmTensor::new();
    // E95: 실제 term0에 높은 항을 직접 commit한다.
    ctx.main
        .begin(unsafe { x.view().reshape::<Chip, InterleavedClusters, InterleavedRowsByColumns, m![Qs % 256]>() }).fetch::<m![Qs / 16 % 16], m![Qs % 16]>().fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>().vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>().commit_trim::<m![Qs % 8]>()
        .commit_view(terms.view_mut().tile::<m![Fp8Term], 1, m![Fp8Term = 1 #{!} 2, Qs % 256]>(0));
    let hi_vrf: VrfTensor<f32, Chip, InterleavedClusters, InterleavedRowsByColumns, m![Qs % 256]> = ctx.sub
        .begin(terms.view().tile::<m![Fp8Term], 1, m![Fp8Term = 1 # 2, Qs % 256]>(0))
        .fetch::<m![Qs / 32 % 8], m![Qs % 32]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>().to_vrf();
    // 낮은 항은 실제 두 번째 half에 직접 commit한다. 첫 half를 덮어쓰지 않는다.
    ctx.main
        .begin(unsafe { x.view().reshape::<Chip, InterleavedClusters, InterleavedRowsByColumns, m![Qs % 256]>() }).fetch::<m![Qs / 16 % 16], m![Qs % 16]>().fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>().vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>().commit_trim::<m![Qs % 8]>()
        .commit_view(terms.view_mut().tile::<m![Fp8Term], 1, m![Fp8Term = 1 #{!} 2, Qs % 256]>(1));
    // term0 Sub fetch가 끝나 hi_vrf를 만든 뒤 term1을 commit한다.
    // 아래 TRF fetch는 두 실제 term의 commit을 모두 기다린다. reshape/broadcast는 없다.
    let terms_trf: TrfTensor<f8e4m3, Chip, InterleavedClusters, InterleavedRowsByColumns, m![1], m![Fp8Term, Qs % 256]> = ctx.sub
        .begin(terms.view()).fetch::<m![Fp8Term, Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![Fp8Term, Qs / 32 % 8], m![Qs % 32]>().to_trf();
    // 30개의 4행 group을 비대칭 타일로 나누되 출력 물리 순서는 120행 연속이다.
    let mut result: DmTensor<bf16, Chip, InterleavedClusters, InterleavedHiddenRows, m![OutputRowGroups, OutputRowBlock]> = DmTensor::new();
    let weight0: DmTensor<f8e4m3, Chip, InterleavedClusters, InterleavedRowsByColumns, m![H % 120 = 104, Qs % 256]> =
        weight.view().tile::<m![H % 120], 104, m![H / 120, H % 120 = 104 # 120, Qs]>(0).to_dm(&mut ctx.tdma);
    let weight1: DmTensor<f8e4m3, Chip, InterleavedClusters, InterleavedRowsByColumns, m![H % 120 = 16, Qs % 256]> =
        weight.view().tile::<m![H % 120], 16, m![H / 120, H % 120 = 16 # 120, Qs]>(104).to_dm(&mut ctx.tdma);
    {
        // 각 slice의 연속 104×256 FP8 값에서 row=4*group+block이며 주소/공간축은 그대로다.
        let weight: DmTensorView<'_, f8e4m3, Chip, InterleavedClusters, InterleavedRowsByColumns, m![OutputRowGroups = 26, OutputRowBlock, Qs % 256]> =
            unsafe { weight0.view().reshape() };
        ctx.main
            .begin(weight)
            .fetch::<m![OutputRowGroups = 26, OutputRowBlock, Qs / 32 % 8], m![Qs % 32]>()
            .collect::<m![OutputRowGroups = 26, OutputRowBlock, Qs / 32 % 8], m![Qs % 32]>()
            .contract_outer::<m![OutputRowGroups = 26, OutputRowBlock, Qs / 32 % 8, Fp8Term], m![Qs % 32], _, _, _>(&terms_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![OutputRowGroups = 26, OutputRowBlock]>()
            .contract_lane::<m![OutputRowGroups = 26, OutputRowBlock], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<InterleavedHiddenRows, m![OutputRowGroups = 26, OutputRowBlock]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![OutputRowGroups = 26], m![OutputRowBlock # 16]>()
            .commit_trim::<m![OutputRowBlock]>()
            .commit_view(result.view_mut().tile::<m![OutputRowGroups], 26, m![OutputRowGroups = 26 #{!} 30, OutputRowBlock]>(0));
    }
    {
        // 각 slice의 연속 16×256 FP8 값에서 row=4*group+block이며 주소/공간축은 그대로다.
        let weight: DmTensorView<'_, f8e4m3, Chip, InterleavedClusters, InterleavedRowsByColumns, m![OutputRowGroups = 4, OutputRowBlock, Qs % 256]> =
            unsafe { weight1.view().reshape() };
        ctx.main
            .begin(weight)
            .fetch::<m![OutputRowGroups = 4, OutputRowBlock, Qs / 32 % 8], m![Qs % 32]>()
            .collect::<m![OutputRowGroups = 4, OutputRowBlock, Qs / 32 % 8], m![Qs % 32]>()
            .contract_outer::<m![OutputRowGroups = 4, OutputRowBlock, Qs / 32 % 8, Fp8Term], m![Qs % 32], _, _, _>(&terms_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![OutputRowGroups = 4, OutputRowBlock]>()
            .contract_lane::<m![OutputRowGroups = 4, OutputRowBlock], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<InterleavedHiddenRows, m![OutputRowGroups = 4, OutputRowBlock]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![OutputRowGroups = 4], m![OutputRowBlock # 16]>()
            .commit_trim::<m![OutputRowBlock]>()
            .commit_view(result.view_mut().tile::<m![OutputRowGroups], 4, m![OutputRowGroups = 4 #{!} 30, OutputRowBlock]>(26));
    }
    // group-major 30×4의 120개 bf16은 H%120과 같은 주소/순서다. padding이나 broadcast는 없다.
    let result: DmTensor<bf16, Chip, InterleavedClusters, InterleavedHiddenRows, m![H % 120]> = unsafe { result.reshape() };
    // E169: direct HBM write from the actual row owners, without local DM gather.
    // Valid owner c=H/120%2, slice=16*(H/240), local e=H%120.
    // The logical H destination assigns exactly one 240-byte row to each owner.
    // 1#16 marks only column0 valid; no padding/replica stores are requested.
    result.to_hbm(&mut ctx.tdma)
}


pub(crate) fn prepare_qkv_inverse_scale_e192(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, m![Dummy2], m![1 # 32, H / 480], m![H % 480]>,
) -> (
    DmTensor<f8e4m3, Chip, m![Dummy2], Replicated, m![QkvFp8Term, H]>,
    DmTensor<f32, Chip, m![Dummy2], Replicated, m![1 # 8]>,
) {
    type LocalSlices = m![1 # 32, H / 480];
    // 각 8-slice group에 실제 H3840이 분할되어 있다. BF16→f32와 abs는 E113과 같다.
    let local_max: DmTensor<f32, Chip, m![Dummy2], LocalSlices, m![1 # 8]> = ctx.main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_reinterpret::<i32>()
        .vector_logic(LogicBinaryOpI32::BitAnd, 0x7fff_ffff)
        .vector_reinterpret::<f32>()
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .commit_trim::<m![1 # 8]>().commit();
    // max 결합은 추가 반올림이 없다. 실제8slice inter-reduce 후 하나의 global scale을 만든다.
    let scale: DmTensor<f32, Chip, m![Dummy2], m![1 # 32, Dummy8], m![1 # 8]> = ctx.main
        .begin(local_max.view()).fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>().vector_init()
        .vector_inter_slice_reduce::<m![1 # 32, Dummy8], m![1]>(InterSliceReduceOpF32::Max)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>().vector_fp_div(256.0)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Max, 1e-30 / 256.0)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    // inter-reduce는 유효한 첫8slice에 같은scale을 쓴다. 앞1#32는 padding이다.
    let scale_local: DmTensorView<'_, f32, Chip, m![Dummy2], LocalSlices, m![1 # 8]> = unsafe { scale.view().reshape() };
    // E192: encoding용 reciprocal은 scalar마다 한 번만 계산한다. 반환 scale은 원래값이다.
    // Sub에서 계산→VRF는 E190 actual 실패. Main이 역수를 실제 DM에 저장하고 Sub는 raw fetch만 한다.
    let inverse_scale: DmTensor<f32, Chip, m![Dummy2], LocalSlices, m![1 # 8]> = ctx.main
        .begin(scale_local).fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0)
        .vector_widen_pad::<m![1 # 8]>().vector_final().commit_trim::<m![1 # 8]>().commit();
    let scale_vrf: VrfTensor<f32, Chip, m![Dummy2], LocalSlices, m![1 # 8]> = ctx.sub
        .begin(inverse_scale.view()).fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>().to_vrf();
    let mut local_terms: DmTensor<f8e4m3, Chip, m![Dummy2], LocalSlices, m![QkvFp8Term, H % 480]> = DmTensor::new();
    ctx.main.begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>().vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>().commit_trim::<m![H % 8]>()
        .commit_view(local_terms.view_mut().tile::<m![QkvFp8Term], 1, m![QkvFp8Term = 1 #{!} 2, H % 480]>(0));
    let hi_vrf: VrfTensor<f32, Chip, m![Dummy2], LocalSlices, m![H % 480]> = ctx.sub
        .begin(local_terms.view().tile::<m![QkvFp8Term], 1, m![QkvFp8Term = 1 # 2, H % 480]>(0))
        .fetch::<m![H / 32 % 15], m![H % 32]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>().to_vrf();
    ctx.main.begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_fp_binary(FpBinaryOp::SubF, &hi_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>().vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>().commit_trim::<m![H % 8]>()
        .commit_view(local_terms.view_mut().tile::<m![QkvFp8Term], 1, m![QkvFp8Term = 1 #{!} 2, H % 480]>(1));
    // Switch OutTime=[Term,H/480]과 Packet H%480은 Term-major fullH 순서로 실제 이동한다.
    let terms: DmTensor<f8e4m3, Chip, m![Dummy2], Replicated, m![QkvFp8Term, H]> = ctx.main
        .begin(local_terms.view())
        .fetch::<m![QkvFp8Term], m![H % 480]>()
        .switch::<Replicated, m![QkvFp8Term, H / 480]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![QkvFp8Term, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>().commit();
    // E118 수정: 1#32는 복제가 아니라 padding이다. 첫8유효slice의 같은scalar 중
    // slice0만 선택하는 view는 유효 범위를 좁힌다. 이어 실제Switch로256slice를 채운다.
    let scale_one: DmTensorView<'_, f32, Chip, m![Dummy2], Slice, m![1 # 8]> = unsafe { scale.view().reshape() };
    let scale: DmTensor<f32, Chip, m![Dummy2], Replicated, m![1 # 8]> = ctx.main
        .begin(scale_one)
        .fetch::<m![1], m![1 # 8]>()
        .switch::<Replicated, m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![1], m![1 # 8]>()
        .commit_trim::<m![1 # 8]>().commit();
    (terms, scale)
}

// E194: 중간 dot BF16 반올림을 최종 채널 scale 뒤로 합치는 후보.
pub(crate) fn project_query_fused_e194(
    ctx: &mut Context,
    terms: &DmTensor<f8e4m3, Chip, m![Dummy2], Replicated, m![QkvFp8Term, H]>,
    scale: &DmTensor<f32, Chip, m![Dummy2], Replicated, m![1 # 8]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
) -> DmTensor<bf16, Chip, m![Ns / 4], Slice, m![Ns % 4, Gs, Ds]> {
    type QueryClusters = m![Qs / 2048];
    type QueryRows = m![Qs / 8 % 256];

    // 두 cluster, 모든 slice에 같은 H가 이미 복제돼 있어 출력 행 매핑으로 선택할 수 있다.
    let x: DmTensorView<'_, f8e4m3, Chip, QueryClusters, QueryRows, m![QkvFp8Term, H]> = unsafe { terms.view().reshape() };
    let x_trf: TrfTensor<f8e4m3, Chip, QueryClusters, QueryRows, m![1], m![QkvFp8Term, H]> = ctx
        .sub
        .begin(x)
        .fetch::<m![QkvFp8Term, H / 32], m![H % 32]>()
        .collect::<m![QkvFp8Term, H / 32], m![H % 32]>()
        .to_trf();

    // scale도 각 실제 H replica에서 동일하게 계산됐으므로 Q 행 축으로 재해석한다.
    let input_scale: DmTensorView<'_, f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = unsafe { scale.view().reshape() };
    let input_scale_vrf: VrfTensor<f32, Chip, QueryClusters, QueryRows, m![1 # 8]> = ctx.sub
        .begin(input_scale).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();

    // E194: 행별 channel scale을 Time축에 배치하여 dot의 각 출력 row와 대응한다.
    // 실제 HBM DMA가 각 weight_scale[row]를 옮기며 padding을 복제로 해석하지 않는다.
    let channel_dm: DmTensor<bf16, Chip, QueryClusters, QueryRows, m![Qs % 8]> = weight_scale.to_dm(&mut ctx.tdma);
    let channel_vrf: VrfTensor<f32, Chip, QueryClusters, QueryRows, m![Qs % 8, 1 # 8]> = ctx.sub
        .begin(channel_dm.view()).fetch::<m![Qs % 8], m![1 # 8]>()
        .fetch_cast::<f32>().collect::<m![Qs % 8], m![1 # 8]>().to_vrf();
    let weight_f8: DmTensor<f8e4m3, Chip, QueryClusters, QueryRows, m![Qs % 8, H]> = weight.to_dm(&mut ctx.tdma);
    let contraction: DmTensor<bf16, Chip, QueryClusters, QueryRows, m![Qs % 8]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Qs % 8, H / 32], m![H % 32]>()
        .collect::<m![Qs % 8, H / 32], m![H % 32]>()
        .contract_outer::<m![Qs % 8, H / 32, QkvFp8Term], m![H % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs % 8]>()
        .contract_lane::<m![Qs % 8], m![1 # 8]>(LaneMode::Interleaved)
        // 全H×Term sum을 f32로 복원하고 channel 곱까지 f32로 처리한다.
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &input_scale_vrf)
        // 첫 dot BF16 반올림을 없애고 channel 곱 뒤 BF16으로 반올림한다.
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &channel_vrf)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 4 % 2], m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>()
        .commit();

    let scaled = contraction;


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
fn project_one_kv_fused_e194(
    ctx: &mut Context,
    x_trf: &TrfTensor<f8e4m3, Chip, KvClusters, KvRows, m![1], m![QkvFp8Term, H]>,
    input_scale_vrf: &VrfTensor<f32, Chip, KvClusters, KvRows, m![1 # 8]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> DmTensor<bf16, Chip, KvClusters, Slice, m![Ps % 1024]> {
    // E194: 행별 channel scale을 Time축에 배치하여 dot의 각 출력 row와 대응한다.
    // 실제 HBM DMA가 각 weight_scale[row]를 옮기며 padding을 복제로 해석하지 않는다.
    let channel_dm: DmTensor<bf16, Chip, KvClusters, KvRows, m![Ps % 4]> = weight_scale.to_dm(&mut ctx.tdma);
    let channel_vrf: VrfTensor<f32, Chip, KvClusters, KvRows, m![Ps % 4, 1 # 8]> = ctx.sub
        .begin(channel_dm.view()).fetch::<m![Ps % 4], m![1 # 8]>()
        .fetch_cast::<f32>().collect::<m![Ps % 4], m![1 # 8]>().to_vrf();
    let weight_f8: DmTensor<f8e4m3, Chip, KvClusters, KvRows, m![Ps % 4, H]> = weight.to_dm(&mut ctx.tdma);
    let contraction: DmTensor<bf16, Chip, KvClusters, KvRows, m![Ps % 4]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Ps % 4, H / 32], m![H % 32]>()
        .collect::<m![Ps % 4, H / 32], m![H % 32]>()
        .contract_outer::<m![Ps % 4, H / 32, QkvFp8Term], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ps % 4]>()
        .contract_lane::<m![Ps % 4], m![1 # 8]>(LaneMode::Interleaved)
        // 全H×Term sum을 f32로 복원하고 channel 곱까지 f32로 처리한다.
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), input_scale_vrf)
        // 첫 dot BF16 반올림을 없애고 channel 곱 뒤 BF16으로 반올림한다.
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &channel_vrf)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![1], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();

    let scaled = contraction;


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
fn project_one_value_fused_e194(
    ctx: &mut Context,
    x_trf: &TrfTensor<f8e4m3, Chip, KvClusters, KvRows, m![1], m![QkvFp8Term, H]>,
    input_scale_vrf: &VrfTensor<f32, Chip, KvClusters, KvRows, m![1 # 8]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> DmTensor<bf16, Chip, m![Ns / 4], ValueHeadSlices, m![Ds]> {
    // E194: 행별 channel scale을 Time축에 배치하여 dot의 각 출력 row와 대응한다.
    // 실제 HBM DMA가 각 weight_scale[row]를 옮기며 padding을 복제로 해석하지 않는다.
    let channel_dm: DmTensor<bf16, Chip, KvClusters, KvRows, m![Ps % 4]> = weight_scale.to_dm(&mut ctx.tdma);
    let channel_vrf: VrfTensor<f32, Chip, KvClusters, KvRows, m![Ps % 4, 1 # 8]> = ctx.sub
        .begin(channel_dm.view()).fetch::<m![Ps % 4], m![1 # 8]>()
        .fetch_cast::<f32>().collect::<m![Ps % 4], m![1 # 8]>().to_vrf();
    let weight_f8: DmTensor<f8e4m3, Chip, KvClusters, KvRows, m![Ps % 4, H]> = weight.to_dm(&mut ctx.tdma);
    let contraction: DmTensor<bf16, Chip, KvClusters, KvRows, m![Ps % 4]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Ps % 4, H / 32], m![H % 32]>()
        .collect::<m![Ps % 4, H / 32], m![H % 32]>()
        .contract_outer::<m![Ps % 4, H / 32, QkvFp8Term], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ps % 4]>()
        .contract_lane::<m![Ps % 4], m![1 # 8]>(LaneMode::Interleaved)
        // 全H×Term sum을 f32로 복원하고 channel 곱까지 f32로 처리한다.
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), input_scale_vrf)
        // 첫 dot BF16 반올림을 없애고 channel 곱 뒤 BF16으로 반올림한다.
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &channel_vrf)
        .vector_widen_pad::<m![1 # 8]>().vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![1], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();

    let scaled = contraction;


    // 각 row slice의 네 값은 한 head의 연속 Ds%4이다. 물리 순서만 재명명한다.
    let scaled: DmTensor<bf16, Chip, m![Ns / 4], m![Ns % 4, Ds / 4], m![Ds % 4]> = unsafe { scaled.reshape() };
    // 64개 row slice씩 모아 각 head를 독립 slice에 둔다.
    ctx.main
        .begin(scaled.view())
        .fetch::<m![1], m![Ds % 4 # 16]>()
        .switch::<ValueHeadSlices, m![Ds / 4]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
        .collect::<m![Ds / 4], m![Ds % 4 # 16]>()
        .commit_trim::<m![Ds % 4]>()
        .commit()
}
pub(crate) fn project_key_value_fused_e194(
    ctx: &mut Context,
    terms: &DmTensor<f8e4m3, Chip, m![Dummy2], Replicated, m![QkvFp8Term, H]>,
    scale: &DmTensor<f32, Chip, m![Dummy2], Replicated, m![1 # 8]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, m![Ns / 4], Slice, m![Ns % 4, Ds]>,
    DmTensor<bf16, Chip, m![Ns / 4], ValueHeadSlices, m![Ds]>,
) {
    // H가 양쪽 cluster와 모든 slice에 복제돼 있으므로 각 K/V 행이 같은 입력을 선택한다.
    let x: DmTensorView<'_, f8e4m3, Chip, KvClusters, KvRows, m![QkvFp8Term, H]> = unsafe { terms.view().reshape() };
    let x_trf: TrfTensor<f8e4m3, Chip, KvClusters, KvRows, m![1], m![QkvFp8Term, H]> = ctx
        .sub
        .begin(x)
        .fetch::<m![QkvFp8Term, H / 32], m![H % 32]>()
        .collect::<m![QkvFp8Term, H / 32], m![H % 32]>()
        .to_trf();

    // scale도 각 실제 H replica에서 동일하게 계산됐으므로 Q 행 축으로 재해석한다.
    let input_scale: DmTensorView<'_, f32, Chip, KvClusters, KvRows, m![1 # 8]> = unsafe { scale.view().reshape() };
    let input_scale_vrf: VrfTensor<f32, Chip, KvClusters, KvRows, m![1 # 8]> = ctx.sub
        .begin(input_scale).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();

    let k: DmTensor<bf16, Chip, KvClusters, Slice, m![Ps % 1024]> = project_one_kv_fused_e194(ctx, &x_trf, &input_scale_vrf, k_weight, k_weight_scale);
    let v: DmTensor<bf16, Chip, m![Ns / 4], ValueHeadSlices, m![Ds]> = project_one_value_fused_e194(ctx, &x_trf, &input_scale_vrf, v_weight, v_weight_scale);

    // Ps의 cluster당 1,024개 값은 Ns의 4개 head × Ds와 동일한 연속 배치다.
    (unsafe { k.reshape() }, v)
}

