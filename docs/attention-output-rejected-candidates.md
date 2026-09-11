# Attention output 기각·보류 후보 기록

작성일: 2026-09-10. 대상은 Stage 1 `sliding_attention_output`이다. E98, E100, E101, E102, E104, E106의 결과를 모아 같은 가설을 반복하는 일을 줄인다. 이 문서 작성 과정에서는 구현 소스와 다른 문서를 변경하지 않았다.

## 비교 원칙과 현재 판단

- **Host 검사**, **SDK 0.6.0 정적 스케줄**, **실제 RNGD 결과**를 구분한다. 정적 시간은 `max(instructions[*].lifetime.end)`이며 실제 cycle과 같지 않다.
- O의 원소별 허용량 사용률은 `budgetfrac = abs(actual-expected) / (0.05 + 0.01*abs(expected))`이다. 1을 초과하거나 출력이 비유한이면 실패한다.
- 아래 18조건 host 검사는 3개 seed × amplitude `[0, 1e-8, 0.001, 1, 16, 10000]`이다. 공개 PRNG·참조 연산을 사용하지만 RNGD의 정확한 Packet/Time/Lane 누산 순서까지 재현하지 않는다.
- 공개 AB/BA 성능은 해당 job의 같은 fixture·동결한 바이너리끼리 비교한다. Continuous 3조건은 정확성 검증이며 공개 성능 통계에 합치지 않는다.
- **E102의 직접 기준은 E95이고 나머지 정적 후보의 기준은 E96이다.** E98은 E87 반올림 경계와 E96 융합 경계를 각각 host에서 비교했다. 서로 다른 수치 경로·입력·job의 최솟값을 조합해 개선율이나 목표 달성을 주장하지 않는다.

| 후보 | 핵심 변경 | 정적 기준 → 후보 | 실제 O 판단 | 상태 |
|---|---|---:|---|---|
| E98 | 동적 FP8에서 lo 보정항 삭제 | 미컴파일 | 미실행, host 12/18조건 실패 | 정확성 사전 검사에서 기각 |
| E100 | Projection의 BF16 중간값을 F32 HBM 전달로 변경 | 23,806 → 23,977 | E96 대비 8쌍 중앙값 +1.54%, 평균 +0.35% | 보류 |
| E101a / E101b | 입력 Div 두 개를 reciprocal × Mul로 변경 | 둘 다 23,806 → 23,806 | 미실행 | 정적 이득 없어 보류 |
| E102a | 전체 HBM 전달 대신 두 cluster에서 RMSNorm, scalar만 교환 | E95 24,024 → 24,428 | E95 대비 4/4쌍 회귀, 중앙값 +19.37% | 기각 |
| E102b | E102a의 최종 gather를 없애고 직접 HBM 쓰기 | E102a 24,428 → 25,128 | 미실행 | 정적 회귀로 기각 |
| E104 | 두 cluster row32 × col8, local H60/Q512 | 23,806 → 24,267 | E96 대비 2/4쌍 개선, 평균 +0.18% | 지속 개선 미입증, 미채택 |
| E106 | 행 그룹 30×4를 15×8로 변경 | 23,806 → 23,806 | 미실행 | 정적 이득 없어 보류 |

하드웨어를 실행한 E100, E102a, E104는 해당 검증에서 세 커널 정확성을 모두 통과했다. Host 통과만 확인한 후보에는 하드웨어 PASS를 부여하지 않는다.

## E98 — 동적 FP8의 상위항만 사용

**변경.** Q256 block마다 `s=max(maxabs(x)/256,1e-30/256)`을 구하고 `hi=FP8(x/s)`만 남긴다. 기존 `lo=FP8(x/s-hi)` 보정항을 삭제하되 block scale 복원과 후처리의 의미는 유지하는 가설이다.

**Host 결과.** E87의 BF16 경계를 유지한 경우와 E96의 channel scale·norm·residual 융합 경계를 따로 검사했다.

| 후처리 경계 | 실패 조건 | 실패 원소 합 | 최대 절대 오차 | 최대 budgetfrac |
|---|---:|---:|---:|---:|
| E87 BF16 경계 | 12 / 18 | 212 | 0.140625 | 2.117421 |
| E96 융합 경계 | 12 / 18 | 200 | 0.140625 | 2.190722 |

BF16 입력 대조군과 hi+lo 대조군은 두 후처리 경계 모두 18조건을 통과했다. 이들의 최대 budgetfrac는 0.440529였다. Hi-only는 amplitude 0과 1e-8에서만 모든 seed를 통과하고 나머지 네 amplitude에서 모두 실패했다.

정규화 입력의 최대 절댓값은 256이고 FP8 비유한 cast, 448 초과, nonzero→zero는 모두 0이다. 입력의 상대 L2 양자화 오차가 약 2.33–2.36%이므로 범위 초과나 underflow를 고치는 문제로 분류하지 않는다. 보정항 삭제에 따른 정밀도 손실이 남는다.

**정적·실제 결과.** NPU 후보를 컴파일하거나 Arena에 제출하지 않았다. Job ID가 없다.

**SDK 대안 조사.** SDK 0.6.0의 `ContractionWeight`는 동일 float dtype끼리, 또는 같은 정수 가족끼리만 허용한다. `i8 × f8e4m3`와 반대 방향은 지원하지 않는다. Int8 activation을 생성할 수 있어도 원래 FP8 weight와 한 native contraction으로 결합할 수 없다. 저장 byte를 i8로 재해석하는 것은 실제 weight 값의 변환이 아니다. 근거는 [SDK cast 계약](/scale/cal/home/sehwan/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/furiosa-opt-std-0.6.0/src/cast.rs:321)과 [contract_outer bound](/scale/cal/home/sehwan/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/furiosa-opt-std-0.6.0/src/engine/contraction/outer/mod.rs:124)다.

**중복 실험 방지.** 같은 Q256 maxabs/256, e4m3 hi-only 식은 후처리 BF16 경계만 바꿔 다시 제출하지 않는다. 새 오차 보정이나 다른 지원 dtype처럼 양자화 오차 자체를 줄이는 일반적 가설이 있어야 재검토한다. 혼합 i8×FP8 경로는 공개 SDK 지원이 달라졌다는 근거가 있을 때만 다시 조사한다.

보존 자료: [host 소스](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-ffn-blockdot/o-e98-hi-only/scripts/screen_e98.py), [18조건 결과](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-ffn-blockdot/o-e98-hi-only/artifacts/host-screen.json), [원본 host 로그](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-ffn-blockdot/o-e98-hi-only/artifacts/host-screen.log), [상세 분석](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-ffn-blockdot/o-e98-hi-only/docs/e98-analysis.md).

## E100 — F32 projection 중간값을 HBM으로 전달

**변경.** E96의 `dot → BF16 → norm용 F32` 경계를 없애고 F32 projection을 HBM으로 전달한다. 중간 전달량은 BF16 7,680 bytes에서 F32 15,360 bytes로 늘어난다. 중간 BF16 반올림이 하나 사라지므로 수치 변경 후보이며 E50의 QKV/FFN은 고정했다.

**Host 결과.** 18조건 모두 통과했고 최대 budgetfrac는 약 0.342466이다.

**정적 결과.** E96의 23,806 / 43명령에서 23,977 / 43명령으로 171 cycle 증가했다.

**실제 결과.** Job16439는 공개 12 process와 continuous 3조건, job16447은 공개 역순 8 process를 실행했다. 총 23 process에서 세 커널 정확성이 통과했다. 첫 job에는 E91 관측도 있지만 아래 8쌍 통계에는 E96과 E100만 사용했다.

| 같은 공개 비교 8쌍 | E96 | E100 | 변화 |
|---|---:|---:|---:|
| O 중앙값 | 49,294 | 50,053 | +1.54% |
| O 평균 | 49,859 | 50,033.75 | +0.35% |
| 후보가 빠른 쌍 | — | 5 / 8 | 중앙값·평균 개선 미입증 |

**중복 실험 방지.** 같은 F32 전체 HBM 전달과 같은 norm 경계를 반올림 한 번 제거한다는 이유만으로 반복하지 않는다. 늘어난 전송 비용을 줄이거나 전달 경계 자체를 없애는 변경, 혹은 새로운 trace에서 실제 병목 감소가 확인되는 경우에 재검토한다.

보존 자료: [projection 소스](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/active-checkout/target/experiments/exp100-f32-output-bridge/projection.rs), [norm 소스](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/active-checkout/target/experiments/exp100-f32-output-bridge/shared-rmsnorm.rs), [host 결과](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/active-checkout/target/experiments/exp100-f32-output-bridge/host-screen.json), [정적 스케줄](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/active-checkout/target/experiments/exp100-f32-output-bridge/sliding_attention_output.json), [컴파일 로그](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/active-checkout/target/experiments/exp100-f32-output-bridge/compile.log), [job16439 로그](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/active-checkout/target/experiments/exp100-f32-output-bridge/combined/job.log), [job16447 로그](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/active-checkout/target/experiments/exp100-f32-output-bridge/reverse/job.log), [8쌍 합산 결과](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/active-checkout/target/experiments/exp100-f32-output-bridge/reverse/summary.json), [실험 기록](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/active-checkout/target/experiments/exp100-f32-output-bridge/README.md).

## E101 — 입력 Div를 reciprocal과 Mul로 치환

**변경.** E96의 hi/lo 입력 정규화에 있는 Div 두 개를 Mul로 바꾸었다. 두 변형을 분리했다.

- E101a는 원래 `s`와 contraction 복원용 VRF를 유지한다. 별도 scalar Main에서 `1/s`를 계산하고 DM commit → Sub fetch로 두 번째 VRF를 만든다.
- E101b는 `min(256/maxabs,1/(1e-30/256))`만 저장한다. Hi/lo에는 inverse scalar를 곱하고 contraction 결과는 inverse scalar로 나눈다. 원래 clip 하한의 reciprocal을 새 clip 상한으로 사용한다.

**Host 결과.** 두 변형 모두 18조건을 통과했다. E87·E96 후처리 경계 각각 최대 절대 오차 0.03125, 최대 budgetfrac 0.440529다. 이 입력들에서는 hi/lo와 최종 BF16 값이 기준과 같았지만 모든 입력에 대한 bit-exact 증명은 아니다.

**정적 결과.**

| 경로 | E96 | E101a: 두 VRF | E101b: 한 VRF |
|---|---:|---:|---:|
| Makespan | 23,806 | 23,806 | 23,806 |
| 명령 수 | 43 | 45 | 43 |
| Terms TRF 준비 완료 | 3,939 | 4,221 | 3,939 |
| Hi/lo Main 각각 | 345 / 345 | 345 / 345 | 345 / 345 |

E101a는 scalar Main 282 cycle와 Sub 267 cycle가 추가되며 일부가 겹친다. Terms 준비는 282 cycle 늦어지지만 weight DMA 종료 15,319보다 일러 전체 종료 시각은 같다. E101b는 각 명령의 정적 시작·종료 시각까지 기준과 같다. Div를 Mul로 바꿔도 입력 Main 패스의 길이가 줄지 않았다.

**실제 결과.** 두 변형 모두 하드웨어를 실행하지 않았다. Job ID가 없다.

**중복 실험 방지.** 같은 Q256 입력 패스에서 연산 종류만 Div→Mul로 바꾸는 후보는 반복하지 않는다. 패스·DM/VRF 경계를 실제로 제거하거나 입력 준비가 weight DMA 이후의 임계 경로가 되는 새 배치가 있어야 재검토한다.

보존 자료: [E101a 소스](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-ffn-blockdot/o-e101-reciprocal/experiments/two-vrf/projection.rs), [E101b 소스](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-ffn-blockdot/o-e101-reciprocal/experiments/inverse-one-vrf/projection.rs), [정적 비교](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-ffn-blockdot/o-e101-reciprocal/artifacts/static-results.json), [E101a 스케줄](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-ffn-blockdot/o-e101-reciprocal/experiments/two-vrf/output.json), [E101b 스케줄](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-ffn-blockdot/o-e101-reciprocal/experiments/inverse-one-vrf/output.json), [E101a 컴파일 로그](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-ffn-blockdot/o-e101-reciprocal/experiments/two-vrf/compile.log), [E101b 컴파일 로그](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-ffn-blockdot/o-e101-reciprocal/experiments/inverse-one-vrf/compile.log), [host 결과](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-ffn-blockdot/o-e101-reciprocal/artifacts/host-screen.json), [상세 분석](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-ffn-blockdot/o-e101-reciprocal/docs/e101-analysis.md).

## E102 — 두 cluster에서 RMSNorm을 끝내고 scalar만 교환

**변경.** E95의 packed FP8 direct terms와 68+52행 타일을 유지한다. BF16 dot 결과를 두 cluster의 원래 row-shard DM에 두고 E84의 local RMS helper를 사용한다. 전체 H의 7,680-byte HBM 전달을 없애고 두 cluster의 f32 partial mean을 교환한다. 교환값은 유효 8 bytes, padding 포함 16 bytes다. 최종 norm·residual은 원래 분산 배치에서 계산한 뒤 gather하고 HBM에 쓴다.

E95 대비 channel scale 후 BF16과 별도 norm BF16 경계가 E70/E84 방식의 f32 융합으로 바뀌며 RMS 합산 트리도 달라진다. FP8 block scale·hi/lo·packed contraction 식은 E95와 같다.

**Host 결과.** E102a와 E95 대조군 모두 18조건을 통과했다. E102a의 최대 절대 오차는 0.03125, 최대 budgetfrac는 0.440529다.

**정적 결과.** E95 24,024에서 E102a 24,428 / 63명령으로 증가했다. 마지막 dot은 17,195에 끝나지만 이후 local mean, scalar Switch/reduce, scalar HBM 쓰기·읽기, global RMS, scalar Sub, norm·residual, 최종 gather가 이어진다. Scalar 쓰기 755 cycle, 읽기 660 cycle와 Switch용 내부 table DMA 719 cycle가 남는다. 전달 byte 감소가 latency 감소로 이어지지 않았다.

E102b는 마지막 gather를 없애고 분산 결과를 HBM에 직접 쓴다. 직접 쓰기가 2,040 cycle로 늘어 기존 gather 882 + 쓰기 458보다 700 cycle 느렸고, makespan은 25,128 / 62명령이다. E102b의 독립 하드웨어 검증은 없다.

**실제 결과.** Job16443에서 E95/E102a 공개 ABBA 4쌍과 continuous 3조건, 총 11 process가 세 커널 정확성을 통과했다. Continuous의 비갱신 K/V cache도 통과했다.

| 같은 공개 비교 4쌍 | E95 | E102a |
|---|---:|---:|
| O 원시값 | 42,839 / 51,113 / 47,305 / 48,769 | 58,419 / 56,327 / 50,219 / 58,353 |
| O 중앙값 | 48,037 | 57,340 |
| O 평균 | 47,506.5 | 55,829.5 |

네 쌍 모두 회귀했다. 중앙값 +19.37%이며 역순 추가 제출은 하지 않았다. 이 job의 E95 단일 최솟값을 E96 기반 후보나 continuous 입력의 최솟값과 합쳐 순위를 만들지 않는다.

**중복 실험 방지.** 같은 scalar HBM 교환·Switch·최종 gather를 그대로 유지한 채 전체 전달 byte만 줄었다는 이유로 재실험하지 않는다. Scalar 동기화나 교환 경계를 실제로 없애는 다른 경로가 있어야 한다. 분산 결과의 직접 HBM 쓰기도 연속 전송 배치가 달라지지 않으면 E102b를 반복하지 않는다.

보존 자료: [E102a projection 소스](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-qkv-prefetch/experiments/e102-packed-two-cluster/candidate-src/device/sliding/projection.rs), [E102a RMS 소스](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-qkv-prefetch/experiments/e102-packed-two-cluster/candidate-src/device/shared/rmsnorm.rs), [E102b 소스](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-qkv-prefetch/experiments/e102-packed-two-cluster/candidate-direct-store-src/ops.rs), [host 결과](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-qkv-prefetch/experiments/e102-packed-two-cluster/host-screen.json), [E102a 스케줄](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-qkv-prefetch/experiments/e102-packed-two-cluster/output.json), [E102b 스케줄](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-qkv-prefetch/experiments/e102-packed-two-cluster/output-direct-store.json), [E102a 컴파일 로그](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-qkv-prefetch/experiments/e102-packed-two-cluster/compile.log), [E102b 컴파일 로그](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-qkv-prefetch/experiments/e102-packed-two-cluster/compile-direct-store.log), [job16443 로그](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-qkv-prefetch/experiments/e102-packed-two-cluster/job.log), [실제 결과](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-qkv-prefetch/experiments/e102-packed-two-cluster/results.json), [상세 근거](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-qkv-prefetch/experiments/e102-packed-two-cluster/notes.md).

## E104 — 두 cluster의 row32 × col8 geometry

**변경.** E96의 두 cluster를 유지하면서 row16 × col16, local H120/Q256을 row32 × col8, local H60/Q512로 바꾼다. Weight는 single60으로 읽고 packed native FP8 및 fused norm을 사용한다. 이는 한 cluster였던 E81과 다른 배치이며, BF16 contraction을 사용했던 E57과도 구분한다.

입력은 TDMA로 실제 복제한다. Weight와 결과의 `15 group × 4 row = H60` reshape는 같은 주소 순서만 재해석한다. Q block이 256→512로 커져 동적 maxabs scale 및 FP8 rounding이 달라질 수 있고 slice reduction도 16→8로 바뀐다. 수치적으로 bit-exact라고 주장하지 않는다.

**Host 결과.** Q256 기준과 Q512 후보 모두 18조건을 통과했다. 후보의 최대 절대 오차 0.03125, 최대 budgetfrac 0.440529다. Q256 기준 대비 최종 BF16 차이는 최대 0.015625였다.

**정적 결과.** E96 23,806에서 24,267 / 43명령으로 461 cycle, 1.94% 증가했다. Packed native Main은 2,337→2,265로 줄었지만 입력 broadcast DMA 933→1,318, weight DMA 13,383→13,512, gather 882→898로 늘었다.

**실제 결과.** 공개 job16453의 8 process와 continuous job16454의 3조건에서 세 커널 정확성을 모두 통과했다. Continuous의 비갱신 K/V cache도 통과했다.

| 같은 공개 비교 4쌍 | E96 | E104 |
|---|---:|---:|
| O 원시값 | 49,805 / 48,519 / 45,069 / 50,767 | 49,144 / 46,509 / 47,320 / 51,536 |
| O 중앙값 | 49,162 | 48,232 |
| O 평균 | 48,540 | 48,627.25 |

중앙값은 1.89% 줄었지만 평균은 0.18% 늘었고 2/4쌍만 개선됐다. 지속 개선은 입증되지 않았다. Continuous의 최대 budgetfrac는 amplitude 1 / 0.001 / 16에서 각각 0.330033 / 0.249377 / 0.224467이며, 해당 cycle은 공개 성능 통계에 합치지 않는다.

**중복 실험 방지.** 같은 두 cluster H60/Q512 single60 배치는 새 근거 없이 재제출하지 않는다. 입력 broadcast나 weight 전송의 증가를 줄이는 변경, 또는 새로운 trace가 특정 배치 경합을 식별한 경우에 다시 검토한다. E57과 E81을 이름만 바꾼 재실험으로 오인하지 않도록 native dtype과 cluster 수를 함께 기록한다.

보존 자료: [projection 소스](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-ffn-blockdot/o-e104-geometry512/experiments/single60/source/device/sliding/projection.rs), [O 호출부 소스](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-ffn-blockdot/o-e104-geometry512/experiments/single60/source/ops.rs), [매핑·ABI 감사](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-ffn-blockdot/o-e104-geometry512/artifacts/source-audit.json), [host 결과](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-ffn-blockdot/o-e104-geometry512/artifacts/host-screen.json), [정적 비교](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-ffn-blockdot/o-e104-geometry512/artifacts/static-comparison.json), [스케줄](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-ffn-blockdot/o-e104-geometry512/experiments/single60/output.json), [컴파일 로그](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-ffn-blockdot/o-e104-geometry512/experiments/single60/compile.log), [job16453 로그](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-ffn-blockdot/o-e104-geometry512/validation/public/job.log), [공개 결과](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-ffn-blockdot/o-e104-geometry512/validation/public/results.json), [job16454 로그](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-ffn-blockdot/o-e104-geometry512/validation/continuous/job.log), [연속입력 결과](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/agent-ffn-blockdot/o-e104-geometry512/validation/continuous/results.json).

## E106 — native output의 8행 그룹

**변경.** E96의 `30 group × 4 row`를 `15 group × 8 row`로 바꾸되 BF16 transpose는 `Block/4`를 Time에 두고 여전히 4행씩 저장한다. 기존 helper와 QKV/FFN은 유지한다.

**Host 결과.** 별도 host 수치 검사 결과는 없다.

**정적 결과.** SDK 0.6.0 컴파일에 성공했지만 23,806 / 43명령으로 E96과 같다.

**실제 결과.** 정적 이득이 없어 하드웨어를 실행하지 않았다. Job ID가 없다.

**중복 실험 방지.** 그룹 축 이름·크기만 바꾸고 실제 4행 저장, contraction, 전송 경로가 그대로인 후보는 반복하지 않는다. Flit당 유효 출력 수나 실제 명령 경계가 바뀐다는 근거가 있어야 재검토한다.

보존 자료: [projection 소스](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/active-checkout/target/experiments/exp106-output-eight-row-group/projection.rs), [O 호출부](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/active-checkout/target/experiments/exp106-output-eight-row-group/ops.rs), [정적 스케줄](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/active-checkout/target/experiments/exp106-output-eight-row-group/sliding_attention_output.json), [컴파일 로그](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/active-checkout/target/experiments/exp106-output-eight-row-group/compile.log), [실험 기록](/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/target/optimization/20260910-145944-baseline.t8OKVA/active-checkout/target/experiments/exp106-output-eight-row-group/README.md).

## 보존·재검토 시 주의할 범위

위 절대 링크는 현재 작업 공간의 동결한 source, schedule, host 결과와 job 로그를 가리킨다. `target/`은 Git에서 제외되는 산출물 경로이므로 정리 전에 필요한 source·manifest·로그를 별도로 보존한다. 새 후보를 만들 때에는 해당 후보의 기준 소스, SDK 버전, 공개 fixture SHA와 실제 연산 차이를 먼저 확인한다. 이 문서의 보류 판단은 현재 구현과 관측 범위에 대한 것이며, 지원 dtype·실제 데이터 경로·병목 증거가 달라진 새 가설까지 금지하는 것은 아니다.
