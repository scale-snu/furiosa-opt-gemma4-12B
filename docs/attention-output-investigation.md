# Attention output 집중 분석

2026-09-10 최신 사용자 목표는 **QKV 103697 / O 43049 / FFN 287992**다. 현재 원본 `src`는 공식 최고 점수 E50이며 MOA a69e3944에서 **106265 / 52192 / 287992**, 점수 **6.1713**을 기록했다. 공식 O 측정에서 목표까지 약 17.5% 감소가 더 필요하다.

## 최근 실제 RNGD 비교

아래 중앙값은 같은 Arena job 안에서 기준/후보를 AB/BA로 교대 실행한 4쌍의 O cycle이다. 각 행의 기준이 다르며, 서로 다른 job의 숫자를 직접 순위로 합치지 않는다. 정적 makespan과 실제 cycle은 별개다.

| 후보 | 비교 기준 | 정적 makespan | 기준 → 후보 O 중앙값 | 정확성 / 판정 |
|---|---|---:|---:|---|
| E57 row32×col8 | E41 | 26409 | 55049 → 62810 | 공개 및 연속 3입력 PASS, 회귀 |
| E59 tile60+60 | E50 | 27012 | 64939 → 53837 | 4/4 개선, 연속 PASS; 공식 두 번은 E50 점수 미달 |
| E60 tile64+56 | E59 | 26851 | 59003 → 52061.5 | 4/4 개선, 연속 PASS; 공식 두 번은 E50 점수 미달 |
| E61 tile48+72 | E59 | 27495 | 56726 → 57451 | PASS, 개선 근거 없음 |
| E62 tile56+64 | E60 | 27173 | 58950 → 59382 | PASS, 개선 근거 없음 |
| E63 tile40+40+40 | E60 | 27746 | 55567 → 59940 | PASS, 4/4 회귀 |
| E64 tile80+40 | E60 | 26358 | 55026 → 55462 | PASS, 중앙값+0.79%/평균−0.30%로 혼합 |
| E65 tile68+52 | E60 | 26691 | 56596 → 51891 | 3/4 개선, 연속 PASS, 공식 제출 중 |
| E66 residual HBM 재사용 | E65 | 26704 | 55561 → 54646 | PASS, 2/4 개선이나 평균+477cycle, 보류 |
| E67 residual direct8 DM | E60 | 26850 | 합산8쌍 58308 → 56776 | PASS, 첫 job 개선/역순 job 회귀, 보류 |
| E68 weight DM 예약 padding | E60 | 25732 | 54778.5 → 55028 | PASS, 2/4 개선으로 우위 없음 |
| E69 norm→residual BF16 제거 | E65 | 26350 | 55752.5 → 55482 | 3/4 개선, 연속 PASS, 효과 작음 |

관련 job은 순서대로 16310/16322, 16290/16308, 16321/16326, 16323, 16330, 16327, 16345, 16339/16346, 16348, 16354/16357, 16359, 16363이다. 원본 로그와 source/binary SHA는 아래 별도 작업본의 실험 디렉터리에 보존한다.

## 런타임 trace로 확인한 병목

SDK 0.6의 profile callback을 별도 진단 harness에서 기록했다. 공개 테스트의 입력/참조식/정확도 기준과 커널은 바꾸지 않았다. `trace`의 상세 span은 정적 명령 ID와 일대일로 연결할 이름이 없어 시간·순서·의존 관계를 함께 보고 해석한다.

job16328에서 E41 O는 65646, E59 O는 61473 cycle이었다. E59의 projection 두 Main 합과 weight DMA는 오히려 더 느렸고, projection HBM write 이후 norm 입력 read 전의 긴 Cluster 구간이 19191→14341로 줄었다. E59 구간에는 겹치는 DMA/TuExec span이 없었다. 같은 E41도 이전 trace에서 이 구간이 10109였으므로 변동이 크다. Cluster span 길이를 엔진의 busy 시간이나 모두 제거 가능한 비용으로 해석하지 않는다.

E67은 residual DM scatter 한 개를 실제로 제거했다. job16361 trace에서 DMA span 수 13→12를 확인했지만, 해당 HBM 경계의 빈 Cluster 구간은 E60 8013/14885, E67 10567/11567로 안정적 감소를 입증하지 못했다. helper 호출 순서만 옮긴 control은 정적 스케줄이 완전히 같았다.

DM `physical_pages`와 저장 주소도 비교했다. E59는 두 weight가 서로 다른 페이지에, E65는 같은 페이지에 배치됐지만 E65가 실측에서 더 빨랐다. 페이지만으로 bank 충돌이나 성능 원인을 확정할 수 없다. E68은 DMA/연산 byte 수와 정적 makespan을 유지하면서 DM 예약 padding으로 페이지를 분리했으나 E60 대비 개선은 입증되지 않았다.

상세 trace: `target/optimization/20260910-145944-baseline.t8OKVA/agent-qkv-prefetch/docs/runtime-profile-audit.md`, 해당 작업본의 `experiments/e67-residual-preload/`.

## 오차 여유를 활용하는 다음 실험

사용자가 오차 margin을 고려한 공격적·다양한 실험을 요청했다. 허용 오차와 참조식은 고정하고, 근사하는 항과 제거하는 반올림을 후보별로 구분한다. 최대 절대 오차뿐 아니라 `abs(error)/(atol+rtol*abs(expected))`의 최댓값도 확인한다.

- O: norm→residual 중간 BF16 제거(E69), channelScale→norm까지 융합(E70), scalar reciprocal로 원소별 division 줄이기(E76/E77).
- FFN: E50에 upScale→GeGLU 융합 조합(E71), postnorm→residual BF16 제거(E72).
- QKV: scalar reciprocal과 곱셈 사용(E73), head RMSNorm과 RoPE 융합(E74).
- O native FP8: 블록 scale과 높은 항/보정항을 써서 `x ≈ scale*(hi+lo/16)`로 계산(E75). 두 FP8 contraction과 보정 비용을 스케줄 및 실측으로 비교한다. host 사전 검사 18경우는 PASS지만 아직 RNGD 통과나 성능 개선을 의미하지 않는다.

## 수치 검증 범위

공개 O 입력과 과거 추가 3seed의 입력은 ±1이라 FP8에서 정확히 표현된다. 이 조건만으로 FP8 activation 근사를 채택하지 않는다. 단순 FP8 후보 E55는 연속 BF16 [-1,1]의 3seed에서 18/20/30원소가 허용 오차를 넘었고, 작은 입력은 대규모 underflow를 일으켜 기각했다.

추가 harness는 O x를 연속 BF16로 생성하며 amplitude 1/.001/16, position 0/1024/32767에서 검증한다. 나머지 커널 참조값과 K/V 비갱신 영역 검사를 유지한다. E41 baseline job16275 및 E59/E60/E65가 모두 PASS했다. 이는 3가지 입력 범위의 실측 근거이며 모든 가능한 입력에 대한 증명은 아니다. 수치 근사 후보는 host 사전 검사와 추가 입력 검사를 함께 사용한다.

새 연속 fixture 3개는 각 67552 bytes다. E57 전송 당시 자동 승인 검토가 이전 승인 범위에 포함되지 않는다고 두 차례 거절했고, 사용자가 해당 바이너리·스크립트·fixture 전송을 명시적으로 승인했다. 이후 job16322가 정상 접수·PASS했으며 같은 fixture를 후속 후보에 재사용한다.

## 재현과 보존

실험은 `target/optimization/20260910-145944-baseline.t8OKVA/` 아래 `active-checkout`, `agent-output-fused`, `agent-ffn-blockdot`, `agent-qkv-prefetch`에서 진행한다. 원본 `tests/`와 `ref/`를 수정하지 않고, 최고 검증 소스는 원본 `src`에 유지한다. 테스트 전용 harness는 제출하지 않는다.

```bash
CARGO_BUILD_JOBS=12 ./scripts/furiosa.sh compile ops::sliding_attention_output --exact --dump-schedule <새 JSON 경로>
furiosa-schedule-viewer --host 127.0.0.1 --port 9254
```

SDK 0.6.0과 고정 nightly를 사용한다. [Viewer 분석 절차](schedule-debugging.md), [E50 정적 분석](schedule-audit-e50.md), [현재 최고 소스](current-best.json), [공식 제출 추이](moa-submission-progress.png), [재적용 패치](patches/e50/README.md)를 함께 확인한다. 정적 JSON과 실제 로그는 실행별로 보존하며 `target/`은 Git 제외 경로다.
