# Attention output: 스케줄과 실측으로 확인한 진행 상황

2026-09-11. SDK0.6.0, Rust nightly-2026-05-01. 원본 `src`의 공식 최고는 **E172 / 339089eb**, QKV/O/FFN **100,571 / 38,873 / 284,644 cycle**, 점수 **6.9613**, 세 커널 PASS다. 동일 소스의 공식3회 결과와 반복 중앙값을 [현재 최고 기록](current-best.md)에 모두 보존했다.

[최약 커널 우선순위](optimization-priority.md)는 다른 팀의 해당 커널 최저 기록과 비교한다. 현재 FFN **−1.52%**, QKV **−2.50%**, O **−9.70%** 순서로 FFN의 우위가 가장 작다. E175 FFN은 실제8쌍에서 E165 대비 중앙값 −1.03%를 보여 공식3회 큐에 넣었다. 이 FFN과 최고 O172를 결합한 E176도 실제 검증한다.

최고 소스와 백업은 [current-best.json](current-best.json), 병합 후 적용은 [E172 패치](patches/e172/README.md)를 따른다. E172는 E161과의 Arena8쌍에서 O 중앙값 −4.35%, 6/8쌍 개선이었다. E172의 공식3회 O는 38,873 / 44,095 / 41,413 cycle로, 단일 최고와 반복 변동을 구분한다. 아래는 이전 실험부터 이어지는 근거 기록이다.

## 실제로 확인한 스케줄

`furiosa-schedule-viewer`를 localhost:9254에서 실행하고 아래 JSON을 실제 UI에 로드했다. 캡처는 생성한 도식이 아니라 viewer 화면이다.

- [E75 화면](schedule-viewer-e75-output-tail.png): hi/lo contraction과 partial VRF preload가 분리된 이전 후보.
- [E87 화면](schedule-viewer-e87-output-tail.png): 두 FP8 항을 단일 contraction에서 합산.
- [E92 화면](schedule-viewer-e92-output-tail.png): lo 항을 실제 terms DM 슬롯에 직접 써서 복사 하나 제거.
- [E107 화면](schedule-viewer-e107-output-tail.png): 첫104행 dot과 다음16행 weight DMA의 겹침.

JSON과 화면 재현 스크립트는 `target/optimization/20260910-145944-baseline.t8OKVA/active-checkout/target/experiments/` 및 `target/viewer-debug/inspect_packed_viewer.py`에 있다. JSON의 `max(instructions[*].lifetime.end)`는 정적 makespan이고, 실제 RNGD cycle과 별개다.

## 개선이 바로 공식 기록으로 이어지지 않았던 이유

1. **FP8 연산 자체의 절감을 중간 저장과 preload 대기가 상쇄했다.** E75의 정적 스케줄은 짧았지만, trace에서 둘째 타일 hi→partial DM→Sub→lo 경로가 반복해서 길어졌다. E87 packed contraction은 이 경로를 제거했다. 마지막 weight DMA 완료→projection HBM 쓰기 완료의 네 번 중앙값이 **15,535→6,331 cycle**로 줄었다. 정적 makespan은 오히려 24,916→25,071로 늘었으므로 정적 최종 숫자만으로는 발견할 수 없던 개선이다.
2. **작은 DM 복사가 weight DMA와 입력 준비의 의존 경로에 끼어 있었다.** E92는 lo 복사, E95는 hi/lo 복사를 모두 제거했다. 두 항을 실제 DM 슬롯에 commit하고 term0 읽기→term1 쓰기→TRF 준비 순서를 컴파일된 스케줄에서 확인했다. reshape로 가짜 broadcast를 만들지 않았다.
3. **Projection 뒤 cluster 전환과 norm 입력 준비의 변동이 크다.** 긴 `Cluster` span 전체를 idle로 해석하면 안 된다. E87의 추가 span은 residual preload와 norm 계산에도 겹친다. trace가 제공하는 정보에는 stall 원인이나 정확한 engine ID가 없어 bank 충돌을 확정하지 않았다.
4. **같은 소스의 공식 결과도 변한다.** E87 두 번 O는 52,179 / 46,108, E91 두 번은 44,212 / 47,805다. 단일 최저값과 반복 중앙값을 구분하고, 공개 fixture 한 번 통과만으로 수치 변경을 채택하지 않는다.

상세 runtime 분석: [E93 보고서](e93-packed-fp8-runtime.md). E99는 E96에서 E75의 긴 preload 정체가 재발하지 않았음을 확인했다. E96의 full120 contraction 약 5,420 cycle은 마지막 weight DMA 뒤에 노출된다. E103은 direct terms를 유지하면서 68+52 두 타일과 fused norm을 결합해 이 부분을 줄이는 후속 실험이다.

## Runtime trace의 cluster 관측 범위

2026-09-10 설치 SDK 0.6의 header·FFI·vendor runtime binary를 추가 감사했다. 현재 **1chip/2cluster 구성의 callback은 첫 logical cluster의 profile buffer를 읽는 경로**로 판단한다. 이는 공개 문서의 보장 문구가 아니라 설치 binary의 readback dataflow를 추적한 결론이다. 두 cluster의 실행 시간을 min/max로 합친 span이 아니다.

두 cluster ARM 이미지 모두 profile 명령을 포함하지만, launch는 `ChipBuffer::offset_per_cluster`로 서로 다른 profile 주소를 주고 `Profiled::run`은 cluster 하나 분량의 `chunk_bytes`만 host에 읽는다. `same_chip_scatter`는 chip마다 offset0의 한 구간을 선택하고, `Symbols::decode`는 그 buffer의 UID별 cycle을 그대로 조회한다. C ABI에는 cluster ID나 engine ID가 없으며 `begin`/`end`는 TUC cycle이다. host callback 소요 시간을 이 cycle 구간의 원인으로 해석할 근거는 없다.

따라서 E105의 bridge `Cluster` 663~18,229 cycle 동안 다른 callback 명령이 겹치지 않았다는 사실은 **칩 전체가 idle이었다는 증거가 아니다**. peer cluster의 DMA·계산이 관측되지 않을 수 있다. embedded PE runtime에서는 peer IPC timestamp가 도달할 때까지 TUC command batch tail 공급을 막는 경로가 확인됐다. peer 작업 완료 지연, PE interrupt/polling, IPC 전달의 기여도는 현재 counter로 구분할 수 없고, 부하 불균형을 확정하지 않는다.

전체 근거와 역어셈블 오프셋·hash는 [Cluster span 감사](../target/optimization/20260910-145944-baseline.t8OKVA/agent-output-fused/experiments/cluster-span-audit/README.md)에 보존했다. 이 관측 한계를 바탕으로 E110은 E107의 H 행을 960행 chunk 단위로 두 cluster에 교대로 배정한다. 산술·타일을 유지하고 실제 weight DMA와 output gather mapping을 바꾸는 실험이며, 정적 22,771→23,077(+306)은 마지막 strided HBM write 증가다. Arena job16479/16481에서 E107 대비 공개8쌍 중7쌍개선, O 중앙값49,932→47,684.5(-4.50%),평균50,448.875→46,526.625(-7.77%)였다. 공개16+연속3 총19process 모두3커널PASS이며 연속3case의cache비갱신영역도bit-exact였다. 최저41,529는40,000목표보다높고, 개선의원인이peer부하불균형임을확정하지않는다. [E110 상세결과와동결소스](../target/optimization/20260910-145944-baseline.t8OKVA/agent-output-fused/experiments/output-e110-h960-interleaved/README.md)를공식평가후보로보존했다.

## 반복 비교 결과

아래 수치는 같은 Arena job에서 실행 순서를 바꿔 비교한 공개 입력 결과다. 서로 다른 행의 후보 최저값만으로 우열을 판단하지 않는다.

| 후보 | 기준 | O 중앙값 기준→후보 | 개선 쌍 | 검증 job |
|---|---|---:|---:|---|
| E87 packed contraction | E75 | 54,125→51,995 | 3/4 | 16410 |
| E91 single120 split FP8 + fused norm | E50 | 56,646→52,836 | 7/8 | 16417/16419 |
| E92 lo direct commit | E50 | 54,511→50,278 | 3/4 | 16423 |
| E94 E92 + fused norm | E50 | 54,879→48,917 | 4/4 | 16430 |
| E94 E92 + fused norm | E92 | 51,216→48,917 | 3/4 | 16430 |
| E95 hi/lo direct commit | E87 | 50,013.5→48,149 | 5/8 | 16427/16431 |
| E96 single120 packed + fused norm | E50 | 55,940→48,724 | 8/8 | 16426/16428 |
| E103 packed68+52 + fused norm | E96 | 47,938→46,101 | 5/8 | 16444/16452 |
| E107 packed104+16 | E96 | 48,096→44,810.5 | 3/4 | 16459 |
| E108 packed100+20 | E107 | 49,286→46,172 | 6/8 | 16471/16474 |
| E110 H960 교대 cluster | E107 | 49,932→47,684.5 | 7/8 | 16479/16481 |
| E119 H480 교대 cluster | E110 | 49,842→47,409.5 | 5/8 | 16489/16492 |
| E128 residual 직접 읽기 | E110 | 48,104.5→49,708.5 | 1/4 | 16508 |
| E134 H240 교대 cluster | E119 | 48,355→47,837 | 4/8 | 16521/16522 |

모두 공개 입력과 기존 승인된 연속 BF16 입력 3case에서 세 커널 정확성을 통과했다. K/V cache의 비갱신 영역 보존도 검증했다. E92 공개 단일 최소 41,965는 현재 1등 O 43,049보다 작지만 반복 중앙값 50,278과 분리해서 기록하며 안정적인 목표 달성으로 선언하지 않는다.

## 이후에도 유지할 절차

1. 별도 checkout에서 후보 소스와 hash를 동결한다. 원본 `src`와 Git index는 실험에 사용하지 않는다.
2. `CARGO_BUILD_JOBS=12 ./scripts/furiosa.sh compile ops::sliding_attention_output --exact --dump-schedule <실험별 JSON>`으로 컴파일하고 viewer에서 engine/메모리 lifetime을 확인한다.
3. 실제 trace에서 DMA 완료, Main/Sub 실행, projection 쓰기와 norm 읽기 경계를 추적한다. trace 계측 cycle과 공개 info cycle을 섞지 않는다.
4. 수치 변경은 host 여러 입력 규모 검사 후 공개+연속 3case에서 세 커널 전체를 검증한다. E98 hi-only FP8은 18case 중 12case가 실패해 기각했다. 검증 기준은 수정하지 않는다.
5. 원본 또는 현재 유망 후보와 같은 job에서 교대 비교한다. 유망하면 역순으로 재현한다. binary/source/fixture hash와 숫자 job ID를 저장한다.
6. 통과한 후보의 `src/ops.rs` 및 `src/device/` 23파일을 동결해 `moa-submitter submit --source <동결 경로>`로 제출한다. 사용자가 허용한 세 회 평가를 순차 실행하고 그동안 다음 실험을 진행한다.
7. 공식 최고를 갱신하면 원본 변경 감시 hash와 index hash를 확인하고 백업 후 반영한다. 적용 패치는 임시 기준 소스에서 실제 적용 및 결과 hash까지 검증한다.

제출 큐는 `target/moa-submissions/20260910-output-frozen-queue.json`과 `.state.json`에 있다. 실행 중에는 다른 프로세스에서 동일 후보를 중복 제출하지 않는다. 큐 실행기는 접수 ID를 못 받으면 자동 재제출하지 않고 중단한다.

리더보드 원본: [공식 페이지](https://micro2026-moa.github.io/leaderboard.html). 확인 시각과 공개 API 응답은 [leaderboard-snapshot.json](leaderboard-snapshot.json)에 보존한다. 현재 알려진 1등 기록을 소스에 하드코딩하지 않고 목표 기록으로만 사용한다.


## 2026-09-10 후속 분석

[E117 실제 타일 겹침](../target/optimization/20260910-145944-baseline.t8OKVA/active-checkout/target/experiments/exp117-e107-e108-runtime/README.md)에서 104+16은 첫 dot 4,835와 둘째 DMA 4,855 cycle이 거의 맞고, 100+20은 둘째 DMA 5,702로 늘어 기다림이 생겼다. 따라서 더 작은 첫 타일을 계속 탐색할 근거는 없다. info 반복의 E108 우세와 타일 자체의 구조적 효과는 구분한다.

E110 공식 2회는 O 45,374 / 45,122 cycle, 점수 6.4426 / 6.4230이다. 전체 최고 점수 기준으로 E110을 원본에 채택했으며, E91의 O 단일 최고 44,212와 다른 기준이다. QKV/FFN 구현은 같고 cycle 차이는 실행 변동을 포함한다.

[E121 DM 직접 전송](../target/optimization/20260910-145944-baseline.t8OKVA/active-checkout/target/experiments/exp121-direct-dm-output-bridge/README.md)은 SDK의 cluster 동기화 검사에서 실패해 원격 시험하지 않았다. QKV E118과 FFN E116은 별도 후보로 진행하며 원본 최고는 검증 뒤에만 갱신한다.


## 현재 진행 중인 개선

E138의 부분 제곱합 교환은 실제 4쌍에서 O 중앙값 +2.82%로 기각했다. E145 trace에서 전체 결과의 HBM 왕복을 줄여도 cluster 대기가 남고, 교환 전 제곱합·gather·compact가 평균 약3,816 cycle 추가됨을 확인했다. 이 비용과 관측 범위의 한계를 [실제 runtime 보고서](../target/optimization/20260910-145944-baseline.t8OKVA/agent-ffn-blockdot/output-e145-runtime/README.md)에 남겼다.

E149는 빠른 E119 projection을 유지하고 norm을16slice로 분할해 VRF 부담을 절반으로 줄이며 residual scatter를 없앴다. 공개8쌍 O 중앙값51,286.5→48,742.5(−4.96%),7/8승, 연속3조건까지 총19process 모든커널PASS다. 검증job16596/16603 이후 공식최고 f5936e50로 확인돼 원본에 E149를 반영했다. [실제 Viewer 화면](schedule-viewer-e149-output-tail.png)과 실험별 JSON/로그를 보존한다.

E147의 Main widening 분리는 실제 우세가 없어 기각했다. 후속으로 FP8 두 항을 유지하면서 입력 준비 의존성을 줄이는 방식, norm의 실제32slice 분할, residual을 Main의 두 입력으로 읽는 방식을 독립 실험한다. 모든 후보는 기존 Stage1 연산과 허용오차를 검증한 뒤 판단한다.

최근비교그림: [O 반복 비교](recent-output-comparison.png), [QKV 반복 비교](recent-qkv-comparison.png), [공식 제출 추이](moa-submission-progress.png), [실제 runtime timeline](e117-runtime-timeline.html).


## 2026-09-11 후속 검증

E150의 새 FP8 2항 분해는 source/host22와 총19process3커널/cache 정확성을 통과했다. 원본 입력을 미리 VRF에 읽어 높은항 commit→재읽기 의존성을 없앴으나 실제8쌍 중앙값+1.10%,평균−1.57%로 안정적 우위가 없어 미채택했다. E153의 실제 두DM 입력으로 residual Sub를 제거하는 방식도4쌍 중앙값+2.42%로 기각했다. E154의 norm8을 다른 물리 slice에 분산하는 경로는 VRU 축 제약상 scalar이동이 필요해 정적+20.59%로 기각했다.

E152의 channel_scale×RMSweight 사전 F32곱은 모든 수치 검사를 통과했지만 E131대비8쌍 중앙값+4.19%,현재E149대비4쌍+7.77%(0/4승)로 기각했다. 첫4쌍에서 보인−3.41%가 역순에 재현되지 않았다. E156은 같은변경을 norm16에 넣은 별도상호작용 후보로 검증하며, E157은 H60×Q512/52+8 타일·교대cluster 배치를 컴파일 중이다.

E155 실제trace의 안정적인 E149이득은 norm입력read완료→최종HBMwrite **평균509.5cycle감소**다. mean1097→741.5, RMS517→659.5, finalMain979→677.5이며 큰cluster대기·residualSubspan은남는다. 모든관측은SDKcallback의첫logicalcluster범위이며 전체chip idle을의미하지않는다.


2026-09-11 갱신: E156 공식 두 결과가 E149보다 낮아 미채택했다. job16674의 안정적인 tail은+25.5cycle이고 긴 대기는 새 계수 Sub로 이동했다. 최약커널은 여전히 O(+4.69%)다. E161 scalar inverse RMS 수치/실측, E162 실제 HBM bridge 대체를 병행한다.


E161 두번째909550aa는사용자가동일소스라고확인했다. 공식121633/45543/287376,score6.1782,세커널PASS라더좋은첫367b33c0를유지한다. 자동submit02.log는별도제출충돌기록으로보존하고status02/server02는사용자확인반복결과로저장했다. 추가E161제출없음. E159는queue11첫43caa7a7접수,이어2회반복. E151 norm32는16716/16722 총19process3PASS/cache6PASS,8쌍5승median48707→46279(−4.98%)라queue12에서E159완료후동일2회공식검증한다. 원본src최고는E161이며실험source를자동합쳐바꾸지않는다. 커널별경쟁비율은O+1.90%,FFN+1.07%,QKV−5.27%로여전히O우선이다.


## E172와 FFN 후속의 실제 viewer

[E172 O 후처리](schedule-viewer-e172-tail.png), [E173 FFN tail16](schedule-viewer-e173-tail.png), [E175 FFN direct store](schedule-viewer-e175-tail.png)는 실제compiledJSON을Schedule Viewer에열어저장한화면이다. E172의O8쌍중앙값−4.35%, E173의FFN8쌍중앙값+0.22%를공식단일최저와구분한다. 세커널현재최고대비우위가가장작은FFN을우선하며E175실제측정을진행한다. 자세한hash/실행명령/전후값은 [최적화기록](optimization-log.md)에보존한다.
