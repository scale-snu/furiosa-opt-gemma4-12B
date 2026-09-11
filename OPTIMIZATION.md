# Stage 1 Kernel Optimization Guide

This guide contains the original kernel-optimization workflow for Stage 1 of the
competition. It explains how to inspect a compiled schedule, find a bottleneck, change a
kernel, and compare results. Stage 1 grading is performed with
[`tests/test_kernels.rs`](tests/test_kernels.rs); the schedule is a development aid, not
the grading authority.

## 1. Dump a schedule

```sh
mkdir -p target/schedules
./scripts/furiosa.sh compile ops::sliding_project_qkv --exact \
    --dump-schedule target/schedules/sliding_project_qkv.json
```

The `--dump-*` flags are single-kernel options: one invocation writes one file, so dump
the three kernels one at a time.

`--exact` is important because the positional filter is a substring match by default.
Some kernel names are prefixes of others. For example,
`ops::sliding_attention` also matches `ops::sliding_attention_output` unless the filter is
exact.

The other dump flags are described in
[Kernel Optimizer](https://developer.furiosa.ai/furiosa-opt/book/tools/kernel-optimizer.html):
`--dump-visa`, `--dump-ir`, `--dump-dfg`, `--dump-graph`, and `--dump-summary` (which takes
a directory rather than a file).

## 2. Open the schedule viewer

```sh
furiosa-schedule-viewer
```

The viewer binds to `127.0.0.1:9254` and opens in a browser by default. Drag the schedule
JSON onto the drop zone, or click the drop zone to select it. `--host` and `--port` change
the bind address.

Use the viewer to inspect:

- **Nodes:** name, lifetime, context, connected nodes, and source location;
- **Cycle range:** a time window to isolate a suspicious region;
- **Brush:** a memory-address range instead of a time range.

The viewer shows the static execution plan. It does not show values or official RNGD
performance, and its instructions do not always correspond one-to-one with source-level
operators.

Reference: [Schedule Viewer](https://developer.furiosa.ai/furiosa-opt/book/tools/schedule-viewer.html).

## 3. Find the bottleneck

Start with the schedule's overall span, then trace the longest-lived nodes to their inputs,
outputs, contexts, and source lines. Gaps usually indicate dependency waits. A context that
stays busy while others are idle is often the limiter for that interval.

The main contexts are:

- **`DmaEngine`:** movement of weights and activations is limiting;
- **`SubContext`:** register-file staging and preloads are limiting;
- **`MainContext` / `VectorEngine`:** vector computation such as softmax, RMSNorm, or
  casts is limiting.

`MainContext` and `SubContext` contend for the Tensor Unit pipeline. Overlap is therefore
bounded: total time can approach the sum of their work, while ideal overlap approaches the
larger contributor.

On DMA nodes, inspect `util` in addition to duration. Low utilization can indicate an
awkward access pattern rather than an unavoidable bandwidth limit. Partition-crossing,
strided, and non-contiguous accesses are common causes. Alignment and HBM bank conflicts
can also be expensive; see the
[Memory Performance](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/memory-performance.html)
and [Schedule](https://developer.furiosa.ai/furiosa-opt/book/scheduling/schedule.html)
chapters for details.

The scheduler splits the TRF into halves automatically. A tensor that fits in half the file
can share the other half with another operation, and both halves share banks.

### Example: `ops::sliding_project_qkv`

```sh
./scripts/furiosa.sh compile ops::sliding_project_qkv --exact \
    --dump-schedule target/schedules/sliding_project_qkv.json
```

In one earlier revision, the schedule spanned roughly 116,600 cycles. `MainContext` was
busy for about 96% of the span, making compute the larger lever. The longest node was a
`MainContext` fetch-and-switch of roughly 62,000 cycles at:

```
--> src/device/layout.rs:16
```

That source location was a broadcast helper, not the projection itself. Three projection
weight loads were also visible on `DmaEngine`, but they overlapped with the longer compute
interval. Compare contributors before optimizing an individual node.

These figures are historical examples and will change as the skeleton changes. The
diagnostic method is the reusable part.

### Inspect the raw schedule JSON

The schedule JSON is plain and scriptable. Each entry in `instructions` includes `tpe`,
`contexts`, `lifetime` (`begin`/`end`), `util` (with `total_util`), and a `description`
containing the source location and expression. Each entry in `tensors` includes
`buffer_type`, `address`, `size`, and `shape`. Makespan is
`max(instruction.lifetime.end)`.

Further reading: [Diagnosis](https://developer.furiosa.ai/furiosa-opt/book/scheduling/diagnosis.html).

## 4. Change the kernel

Choose the optimization lever that matches the bottleneck:

1. **Execution-engine path:** use this when the current resource is the bottleneck.
2. **Tile or split shape:** use this for avoidable serial work or an ill-fitting reduction.
3. **Mapping, padding, or transfer boundary:** use this when movement or an address
   dependency limits the interval.

The relevant API is documented in the
[`prelude`](https://docs.rs/furiosa-opt-std/latest/furiosa_opt_std/prelude/index.html),
including [`m!`](https://docs.rs/furiosa-opt-std/latest/furiosa_opt_std/prelude/macro.m.html),
[`Tensor`](https://docs.rs/furiosa-opt-std/latest/furiosa_opt_std/prelude/struct.Tensor.html),
the memory-tier aliases, and the
[`contraction`](https://docs.rs/furiosa-opt-std/latest/furiosa_opt_std/prelude/contraction/index.html)
and [`vector`](https://docs.rs/furiosa-opt-std/latest/furiosa_opt_std/prelude/vector/index.html)
modules.

## 5. Compare the result

Re-dump the kernel and compare its makespan with the previous schedule. Keep the old JSON
files so that improvements remain reproducible. Also record which context dominates after
the change; a shift from `MainContext` to `DmaEngine`, for example, indicates a different
next optimization target.

Makespan is static: it describes the compiler's plan. Confirm any promising change with
the Stage 1 test on RNGD:

```sh
./scripts/rngd_test.sh
```

Do not report a cycle improvement without a reproducible schedule comparison or separately
documented RNGD evidence. A faster but numerically incorrect kernel does not receive Stage 1
performance credit because accuracy is a hard grading gate.

## 지속할 상세 디버깅 절차

사용자 요청에 따라 Viewer를 실제 실행하고 임계 경로·DMA 효율·공유 엔진 대기를 상세 분석한다. 실행 명령, 화면 확인 순서와 현재 관측은 [docs/schedule-debugging.md](docs/schedule-debugging.md)에 유지한다. 정적 schedule과 실제 RNGD cycle을 구분하고 검증된 최고 소스를 원본 src에 반영한다.


## 2026-09-10 지속 검증 규칙

- 유망한 후보는 평가 대상 23파일을 동결하고 같은 source SHA로 공식 3회 제출한다. 세 기록을 보존하며 접수 ID가 없으면 성공으로 기록하지 않는다. 제출 중에는 별도 checkout에서 다음 실험을 계속한다.
- 원본 `src`는 `docs/current-best.json`의 공식 최고 source와 hash가 일치해야 한다. 더 높은 공식 score와 세 커널 PASS를 확인하면 이전 7파일을 백업하고 Git index hash를 보존하며 반영한다. 패치는 임시 기준 트리에 실제 적용해 결과 hash까지 검증한다.
- 빌드 harness를 교체할 때는 `write_bytes`처럼 현재 mtime으로 기록한다. `copy2`로 과거 mtime을 복원하면 Cargo가 이전 ELF를 fresh로 재사용할 수 있다. 공개/연속 ELF hash가 다르고 연속 환경변수 marker가 있으며, 세 nonempty kernel image가 같은지 확인한다. 빈 비Stage1 bin은 전체 컴파일 검증으로 세지 않는다.
- compiled schedule의 최대 lifetime과 실제 RNGD cycle을 구분한다. Viewer에서 실제로 JSON을 열고 캡처·DOM·명령을 보존한다. 현재 SDK의 runtime callback은 첫 logical cluster만 관측하므로 `Cluster` span 전체를 칩 전체 idle이나 제거 가능한 시간으로 해석하지 않는다.
- 회귀와 컴파일 실패도 실험별 source/log에 남긴다. 작은 DMA, VRF prepare, 단일 cluster처럼 기각한 가설은 전제가 달라졌을 때 다시 검토한다.
- 공식 최고와 1등의 격차 그림은 `scripts/plot_current_best.py`와 `docs/current-optimization-progress.png`, 교대 실측 비교는 `scripts/plot_recent_output.py`와 `docs/recent-output-comparison.png`로 재생성한다.


## 2026-09-11 사용자 지시: 최약 커널 반복 개선

종합 1위에 도달해도 최적화를 종료하지 않는다. 공식 최고 소스를 유지하면서 각 커널의 `현재 cycle / 다른 팀 해당 커널 최저 cycle`을 비교해 가장 큰 비율의 커널부터 계속 개선한다. 커널별 경쟁 팀은 서로 다를 수 있다. 세 커널 모두 경쟁 기록보다 빠르면 우위가 가장 작은 커널을 다음 대상으로 선택한다. 공식 기록·리더보드가 갱신될 때 우선순위를 다시 계산한다.

현재 E149/f5936e50의 비교는 O45,068/43,049(+4.69%), FFN289,342/289,031(+0.11%), QKV98,653/103,145(−4.36%)이므로 O가 우선이다. 정확성 계약을 유지하고 actual compiled schedule→실측 교대 비교→유망 소스 동일2회 제출→공식 최고 반영→우선순위 재계산을 반복한다. 실제 source hash와 공개/연속 kernel image를 확인한다. 사용자가 별도로 제출한 동일 소스는 확인 후 반복 측정에 포함하며 중복 큐를 만들지 않는다.

`docs/leaderboard-snapshot.json`을 공개 API 응답으로 갱신한 뒤 `python3 scripts/refresh_optimization_priority.py`로 `docs/optimization-priority.json/md`를 재생성한다. 순위는 팀별 대표 제출만 공개하므로 숨겨진 전체 제출의 커널별 최저값을 알 수 있다는 뜻은 아니다. 진행 그림은 `docs/current-optimization-progress.png`이며 각 bar는 다른 팀 해당 커널 최저 기록과 비교한다.

사용자는 "E152 및 후속 후보 전송 승인"으로 Stage1 테스트 ELF·실행 스크립트와 기존 공개/연속fixture의 기존Arena 목적지 후속 전송을 명시적으로 승인했다. E152 자동 검토 장애는 해당 승인 후 해소돼job16629를 완료했다.


## 2026-09-11 사용자 지시: 근접 성능 후보도 공식 검증

사용자는 "비슷하게라도 나오면 실제로 제출해서 검증"하도록 지시했다. 정확성 및 소스/바이너리 동일성을 통과한 후보는 Arena에서 확실한 우위가 없어도 현재 기준과 성능이 비슷하면 공식 서버에 동일23개 평가소스로 두 번 제출한다. 교대 비교 중앙값/평균/승률은 판단 근거로 보존하되 4/8승이나 작은 차이만으로 공식 검증을 생략하지 않는다. 큰 회귀·정확성 실패·컴파일 실패 후보와 실제 미검증 후보는 구분한다. 공식 결과가 더 좋을 때 원본src/best-checkout/패치를 갱신하고 최약커널을 다시 고른다. 이미 완료된 동일소스 반복을 불필요하게 재제출하지 않는다. E161/E159는 이 지시에 따라 queue10에서 각각2회 공식 검증으로 전환했다.


## 2026-09-11 최신 지시: 공식 제출 3회

사용자는 “3번씩 실행해도될듯? 제출하는거”라고 지시했다. 앞으로 정확성 및 source/ELF 동일성을 검증한 유망·근접 후보는 **동일 frozen 평가 소스 23개로 공식 제출 3회**를 수행한다. 세 결과를 모두 보존하고 최고 점수 소스를 원본 src에 유지하며, 최솟값만으로 구조적 개선을 단정하지 않는다. 대기 중에는 별도 checkout의 다음 실험을 계속한다. 기존 2회 완료 후보는 현재 최고·유망 후보부터 세 번째 결과를 추가하고, 사용자 별도 제출은 동일 소스임이 확인된 경우에만 반복에 포함한다. 실패·명확한 회귀 후보를 일괄 재제출하지 않는다.

현재 최고는 E165/3fae5048, 98,125 / 43,492 / 285,067 cycle, 점수6.7575, 세 커널 PASS다. 원본 src/best-checkout/patches/e165에 반영했다. E161은 이전 최고로 보존한다. 최신 최고·제출 상태·우선순위는 docs/current-best.json, docs/ongoing-optimization-state.json, docs/optimization-priority.json을 우선한다.


2026-09-11 공식3회 결과 갱신: 같은 E165의 세 번째 006582e2가 98,359 / 40,066 / 282,925, 점수6.9569, 세 커널PASS로 최고다. 원본 소스는 동일 E165이며 3회 결과를 모두 보존한다. source 변경 없는 기록 차이를 구조적 개선으로 해석하지 않는다. 현 비교에서 FFN의 우위가 가장 작아 다음 우선대상은 FFN이다. 패치 docs/patches/e165-r3 및 current-best.json을 따른다.


2026-09-11 최신 최고: E172/339089eb, 100,571 / 38,873 / 284,644 cycle, 점수6.9613, 세 커널 PASS. 동일 소스 공식3회가 완료됐고 원본 src/best-checkout/patches/e172를 기준으로 한다. FFN의 경쟁 대비 우위가 −1.52%로 가장 작아 다음 우선 대상이다. E175 FFN의 8쌍 중앙값은 E165 대비 −1.03%, 이를 최고 O172와 결합한 E176을 실제 검증한다. 최신 상태는 docs/current-best.json과 ongoing-optimization-state.json을 먼저 읽는다.


## 2026-09-11 최신 공식 최고: E211

공식 c0ffdd40:92,586 /39,406 /273,805 cycle,점수7.2164,세 커널PASS. 원본src 및 best-checkout은 이 소스와 일치한다. 후보마다 같은동결소스3회 제출하고 모든결과를보존한다. 종합1위에도 경쟁대비우위가가장작은커널부터반복개선한다. 최신근거는 [current-best](docs/current-best.md), [우선순위](docs/optimization-priority.md), [진행그림](docs/current-optimization-progress.png), [E211패치](docs/patches/e211/README.md)에서확인한다. 앞부분의 오래된 미검증체크리스트는2026-09-10당시기록이다.


## 2026-09-11 후속 진단에서 확인한 제약

E229/E231은 같은 정적42522와 Main34/Sub21이어도 큰 RoPE 그룹은 실제 timeout, 두 F32 flit 그룹은 정확성 PASS였다. SDK0.6.0은 한 Vector ALU 노드의 외부 VRF 둘과 여러 provenance SRAM fetch 입력을 지원하지 않으므로 실제 진단을 확인한다. 단순 weight preload 분리(E115)는 같은 image, 입력 TRF 공유(E234)는 Sub가 줄어도 실제9.12% 회귀였다. [RoPE 제약](docs/qkv-rope-e229-e231.md), [E232 runtime](docs/qkv-runtime-e232.md), [실험 기록](docs/optimization-log.md)을 참조한다. 공식 제출은 계속 동일 frozen source23으로3회이며 모든 결과를 보존한다.


## 2026-09-11 E235 이후의 재현·진단 기록

동일 소스 공식3회 정책을 계속 적용했다. E235의 Q weight4+4 분할은 실제 스케줄에서 DMA/첫 타일 contraction 중첩이 생겼지만 전체8쌍과 공식3회에서 최고를 갱신하지 못했다. E238의 입력 RMS division 이동은 head/입력 정규화·EPS를 유지한 수학적 재배치이며 일반BF16 host40조건과 실제 공개/연속27process를 모두 통과했다. 다만 E240 runtime에서 TuExec45→46, 부모 대비 trace중앙값+2.056%로 지속적인 개선을 입증하지 못했다. E239도 역순 포함8쌍을 확인한 뒤 근접 후보로 공식3회 제출했다. [전송·RMS 기록](docs/qkv-transfer-rms-e235.md), [실제 runtime 그림](docs/qkv-runtime-e240.md)을 따른다.

E235/E238/E239 source generator는 별도 임시 checkout에서 다시 실행해 동결 source42와 모든 SHA가 일치함을 확인했다. SDK0.6.0에서는 floating contraction의 혼합 dtype(F8×BF16), `DmTensorView.clone()`의 MIR 변환, 비정형6행 transpose가 검증된 제약이다. 실패 로그와 최종 재현 소스를 각각 보존한다. 입력을 `begin_interleaved`로 나눈 VectorTensorPair는 공개 API상 zip으로 한 tensor에 합치는 경로를 사용하므로, 이를 두 독립 scalar 결과를 그대로 저장하는 API로 가정하지 않는다. 이 마지막 항목은 API 조사이며 실제 실패 실험을 뜻하지 않는다.

## 2026-09-11 E242 이후의 의존성 검증

Q/K/V 공동 RMS는 Main 수가 줄어도 V 완료까지 Q/K 후처리를 지연시켜 실제24% 이상 회귀했다. V를 제외한 Role3 표현은 현재 compiler에서 split/stride 오류로 실패했고, cross-cluster RoPE DM 복제도 tag 오류를 만났다. 작은 버퍼라는 이유로 매핑·동기화 지원을 가정하지 않는다. K/V TRF를 분리한 E246은 공개8쌍에서개선돼같은소스로공식3회모두검증했으나최고는갱신하지못했다. info/trace의다른방향도보존한다. [원본/재현](docs/qkv-joint-rms-e242.md), [실제 trace](docs/qkv-runtime-e248.md).


## 2026-09-11 후속: source 순서와 타일 분할의 실제 확인

E255는 channel DMA 선언을 옮겨도 E253과 같은 image/lifetime이었다. E256은 유효 output을104/16행으로 나눠 HBM에 써도 각 masked DMA가 전체 쓰기와 같은2,040cycle이어서 실제13.87% 회귀했다. 타일의 논리 byte 수만 보고 전송 비용이 비례 감소한다고 가정하지 않는다. E257 norm32 inverse는 정적222cycle 증가에도 실제8쌍 중앙값2.71% 감소로 동일소스 공식3회 검증한다. [원본·수치·재현](docs/output-channel-e253.md).
