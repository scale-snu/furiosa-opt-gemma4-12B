# Gemma-4 Stage 1 작업 지침

이 파일은 이 저장소 전체에서 작업하는 에이전트가 참고할 프로젝트 지침이다. 설명과 작업 기록은 한국어로 작성하고, 코드 식별자와 명령어는 원문을 유지한다.

환경·진행 상태 확인일: **2026-09-10**. 아래 버전과 경로는 이 날짜에 확인한 값이며, 새 세션에서는 실제 파일과 도구 버전을 다시 확인한다.

## 현재 목표와 자료

현재 작업은 **Furiosa RNGD에서 Gemma-4-12B-it의 Stage 1 커널 세 개를 최적화하는 것**이다. 정확성을 유지하면서 커널별 실제 RNGD cycle을 줄인다. 각 커널은 한 번의 호출 단위로 평가된다. 사용자가 범위를 바꾸기 전까지 Stage 2의 E2E 서빙, 토크나이저, 이미지·오디오 전처리 최적화로 작업을 확장하지 않는다.

다음 자료를 우선 읽고, 이 요약과 충돌하면 해당 원본을 다시 확인한다.

| 자료 | 참고할 내용 |
|---|---|
| [README.md](README.md) | Stage 1 범위, 제출 계약, 정확도 기준, 평가·환경 준비 명령 |
| [OPTIMIZATION.md](OPTIMIZATION.md) | 스케줄 덤프 → 병목 분석 → 수정 → 비교 절차 |
| [tests/test_kernels.rs](tests/test_kernels.rs) | Stage 1 정확성과 성능 평가의 기준 구현 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | host/device 구분, 모듈 역할, 양자화와 모델 구조 |
| [src/axes.rs](src/axes.rs), [src/lib.rs](src/lib.rs) | 텐서 축 크기와 모델 상수의 실제 정의 |
| [Programming Tensor Contraction Processors.pdf](Programming%20Tensor%20Contraction%20Processors.pdf) | TCP 매핑, 데이터 이동, 연산 엔진, 스케줄링 프로그래밍 가이드 |
| [docs/schedule-generation-log.md](docs/schedule-generation-log.md) | 0.6.0 스케줄 생성 검증, 원본 경로와 정적 makespan |
| [docs/rngd-test-log.md](docs/rngd-test-log.md) | 원격 제출 스크립트 수정, 빌드·모의 검증, 실제 제출 미실행 기록 |

PDF의 로컬 경로는 `/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/Programming Tensor Contraction Processors.pdf`이다. 아래 페이지 번호는 현재 PDF의 1부터 시작하는 페이지 번호다. PDF의 일반 템플릿·예제 지침보다 이 저장소의 Stage 1 제출 계약을 우선한다. 예제에 맞추려고 커널을 `src/kernel/`로 옮기거나 테스트 구조를 바꾸지 않는다.

## 최적화 대상과 수치 계약

세 진입점은 모두 `src/ops.rs`에 있다. `sliding_attention` 자체는 Stage 1의 세 평가 대상에 포함되지 않는다.

| 커널 | 보존할 연산 흐름 | 절대 허용 오차 | 상대 허용 오차 |
|---|---|---:|---:|
| `ops::sliding_project_qkv` | 입력 RMSNorm → Q/K/V projection → Q/K/V 정규화 → Q/K RoPE → Q 출력 및 K/V ring-cache 쓰기 | `0.04` | `1e-2` |
| `ops::sliding_attention_output` | head broadcast → O projection → post-attention RMSNorm → residual add | `0.05` | `1e-2` |
| `ops::decoder_feedforward` | pre-FF RMSNorm → GeGLU MLP → post-FF RMSNorm → residual add → layer gate | `0.01` | `1e-2` |

- 비교 조건은 원소별 `abs(actual - expected) <= atol + rtol * abs(expected)`이다. 비유한 출력은 실패한다. 세 커널 모두 정확성을 통과해야 성능 평가를 받을 수 있다.
- `sliding_project_qkv`에서 Q/K는 학습된 RMSNorm weight를 사용하고 V는 weight 없는 정규화를 사용한다. README의 짧은 연산 설명만 보고 V 정규화를 없애지 않는다.
- Q/K/V/O projection은 `f8e4m3` weight와 출력 채널별 `bf16` scale을 사용한다. MLP는 packed `f4e2m1` weight, 16개 원소 단위 `f8e4m3` scale, 행렬별 `f32` global scale을 사용한다. 패킹, scale 의미와 적용 순서를 보존한다.
- GeGLU의 의미는 `down(gelu(gate(x)) * up(x))`이다. 다른 모델 예제의 SwiGLU/SiLU로 바꾸지 않는다. RMSNorm의 `EPS = 1e-6`, reduction 축, cast·반올림 위치 변경은 수치 영향까지 검증한다.
- Q 출력, K/V cache의 갱신 위치, residual의 제자리 갱신, 마지막 layer gate까지 커널 계약에 포함된다. `kv_offset`과 `rope_offset`은 현재 호출부에서 **바이트 오프셋**으로 전달된다. 원소 번호나 행 번호로 해석하지 않는다.
- 공개 fixture의 고정 위치·입력·출력을 하드코딩하거나 정규화·양자화 처리를 생략해 테스트에 맞추지 않는다.

## 수정 가능한 범위

- 평가에 반영되는 구현 변경은 `src/device/` 및 `src/ops.rs`, `src/ops_vision.rs`, `src/ops_audio.rs`의 **함수 본문**이다. 현재는 세 Stage 1 진입점과 그 호출 경로에 집중한다.
- 모든 기존 `#[device]` 함수의 이름, 인자, 타입, 반환 타입을 유지한다. 공개 텐서 레이아웃과 모델의 수치적 동작을 유지한다.
- `ops.rs`, `ops_vision.rs`, `ops_audio.rs`는 crate root에 그대로 둔다. 컴파일된 커널 이름에 `module_path!()`가 들어가므로 모듈 이동·이름 변경은 평가 계약을 깨뜨린다.
- `src/axes.rs`, `src/host/`, `src/api/`, `src/bin/`, `src/lib.rs`, `tests/` 변경은 Stage 1 평가에 반영되지 않는다. 이 파일을 수정해야만 동작하는 최적화나 테스트 허용 오차·기준값 변경을 해결책으로 삼지 않는다.
- 문서와 로컬 분석용 스크립트는 작업 지원용으로 남길 수 있지만, 제출 구현이 평가 제외 파일의 변경에 의존하면 안 된다.
- 공통 helper 변경 전 호출처를 확인한다. 특히 `device/shared/rmsnorm.rs`, `device/shared/mlp.rs`는 다른 attention·vision·audio 경로에도 영향을 줄 수 있으므로 관련 호출 경로의 컴파일·수치 영향을 확인한다.

## 먼저 살펴볼 코드

| 경로 | 확인할 부분 |
|---|---|
| `src/device/layout.rs` | `Cluster`, `Slice`, `Replicated`, hidden/head broadcast |
| `src/device/sliding/projection.rs` | Q/K/V/O projection, weight 로드, TRF 배치 |
| `src/device/sliding/rmsnorm.rs` | Q/K/V의 head별 정규화 |
| `src/device/sliding/rope.rs` | RoPE 계산, 테이블 접근과 레이아웃 |
| `src/device/shared/rmsnorm.rs` | hidden 차원의 RMSNorm과 slice 간 reduction |
| `src/device/shared/mlp.rs` | NVFP4 복원·scale, up/gate/down projection, GeGLU |
| `src/device/shared/residual.rs` | residual add와 layer gate |
| `scripts/generate_references.py`, `scripts/reference/gemma4.py` | fixture 생성과 PyTorch 참조 연산 |

현재 주요 축은 `H=3840`, `L=15360`, `Ns=8`, `Gs=2`, `Ds=256`, `Qs=4096`, `Ps=2048`, `Ts=1024`, `E=32768`이다. 정확한 값은 항상 `src/axes.rs`로 확인한다. 커널 내부에는 batch나 전체 sequence 루프를 추가하지 않는다. 한 토큰 위치의 작업을 처리하는 기존 호출 계약을 유지한다.

## TCP 프로그래밍 시 지킬 원칙

### 매핑과 메모리

- PDF 31–58쪽: 매핑은 논리 인덱스와 물리 위치의 대응이다. `Chip / Cluster / Slice / Lane`의 공간 분할과 `Time / Packet`의 스트림 배치를 먼저 그린 뒤 연산을 선택한다. 같은 shape라도 매핑에 따라 통신량과 연속 접근이 달라진다.
- `m![A / k, A % k]`는 축의 바깥 블록과 블록 내부를 나눈다. `#`는 패딩, `=`는 view의 유효 범위 축소다. 패딩 종류를 구분한다. 일반 `#`는 임의 값, `#{0}`은 0임을 나타내는 계약, `#{!}`는 접근 불가 영역이다. `#{0}` 표기 자체가 메모리를 0으로 채우지는 않는다.
- `tile`은 복사 없는 view다. 시작 인자는 해당 매핑의 논리 시작 좌표이며, 단순 tile 번호로 가정하지 않는다. 쓰기 view는 tile 밖을 덮어쓰지 않도록 `#{!}` 등 실제 API의 제약을 따른다.
- `unsafe { tensor.reshape() }`는 데이터를 이동하거나 broadcast하지 않는다. 새 매핑이 주장하는 값이 각 물리 위치에 이미 있어야 한다. 원소 수가 같다는 이유만으로 사용하지 말고, 재해석이 유효한 이유를 주석으로 남긴다. 관련 설명은 PDF 166쪽도 참고한다.
- PDF 53쪽, 160–168쪽: DM은 slice당 512 KB, VRF는 slice당 8 KB를 기준으로 확인한다. TRF는 데이터 타입·활성 lane·bank·half 모드에 따라 사용 가능한 배치가 달라지므로 상세 제약과 컴파일러 검증을 확인한다. 동시에 살아 있는 중간 텐서와 레지스터 operand까지 고려한다.
- PDF 59–64쪽, 98–123쪽: HBM↔DM 이동은 `ctx.tdma`, host↔HBM은 `ctx.pdma`다. Tensor Unit은 DM을 fetch하고 DM에 commit한다. DM↔DM 이동은 DMA와 fetch/commit 경로를 비교하되, 어느 쪽이 빠른지는 스케줄로 확인한다.
- 불필요한 HBM 왕복, 반복 broadcast, 작은 DMA 명령의 반복, 비연속 접근을 병목 후보로 본다. 전송 크기·주소 정렬·HBM/DM bank 분산을 확인하고 dtype을 고려해 **바이트 단위**로 계산한다.

### 연산과 병렬성

- PDF 124–128쪽: 기본 흐름은 `Fetch → Adapter/Switch → Collect → Contraction/Vector → Cast/Transpose → Commit`이다. Collect 이후는 32바이트 flit 기준이므로 dtype 변경 때 Packet과 padding도 함께 확인한다.
- PDF 160–195쪽: contraction은 TRF에 둔 operand와 스트리밍 operand를 결합한다. reduction을 Packet, Time, Lane 중 어디에서 수행할지 명확히 하고, 필요한 slice 간 reduction을 누락하지 않는다.
- PDF 196–261쪽: Vector Engine 입력은 `i32`/`f32`다. `bf16` 등을 직접 처리하는 경로는 widening을 확인한다. float 연산의 4-way narrow와 이후 widen을 맞춘다. 유효 원소가 양쪽에 있으면 `narrow_split`이 필요하며, `narrow_trim`으로 유효 값을 버리지 않는다.
- PDF 262–278쪽: cast 후 padding과 `commit_trim`의 유효 범위를 확인한다. 연산을 합치면서 중간 `bf16` 반올림을 바꾸면 오차가 달라질 수 있다.
- PDF 287–296쪽: `ctx.main`, `ctx.sub`, DMA가 항상 병렬로 실행되는 것은 아니다. 데이터 의존성, 동일 엔진 점유, 공유 메모리의 RAW/WAR/WAW 의존성과 bank 경합을 확인한다. 소스 순서만 바꾸고 병렬화됐다고 판단하지 않는다.
- 같은 DM bank에 대한 연속 접근이 여러 엔진 합계로 64회 이상 쌓이는 패턴을 피한다. 컴파일러가 일부 패턴을 직렬화하더라도 동시 명령 전체의 경합을 스케줄에서 확인한다.
- TRF double buffering의 half 배치는 스케줄러가 선택한다. 두 half는 bank를 공유하므로 타일을 줄였다는 이유만으로 preload와 연산이 겹친다고 가정하지 않는다.

## 환경과 실행 명령

### 현재 확인한 환경

| 구성 | 현재 값 |
|---|---|
| 호스트 환경 | x86_64, Ubuntu 24.04.3 LTS, GLIBC 2.39 |
| Rust | `nightly-2026-05-01`, `rustc 1.97.0-nightly (f53b654a8 2026-04-30)` |
| 라이브러리 | `Cargo.toml`의 `furiosa-opt-std = "=0.6.0"` |
| 프로젝트용 컴파일러 | `target/toolchains/furiosa-opt-0.6.0/bin/cargo-furiosa-opt`, 버전 `0.6.0` |
| 전역 컴파일러 | `~/.cargo/bin/cargo-furiosa-opt`, 버전 `0.7.0` — 이 프로젝트 빌드에 그대로 사용하지 않는다 |
| 원격 CLI | `furiosa-arena 0.8.0`; `rngd` 실행 파일은 없다 |
| Schedule Viewer | 설치 기록 기준 `0.3.0`; `--version`은 지원하지 않으며 `--help`로 실행 옵션을 확인한다 |
| Python | `~/miniconda3/envs/micro/bin/python3`, 버전 `3.14.7` |
| Python 패키지 | `micro` 환경의 패키지 메타데이터 기준 NumPy `2.5.3`, PyTorch `2.14.0`, safetensors `0.8.0` |

지원 최소 환경은 x86_64 Ubuntu 22.04 이상, GLIBC 2.34 이상이다. 시스템 의존성은 `build-essential`, `libclang-dev`, NPU 빌드용 `gcc-aarch64-linux-gnu`이며 현재 `/usr/bin/aarch64-linux-gnu-gcc`가 있다.

### 도구 선택과 복구

모든 명령은 저장소 루트에서 실행한다. NPU 명령은 SDK와 Rust 버전을 검사하는 공통 스크립트로 실행한다.

```bash
cd /scale/cal/home/sehwan/furiosa-opt-gemma4-12B
source "$HOME/.cargo/env"
./scripts/furiosa.sh --version
# 기대 출력: cargo-furiosa-opt 0.6.0
```

- `Cargo.toml`은 라이브러리 버전, `rust-toolchain.toml`은 Rust 버전, PATH는 `cargo-furiosa-opt` 실행 파일을 선택한다. 라이브러리 버전만 바꿔도 빌드 도구가 자동으로 바뀌지는 않는다. 현재 검증된 0.6.0 조합과 고정된 nightly를 유지한다.
- `scripts/furiosa.sh`는 `Cargo.toml`의 정확한 SDK 버전과 `rust-toolchain.toml`의 Rust 버전을 읽어 로컬 컴파일러를 선택·검사한다. `generate_compiled_schedules.sh`, `local_test.sh`, `run.sh`, `run_server.sh`도 이 경로를 사용한다. `rngd_test.sh`는 기존대로 자체 PATH에서 로컬 0.6.0을 선택하며 실행 흐름을 유지한다. Stage 1에서는 전체 모델·서버 실행으로 범위를 넓히지 않는다.
- 터미널에서 `cargo furiosa-opt`를 직접 실행하면 여전히 전역 도구를 선택할 수 있다. 필요하면 `export PATH="$PWD/target/toolchains/furiosa-opt-0.6.0/bin:$PATH"`를 설정하고 버전을 확인한다. 스크립트 안의 PATH 변경은 부모 셸에 전파되지 않는다.
- 현재 `base-template/`는 없다. 별도 예제 생성은 Gemma 빌드에 필요하지 않다.
- 프로젝트용 도구가 없으면 아래 명령으로 SDK와 같은 버전을 로컬에 설치한다. 올바른 버전이 이미 설치돼 있으면 재사용한다. `target/` 정리 시 로컬 도구도 없어질 수 있다.

```bash
./scripts/furiosa.sh --install
```

설치와 실행 시 공통 스크립트가 버전을 검사한다. 환경 재빌드 결과와 재현 명령은 `docs/environment-log.md`에 기록한다.

### Fixture 준비

현재 `ref/fixtures.safetensors`가 있다. 67,336 bytes, 텐서 38개이며 SHA-256은 `9801c5732d9f5bef89d544631c37f74dae1603df35b67a2cfdd6377915b7599e`다. Python/Rust가 같은 PRNG로 입력을 재현하며 fixture에는 기대 출력과 입력 checksum이 들어간다. 전체 모델 checkpoint는 필요하지 않다.

파일이 없거나 명시적으로 기준값을 재생성해야 할 때만 다음 명령을 사용한다. 에이전트 셸의 기본 `python3`는 `/usr/bin/python3`일 수 있으므로, 사용자 터미널의 `micro` 환경이 자동 적용됐다고 가정하지 않는다.

```bash
"$HOME/miniconda3/envs/micro/bin/python3" scripts/generate_references.py
```

### 원격 제출 없이 빌드만 확인

다음 명령은 Arena 로그인이나 원격 제출 없이 NPU 테스트 바이너리만 빌드한다.

```bash
CARGO_BUILD_JOBS=12 ./scripts/furiosa.sh test --release --test test_kernels --no-run
```

이 명령은 테스트 바이너리를 만들며 원격 제출이나 테스트 실행을 하지 않는다. 일반 `cargo`의 CPU 실행, NPU용 바이너리 빌드, 커널 스케줄 생성, 실제 RNGD 정확도·cycle 검증을 구분한다. 빌드 성공만으로 Stage 1 통과를 선언하지 않는다.

### 기준 스케줄 저장

스케줄 생성 스크립트는 세 Stage 1 커널을 0.6.0으로 컴파일하고 실행마다 새 `target/schedules/<시각>-stage1.<고유값>/` 디렉터리에 JSON·로그·컴파일러 버전을 저장한다. 저장 위치는 실행 출력으로 확인한다. 기존 결과는 덮어쓰지 않는다. 실험별 가설과 baseline·후보 구분은 별도 작업 기록에 남긴다.

```bash
./scripts/generate_compiled_schedules.sh
furiosa-schedule-viewer
```

`--dump-*`는 커널 하나씩 실행한다. 필터는 기본적으로 부분 문자열 매칭이므로 `--exact`를 유지한다. viewer의 기본 주소는 `127.0.0.1:9254`이며 JSON을 열어 확인한다. 상세 도구 설명은 PDF 298–302쪽, 308–310쪽을 참고한다.

개별 커널만 덤프할 때는 `./scripts/furiosa.sh compile ops::<함수명> --exact --dump-schedule <새 JSON 경로>`를 사용한다.

### 실제 RNGD 검증

원격 실행 전 `furiosa-arena login` 설정이 필요하다. 별도 주소를 안내받았다면 `FURIOSA_ARENA_URL`로 지정하며, 생략하면 CLI의 기본 주소를 사용한다. 기존 `RNGD_URL` 설정도 스크립트가 전달한다. 주소와 인증 값을 추측하거나 비밀 값을 문서에 저장하지 않는다.

```bash
./scripts/rngd_test.sh
```

- `--no-build`: `target/release/deps/`에서 수정 시각이 최신인 테스트 바이너리를 재사용한다. 소스·도구 버전 변경 후에는 빌드를 생략하지 않는다. 여러 버전의 산출물이 섞여 있으면 원하는 바이너리라고 보장할 수 없다.
- `--no-wait`: 제출만 하고 반환한다. 반환 성공은 테스트 통과가 아니며 job 결과를 확인해야 한다.
- 추적: `furiosa-arena status <id>`, `furiosa-arena logs <id> --follow`, `furiosa-arena list`.
- 관련 환경 변수: `RNGD_JOB_NAME`, `RNGD_TIMEOUT`(원격 실행 제한, 생략 시 서버 기본값), `RNGD_WAIT_TIMEOUT`(대기열을 포함한 로컬 대기 제한, 기본 1800초), `RNGD_POLL_SECONDS`(기본 5초). 이 서버에서 1800초 실행 제한은 최대 70초를 초과해 거절됐다. 실행 제한과 로컬 대기 시간을 혼동하지 않는다.
- `rngd_test_<random>`은 작업 이름이다. 접수 응답의 `submitted job <id>`에 있는 숫자를 `furiosa-arena status <id>`에 사용한다. `submitting` 출력만으로 제출 성공을 판단하지 않는다.
- 로컬 RNGD와 SDK가 준비돼 있으면 `./scripts/local_test.sh`를 사용한다. 이 환경에서 로컬 RNGD 실행 가능 여부는 확인하지 않았다.
- `RNGD_WAIT_TIMEOUT`에 도달하면 스크립트가 종료될 뿐, 원격 작업이 자동 취소되지는 않는다. 작업 ID로 상태를 확인한 뒤 필요하면 취소한다.
- 스크립트는 `TUC_PROFILE_LEVEL`의 기본값을 `info`로 사용한다. `cycles=none observed`는 측정 실패/미관측 상태이며 0 cycle이나 성능 개선으로 기록하지 않는다.

### 이미 확인한 오류와 대응

| 증상 | 확인할 사항 |
|---|---|
| `FURIOSA_OPT_OUT_DIR not defined`가 여러 `#[device]`에서 반복 | 0.6.0 라이브러리와 0.7.0 컴파일러 조합에서 재현했다. 두 버전을 먼저 확인하며, 환경 변수를 임의로 만들어 오류를 숨기지 않는다. |
| `decoder_feedforward: visa: while lowering Dma`, `merge failed` | 0.7.0 도구를 사용한 빌드·스케줄 생성에서 발생했고 0.6.0 도구로 세 스케줄 생성에 성공했다. 먼저 도구 버전을 확인한다. 구체적인 컴파일러 내부 원인은 아직 분석하지 않았다. |
| `could not compile ... due to N previous errors`만 보임 | JSON 출력을 변수에 담으면 상세 진단이 안 보일 수 있다. 현재 스크립트처럼 `--message-format=json-render-diagnostics`로 오류를 stderr에 표시한다. |
| `submitting` 뒤 아무 메시지 없이 종료 | 예전 `rngd` 호출과 `set -e` 때문에 제출 오류가 숨겨졌었다. 현재 스크립트의 `furiosa-arena` 호출·실패 출력을 확인하고, 숫자 접수 ID가 없으면 목록을 조회한다. |

오류 뒤에 `Finished ... compiled`가 출력돼도 전체 명령의 성공을 보장하지 않는다. 종료 코드, 생성된 파일, 실제 검증 결과를 함께 확인한다.

## 반복할 최적화 절차

1. `git status --short`와 diff를 읽고 기존 사용자 변경을 구분한다. README, 평가 테스트, 대상 커널과 helper의 현재 구현을 확인한다.
2. 환경·fixture를 준비하고 세 커널의 기준 정확도, 실제 cycle, 스케줄을 저장한다. 측정되지 않은 기준값을 만들어 쓰지 않는다.
3. 스케줄의 `max(instructions[*].lifetime.end)`로 makespan을 구한다. 긴 노드의 `contexts`, `lifetime`, `description`에 있는 소스 위치, 연결 텐서, DMA `util`/`total_util`을 확인한다.
4. `DmaEngine`의 전송, `SubContext`의 register preload, `MainContext`/`VectorEngine`의 연산 중 임계 구간을 지배하는 원인을 찾는다. 긴 노드 하나보다 최종 종료 시각을 결정하는 의존 경로를 기준으로 판단한다.
5. 병목에 맞는 가설 하나를 선택한다. 엔진 경로, 타일·분할 크기, 매핑·패딩, 전송 경계를 후보로 삼되 한 실험에서 변경을 섞지 않는다.
6. 후보를 컴파일하고 같은 조건으로 스케줄을 비교한다. 바꾼 매핑·타일은 명시하고, toolchain·입출력 shape·dtype·빌드 조건 등 다른 요인을 고정한다. 다른 조건이 바뀌면 별도 기준선을 만든다.
7. 유망한 후보는 Stage 1 테스트로 **세 커널 모두** 검증한다. 공통 helper를 수정했다면 다른 호출 경로에 대한 필요한 검증도 수행한다.
8. 정확성 및 성능 결과로 채택 여부를 결정한다. 실패한 실험도 기록하며, 되돌릴 때는 이번 실험의 변경만 제거한다.

`OPTIMIZATION.md`의 약 116,600 cycle·MainContext 96% 수치는 과거 예시다. 현재 기준 성능이나 현재 병목으로 인용하지 않는다. 정적 makespan 감소와 실제 RNGD cycle 감소는 각각 구분해서 보고한다.

## 기록과 완료 기준

- 최적화 실험을 시작하면 `docs/optimization-log.md`를 만들거나 기존 기록을 이어 쓴다. 날짜/run ID, 가설, 변경 파일, Git commit과 dirty 상태, 정확한 실행 명령, 도구 버전·하드웨어, fixture 정보, 스케줄·로그 경로, 전후 수치, 채택 여부와 다음 실험을 남긴다.
- 원본 스케줄과 로그는 실행별로 보존한다. `target/`은 무시되는 빌드 산출물 경로이므로 장기 보존 위치와 재생성 방법도 기록한다. 큰 바이너리·모델·fixture를 무조건 Git에 추가하지 않는다.
- 사용자 변경을 임의로 수정·되돌리거나 관련 없는 파일을 일괄 포맷하지 않는다. 완료 전 diff와 변경 범위에 맞는 검증 결과를 확인한다.
- 완료 보고에는 변경 내용, 검증 명령, 세 커널의 정확도·실제 cycle 결과, 기준 대비 변화와 미검증 범위를 적는다. 하드웨어나 인증이 없어 실행하지 못했다면 컴파일/정적 분석 결과와 명확히 구분한다.
- 최신 README의 대회 제출 명령은 `moa-submitter submit`이며 `src/ops.rs`와 `src/device/` 전체를 업로드한다. 점수는 세 커널 baseline speedup의 기하평균이고 팀 최고 점수가 게시된다. 마감·제출 횟수·실제 서버 설정은 확인된 정보만 사용한다. Arena 테스트와 MOA 제출은 별도 단계로 기록한다.

## 다음 작업의 시작점

현재 `src/ops.rs`의 Stage 1 세 함수 본문은 구현돼 있고, 과거의 `// TODO` 세 개는 없다. 기존 구현을 기준으로 최적화한다. 최근 검증 기준 Git commit은 `e3e928070c1e87848771904cf5d63beecdaa7e09`이며, 이후 작업에서는 현재 Git 상태와 diff를 다시 확인한다.

0.6.0 NPU 빌드·스케줄뿐 아니라 실제 Arena 검증도 진행했다. 사용자가 외부 RNGD 실행을 명시적으로 허용했으며 기준 job **15922**는 세 커널 PASS, 실제 cycle **259,937 / 412,007 / 3,703,859**다. 이전 제출 차단 기록은 당시 이력이고 현재는 실제 제출·결과 회수가 가능하다. 최신 실험과 원본 job은 [docs/optimization-log.md](docs/optimization-log.md) 및 [누적 실측표](docs/optimization-progress.md)를 확인한다.

사용자 제공 비교 baseline은 **250,514 / 404,633 / 3,703,473**, 넘어서야 할 최고 기록은 **101,132 / 46,318 / 289,085** cycle이다. 최고 기록과 장치·평가 조건 동일성은 아직 확인하지 않았다. 총 speedup은 비교 baseline 합계 / 같은 실행의 세 커널 합계이며 대회 점수 산식으로 해석하지 않는다.

실측 그림은 `docs/optimization-progress.png`, 목표 격차 그림은 `docs/optimization-targets.png`, 확대 가능한 HTML은 `target/optimization/progress.html`이다. 정확도 실패 실행은 기록에는 남기고 최고 성능에서 제외한다.

확인한 스케줄은 `target/schedules/20260910-140656-stage1.I8rJYK/`와 이후 생성된 `target/schedules/20260910-140731-stage1.84AoRL/`에 있다. 두 폴더의 컴파일러 기록은 0.6.0이고, 아래 값은 두 폴더의 JSON에서 같은 값으로 확인했다.

| 커널 | 스케줄 명령 수 | 정적 makespan | 실제 RNGD 정확도·cycle |
|---|---:|---:|---|
| `sliding_project_qkv` | 131 | 116,583 | 기준 job 15922 PASS / 259,937 |
| `sliding_attention_output` | 163 | 194,020 | 기준 job 15922 PASS / 412,007 |
| `decoder_feedforward` | 1,802 | 1,693,200 | 기준 job 15922 PASS / 3,703,859 |

위 값은 `max(instructions[*].lifetime.end)`로 계산한 정적 스케줄 수치다. 성능 개선이나 실제 RNGD cycle 측정으로 해석하지 않는다. `target/`, `ref/`, `Cargo.lock`은 현재 Git에서 제외돼 있으므로 새 checkout에 자동으로 존재한다고 가정하지 않는다. 재현에 필요한 로그·lockfile·fixture checksum은 실행별로 보존한다.

- [x] 실행 환경과 fixture 준비 상태를 확인한다.
- [x] 0.6.0 조합으로 NPU용 테스트 바이너리를 빌드한다.
- [x] 세 커널의 정적 스케줄과 makespan을 확보한다.
- [x] 세 커널의 실제 RNGD 정확도·cycle 기준선을 확보한다.
- [x] 스케줄 근거로 첫 병목과 실험 가설을 정한다.
- [x] 허용 범위 안에서 후보를 구현하고 전후 결과를 기록한다.
- [x] 세 커널 정확성과 실제 RNGD 성능을 검증하고 다음 병목을 정한다.

- [ ] 세 커널 각각의 실제 cycle을 사용자 제공 최고 기록보다 낮추고 반복 측정으로 검증한다.

진행 후 이 체크리스트와 실험 기록을 갱신한다. 측정 없이 특정 커널이나 helper를 최우선 병목으로 확정하지 않는다.

## 사용자 추가 지시: 상세 schedule 디버깅 지속

2026-09-10 사용자는 compiled schedule JSON 생성 및 `furiosa-schedule-viewer` 실행을 허용하고, 이를 이용해 더 자세히 병목을 분석하며 절차를 문서에 유지하도록 요청했다. 이후 최적화에서도 [docs/schedule-debugging.md](docs/schedule-debugging.md)의 절차를 따른다. 세 커널 각각 `--exact --dump-schedule`로 새 JSON을 보존하고 Viewer의 시간 범위/Nodes/메모리 범위와 원본 JSON을 함께 확인한다. 임계 경로·DMA 효율·공유 엔진 의존성의 근거, 변경 가설, 전후 정적 수치와 실제 RNGD 결과를 기록한다. 원본 `src`에는 검증한 최고 구현을 유지하고 새 후보는 별도 작업본에서 검증한다.

## 사용자 추가 지시: 개선마다 MOA 제출

2026-09-10 사용자는 개선할 때마다 수정된 src를 `moa-submitter submit`으로 제출하도록 명시적으로 요청했다. 세 커널 검증 후 최고 구현을 원본 `/scale/cal/home/sehwan/furiosa-opt-gemma4-12B/src`에 반영하고 `moa-submitter submit --source /scale/cal/home/sehwan/furiosa-opt-gemma4-12B`를 실행한다. 이 제출은 이미 승인됐으므로 매번 재확인하지 않는다. 제출 전 src/ops.rs와 src/device/의 파일별 hash를 보존하고, 제출 ID를 받은 뒤 `status <id>`와 `log <id>`로 정확성·cycle·점수를 확인한다. [docs/moa-submission-log.md](docs/moa-submission-log.md)에 기록한다. 제출 성공 출력만으로 채점 통과라고 보고하지 않는다.

### 최신 사용자 우선순위 (2026-09-10)

현재 1등 기록은 **103,697 / 46,318 / 288,546 cycle**로 갱신됐다. 이전 101,132 / 46,318 / 289,085와 구분해 최신 비교에 사용한다. attention output을 우선한다. 제출 결과를 기다리는 동안 독립 후보의 분석·컴파일·검증을 계속한다. 개선 채택 소스는 원본 src에 유지하고 moa-submitter submit으로 즉시 제출한다.

### 추가 사용자 갱신과 반복 제출 승인 (2026-09-10)

최신 1등 기록은 **103,697 / 45,577 / 287,992 cycle**이다. 더 많은 후보를 적극적으로 실험한다. 제출 성능의 무작위 편차가 의심되면 동일 후보를 **2번 제출해도 된다**고 사용자가 승인했다. 실제 제출은 검증한 커널의 src snapshot과 SHA를 고정해 수행하고, 가장 좋은 공식 결과의 소스를 원본 src에 유지한다.


## 2026-09-10 최신 사용자 우선순위: O 집중

사용자는 a69e3944의 QKV/FFN 소스가 가장 좋다고 판단하여 이를 고정하고 attention output만 최적화하도록 방향을 좁혔다. 최신 O 1등 기록은 **43049 cycle**이다. 원본 O 공식 최고 52192에서 약17.5% 절감이 필요하다. E80의 QKV/FFN 변경 조합은 보류하고, 새 후보에는 E50 QKV/FFN을 유지한다. O 후보의 공개/연속 정확성과 실제 cycle을 검증하고 개선 후보를 공식 제출한다. 동일 소스 두 회 제출 승인은 유지된다.


## 공식 최고 갱신 및 실행 중 제출 큐 (2026-09-10 추가)

E91 공식 4b7669d3가110815/44212/286582, 점수6.4423으로 최고를 갱신했다. 원본 src와 best-checkout에 반영했고 패치는 docs/patches/e91이다. 이후 최고는 docs/current-best.json을 우선한다. E50/a69e3944 QKV/FFN 소스는 고정, O 목표는 우선40000 이하이다. 사용자가 제공한 https://micro2026-moa.github.io/leaderboard.html 의 실제 공개 API를 읽고 docs/leaderboard-snapshot.json에 시각과응답을 보존한다.

compiled schedule 실제 viewer 캡처와 runtime trace 분석은 docs/attention-output-current-analysis.md를 참고한다. 정적 makespan과 실제cycle을 구분하고 Cluster 전체를 idle로 단정하지 않는다. 실행 중 공식 제출 큐는 target/moa-submissions/20260910-output-frozen-queue.json 및 .state.json이며, 같은 후보를 다른 프로세스에서 중복 제출하지 않는다. E96/E95/E94/E92 정확성 통과 소스23개를 hash확인후각2회평가한다. 원본 Git index는 변경하지 않는다.


E96/E95/E94/E92 각2회 공식 제출 큐는 완료했다. 최고는 E91/4b7669d3를 유지한다. 새 단독 제출 큐는 `target/moa-submissions/20260910-output-frozen-queue2.json` 및 `.state.json`이며 E107/E103을각2회평가한다. 중복 제출하지 않는다. E107(104+16)은 정적22771, Arena16459에서E96 O중앙값48096→44810.5(-6.83%),3/4개선이고공개·연속15process모두3커널PASS다. 최초공식ID db777785.


압도적인 전체점수 개선의 후속 가능성으로 E109 QKV packed FP8 host 검사를진행했다. 공개/기존3case와일반연속18case에서 Q/K/V norm/RoPE/cache 모델 모두PASS했고 global scale+H1920×2 encoding은wholeH와bitexact였다. E113은 E91 O/FFN고정의별도QKV nativepacked prototype 컴파일만진행한다. 아직device실측없으며 원본QKV를바꾸지않는다. 신규fixture생성/전송없이기존승인범위유지. O최적화와최고소스유지는계속한다.


### 2026-09-10 13:30 UTC 후속 갱신

현재 원본 최고는 E110 /82da3e4e, 107,669 /45,374 /287,356 cycle, 공식 점수6.4426이다. 두 번째 제출55a9e1b3은6.4230이다. `docs/current-best.json`과`docs/patches/e110/`을 기준으로 삼는다. 기존 E91은 백업했다. 최고 점수와 O 단일 최저cycle은 구분한다.

Runtime callback은 첫 logical cluster만 읽는 경로로 확인됐다. `docs/attention-output-current-analysis.md` 및 cluster-span-audit를 따른다. 빈 trace 구간을 칩 전체 idle로 해석하지 않는다. E117은104+16의 실제 Main/DMA 겹침을 검증했다. E121 direct cross-cluster DM은 동기화 checker에서 실패, E124 Q128은 DMA가 늘어 정적 회귀했으므로 원격 시험하지 않았다.

E119(H480 교대 배치)는16489/16492에서 공개8쌍+연속3case 세커널PASS, O중앙값-4.88%로 공식2회 큐5를 진행한다. E118은 QKV입력을8slice에서FP8변환한뒤복제해정적45,279→42,579이며job16493실측을진행한다. E116은FFNnativeFP8복원후보로host21casePASS,실측예정이다. 원본은공식최고갱신시에만반영한다.


### 2026-09-10 최신 우선 상태

원본 `src` 최고는 E131 / f4d02d5c, 공식99,739 /47,560 /289,234, 점수6.4921, 세 커널PASS. 이전 E110은 backup에 보존한다. `docs/current-best.json`과 `docs/ongoing-optimization-state.json`을 먼저 읽는다. 유망 후보는 동일 frozen23파일/hash로 공식2회 제출하고 두 결과를 보존한다. 병합 index는 변경하지 않는다.


## 2026-09-11 00시 최신 최고 상태

공식 E149/f5936e50(98,653/45,068/289,342,score6.6329),세커널PASS이며 공개API현재1위. 원본src와best-checkout에해당소스를반영했다. index는보존하고직전E131은백업했다. norm16+residual직접읽기; QKV127/FFN50유지. docs/current-best.json과docs/patches/e149가현재기준이며이전E131목록은이력이다. E152및후속 Stage1후보의바이너리·스크립트+기존공개/연속fixture를Arena에계속전송하도록사용자가명시적추가승인했다. 승인반영후E152job16629정상접수.


## 2026-09-11 사용자 지시: 최약 커널 반복 개선

종합 1위에 도달해도 최적화를 종료하지 않는다. 공식 최고 소스를 유지하면서 각 커널의 `현재 cycle / 다른 팀 해당 커널 최저 cycle`을 비교해 가장 큰 비율의 커널부터 계속 개선한다. 커널별 경쟁 팀은 서로 다를 수 있다. 세 커널 모두 경쟁 기록보다 빠르면 우위가 가장 작은 커널을 다음 대상으로 선택한다. 공식 기록·리더보드가 갱신될 때 우선순위를 다시 계산한다.

현재 E149/f5936e50의 비교는 O45,068/43,049(+4.69%), FFN289,342/289,031(+0.11%), QKV98,653/103,145(−4.36%)이므로 O가 우선이다. 정확성 계약을 유지하고 actual compiled schedule→실측 교대 비교→유망 소스 동일2회 제출→공식 최고 반영→우선순위 재계산을 반복한다. 실제 source hash와 공개/연속 kernel image를 확인한다. 사용자가 별도로 제출한 동일 소스는 확인 후 반복 측정에 포함하며 중복 큐를 만들지 않는다.

`docs/leaderboard-snapshot.json`을 공개 API 응답으로 갱신한 뒤 `python3 scripts/refresh_optimization_priority.py`로 `docs/optimization-priority.json/md`를 재생성한다. 순위는 팀별 대표 제출만 공개하므로 숨겨진 전체 제출의 커널별 최저값을 알 수 있다는 뜻은 아니다. 진행 그림은 `docs/current-optimization-progress.png`이며 각 bar는 다른 팀 해당 커널 최저 기록과 비교한다.

사용자는 "E152 및 후속 후보 전송 승인"으로 Stage1 테스트 ELF·실행 스크립트와 기존 공개/연속fixture의 기존Arena 목적지 후속 전송을 명시적으로 승인했다. E152 자동 검토 장애는 해당 승인 후 해소돼job16629를 완료했다.


## 2026-09-11 사용자 지시: 근접 성능 후보도 공식 검증

사용자는 "비슷하게라도 나오면 실제로 제출해서 검증"하도록 지시했다. 정확성 및 소스/바이너리 동일성을 통과한 후보는 Arena에서 확실한 우위가 없어도 현재 기준과 성능이 비슷하면 공식 서버에 동일23개 평가소스로 두 번 제출한다. 교대 비교 중앙값/평균/승률은 판단 근거로 보존하되 4/8승이나 작은 차이만으로 공식 검증을 생략하지 않는다. 큰 회귀·정확성 실패·컴파일 실패 후보와 실제 미검증 후보는 구분한다. 공식 결과가 더 좋을 때 원본src/best-checkout/패치를 갱신하고 최약커널을 다시 고른다. 이미 완료된 동일소스 반복을 불필요하게 재제출하지 않는다. E161/E159는 이 지시에 따라 queue10에서 각각2회 공식 검증으로 전환했다.


2026-09-11 최신 최고갱신: E161/367b33c0=97707/43866/292136,score6.6930,세커널PASS. 원본src/best-checkout/patches/e161을유지한다. E149는이전기준백업이다. 동일소스사용자반복909550aa는낮은결과로보존. 진행큐/우선순위는docs/ongoing-optimization-state.json과optimization-priority.json을우선한다.


## 2026-09-11 최신 지시: 공식 제출 3회

사용자는 “3번씩 실행해도될듯? 제출하는거”라고 지시했다. 앞으로 정확성 및 source/ELF 동일성을 검증한 유망·근접 후보는 **동일 frozen 평가 소스 23개로 공식 제출 3회**를 수행한다. 세 결과를 모두 보존하고 최고 점수 소스를 원본 src에 유지하며, 최솟값만으로 구조적 개선을 단정하지 않는다. 대기 중에는 별도 checkout의 다음 실험을 계속한다. 기존 2회 완료 후보는 현재 최고·유망 후보부터 세 번째 결과를 추가하고, 사용자 별도 제출은 동일 소스임이 확인된 경우에만 반복에 포함한다. 실패·명확한 회귀 후보를 일괄 재제출하지 않는다.

현재 최고는 E165/3fae5048, 98,125 / 43,492 / 285,067 cycle, 점수6.7575, 세 커널 PASS다. 원본 src/best-checkout/patches/e165에 반영했다. E161은 이전 최고로 보존한다. 최신 최고·제출 상태·우선순위는 docs/current-best.json, docs/ongoing-optimization-state.json, docs/optimization-priority.json을 우선한다.


2026-09-11 공식3회 결과 갱신: 같은 E165의 세 번째 006582e2가 98,359 / 40,066 / 282,925, 점수6.9569, 세 커널PASS로 최고다. 원본 소스는 동일 E165이며 3회 결과를 모두 보존한다. source 변경 없는 기록 차이를 구조적 개선으로 해석하지 않는다. 현 비교에서 FFN의 우위가 가장 작아 다음 우선대상은 FFN이다. 패치 docs/patches/e165-r3 및 current-best.json을 따른다.


2026-09-11 최신 최고: E172/339089eb, 100,571 / 38,873 / 284,644 cycle, 점수6.9613, 세 커널 PASS. 동일 소스 공식3회가 완료됐고 원본 src/best-checkout/patches/e172를 기준으로 한다. FFN의 경쟁 대비 우위가 −1.52%로 가장 작아 다음 우선 대상이다. E175 FFN의 8쌍 중앙값은 E165 대비 −1.03%, 이를 최고 O172와 결합한 E176을 실제 검증한다. 최신 상태는 docs/current-best.json과 ongoing-optimization-state.json을 먼저 읽는다.


2026-09-11 최신 공식 최고는 **E184/9442ca64=100029/38330/272870,score7.1060,세커널PASS**다. 원본src/best-checkout/patches/e184로유지한다. 공식동일소스3회,실제FFN교대8쌍전승/중앙값−5.91%. details docs/ffn-e184-analysis.md. 현재상대우선은QKV(+4.02%)이며이전최고값은이력이다. latest current-best.json/ongoing-optimization-state.json/optimization-priority.json을우선한다.


## 2026-09-11 최신 공식 최고: E211

공식 c0ffdd40:92,586 /39,406 /273,805 cycle,점수7.2164,세 커널PASS. 원본src 및 best-checkout은 이 소스와 일치한다. 후보마다 같은동결소스3회 제출하고 모든결과를보존한다. 종합1위에도 경쟁대비우위가가장작은커널부터반복개선한다. 최신근거는 [current-best](docs/current-best.md), [우선순위](docs/optimization-priority.md), [진행그림](docs/current-optimization-progress.png), [E211패치](docs/patches/e211/README.md)에서확인한다. 앞부분의 오래된 미검증체크리스트는2026-09-10당시기록이다.
