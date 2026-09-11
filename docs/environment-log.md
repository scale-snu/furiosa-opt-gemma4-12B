# 환경 정비 및 재빌드 기록

## 2026-09-10 — SDK와 컴파일러 선택 통일

- Git 기준: `e3e928070c1e87848771904cf5d63beecdaa7e09`. 시작 시 추적 파일 변경은 없었으며, `AGENTS.md`, PDF, `docs/`, 스케줄 생성 스크립트는 기존 미추적 파일이었다.
- 실행 경로: `target/environment/20260910-144441-rebuild.bRmsyW`.
- 시작 시 파일과 lockfile은 `before/`, Git 상태는 `git-status.txt`, 도구·Python·fixture 정보는 `environment.json`에 보관했다.

### 변경

`Cargo.toml`의 기존 `furiosa-opt-std = "=0.6.0"`을 유지하고, 새 `scripts/furiosa.sh`가 이 값을 읽어 같은 버전의 로컬 컴파일러를 설치·선택·검사하도록 했다. Rust는 `rust-toolchain.toml`의 `nightly-2026-05-01`을 명시적으로 선택한다. 설치된 전역 컴파일러는 0.7.0이므로 이 프로젝트의 NPU 명령은 공통 스크립트를 거친다.

`local_test.sh`, `generate_compiled_schedules.sh`, `run.sh`, `run_server.sh`를 연결했다. README, OPTIMIZATION, ARCHITECTURE 및 AGENTS의 관련 명령을 갱신했다. 커널 구현과 평가 코드는 수정하지 않았다.

작업 중 `rngd_test.sh`에도 공통 스크립트 호출과 `--build-only`를 추가했으나, 사용자가 기존 실행 흐름 유지를 요청해 이번 변경을 전부 되돌렸다. 작업 전 백업과 `cmp`로 일치하며 이 파일의 Git diff도 없다. 현재 스크립트에 `--build-only`는 없다. 빌드만 하려면 `./scripts/furiosa.sh test --release --test test_kernels --no-run`을 사용한다.

### 확인한 환경

| 구성 | 값 |
|---|---|
| Rust | `nightly-2026-05-01`, `rustc 1.97.0-nightly (f53b654a8 2026-04-30)` |
| Furiosa SDK 및 컴파일러 | `0.6.0`; Cargo.lock의 Furiosa 의존성도 모두 `0.6.0` |
| 호스트 | x86_64, GLIBC 2.39; gcc, clang, aarch64-linux-gnu-gcc 사용 가능 |
| CLI | cargo-binstall 1.23.0, furiosa-arena 0.8.0, cargo-generate 0.24.0 |
| Python | `/scale/cal/home/sehwan/miniconda3/envs/micro/bin/python3`, 3.14.7 |
| Python 패키지 | NumPy 2.5.3, PyTorch 2.14.0+cu130, safetensors 0.8.0 |
| Fixture | 기존 파일 재사용, 67,336 bytes, 텐서 38개 |

Fixture SHA-256: `9801c5732d9f5bef89d544631c37f74dae1603df35b67a2cfdd6377915b7599e`.
Python 패키지는 실제 import를 확인했다. Rust의 rustfmt·clippy·rust-src와 Schedule Viewer의 실행 옵션도 확인했다. 이미 준비된 도구와 fixture는 재설치·재생성하지 않았다.

### 재빌드와 검사

모든 명령은 저장소 루트에서 실행한다. 이번 새 빌드 디렉터리는 다음과 같다.

```bash
run_dir="target/environment/20260910-144441-rebuild.bRmsyW"
./scripts/furiosa.sh --install
CARGO_BUILD_JOBS=12 CARGO_TARGET_DIR="$PWD/$run_dir/build" \
    cargo +nightly-2026-05-01 build --release --all-targets --locked
env -u TUC_PROFILE_LEVEL CARGO_BUILD_JOBS=12 CARGO_TARGET_DIR="$PWD/$run_dir/build" \
    cargo +nightly-2026-05-01 test --release --test test_kernels --locked
CARGO_BUILD_JOBS=12 CARGO_TARGET_DIR="$PWD/$run_dir/build" \
    ./scripts/furiosa.sh build --release --all-targets --locked
```

- CPU 전체 타깃 빌드 성공: `cpu-build.log`, exit 0.
- CPU fixture 테스트: `cpu-test.log`, exit 101. `sliding_project_qkv`의 Q/K/V는 모두 통과했지만 다음 `sliding_attention_output` 경로에서 `furiosa-mapping-0.6.0/src/lib.rs:156`의 `carve: piece must be contained in self` panic으로 중단됐다. `decoder_feedforward` CPU 테스트는 실행되지 않았다. 이 결과를 NPU 정확도 검증으로 해석하지 않는다.
- 공통 스크립트 검사 6개 통과: 부모 셸의 Rust 버전과 무관하게 고정 nightly 선택, 설치된 도구 재사용, 상충하는 빌드 옵션 거절, 로컬 컴파일러 누락·버전 불일치·SDK 고정 누락 거절. `script-checks.json`과 사례별 로그에 기록했다.
- 새 디렉터리의 NPU 전체 빌드 성공: `npu-build.log`, exit 0. 15개 커널 `.bin`이 모두 생성됐으며 경로·크기·SHA-256은 `npu-kernels.json`에 기록했다.
- 기본 `target/`의 NPU 전체 빌드 성공: `default-npu-build.log`, exit 0. 12개 커널을 컴파일하고 기존 0.6.0 커널 3개를 재사용했다.
- 기본 경로의 테스트 바이너리 빌드 성공: `build-only.log`, exit 0. 작업 중 추가했던 `--build-only`를 Arena 호출 시 실패하는 대역과 함께 검사했으며 Arena를 호출하지 않았다. 이 옵션 자체는 이후 원복했으므로 현재 실행 명령으로 사용하지 않는다.
- 사용자가 보고한 원격 테스트 오류는 메시지를 요청한 상태다. 위 빌드 성공만으로 원격 테스트 오류가 해결됐다고 판단하지 않는다.

원본 로그·바이너리는 Git에서 제외되는 `target/`에 있다. 장기 보존하려면 위 실행 디렉터리를 별도로 보관한다. 새 검증 시에는 새 run 디렉터리를 사용하며, 동일한 의존성 재현이 필요하면 보관한 `before/Cargo.lock`도 함께 사용한다. 실제 RNGD 실행·정확도·cycle은 아직 검증하지 않았다.
