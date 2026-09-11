# 원본 src에 유지 중인 검증 소스

원본 `src/`와 `best-checkout/src/`에는 **E211**을 유지한다. 공식 제출 **c0ffdd40**가 세 커널 정확성 PASS, **92,586 / 39,406 / 273,805 cycle**, 점수 **7.2164**로 현재 최고다. 같은 소스 3회 점수는 6.8484 / 6.8883 / 7.2164이며 전부 보존한다.

직전 E184 대비 QKV 구현을 변경했다. O/FFN kernel image는 바이트 단위로 같으므로 그 cycle 차이는 실행 변동으로 구분한다. Arena job17309/17316의 교대8쌍에서 QKV는 E184 대비 중앙값−5.306%, 평균−5.458%,7/8승이다. 공개 및 기존 연속3조건을 포함한 총27개 프로세스에서 세 커널 PASS, cache 미갱신 영역6검사도 PASS다. 일반 BF16 입력40조건은 별도 host 검증이며 실제 연속fixture의 QKV 입력은 exact_rmsnorm_input이다.

[진행 그림](current-optimization-progress.png) · [반복 우선순위](optimization-priority.md) · [소스 checksum·백업·검증 기록](current-best.json) · [병합 후 적용할 E211 패치](patches/e211/README.md). 패치 적용과 SHA-256 일치를 임시 기준 소스에서 검증했고 Git index는 변경하지 않았다.
