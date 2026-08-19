# Phase 5: ONNX 모델 변환 및 검증 종합 결과 보고서

- **입력 해상도**: 640x640
- **ONNX OpSet 버전**: 17
- **벤치마크 디바이스**: CPU
- **측정 반복 횟수**: 10회 (워밍업 3회)

## 1. 모델별 변환 및 수치 동등성 검증 결과

| 실험    | 모델명                           | ONNX 파일                    |   ONNX 용량 (MB) |   최대 절대 오차 (Max Abs Diff) |   코사인 유사도 (Cosine Sim) | 수치 동등성 판정   |
|:------|:------------------------------|:---------------------------|---------------:|--------------------------:|-----------------------:|:------------|
| Exp 1 | YOLO11m Baseline              | yolo11m_baseline_best.onnx |          76.7  |                  0.000427 |                      1 | 합격          |
| Exp 2 | YOLO11m + CBAM                | yolo11m_cbam_best.onnx     |          77    |                  0.000259 |                      1 | 합격          |
| Exp 3 | YOLO11m + BiFPN               | yolo11m_bifpn_best.onnx    |          77.46 |                  0.000778 |                      1 | 합격          |
| Exp 4 | YOLO11m + CBAM + BiFPN (Full) | yolo11m_full_best.onnx     |          80.56 |                  0.000198 |                      1 | 합격          |

---

## 2. 추론 속도 (PyTorch vs ONNX Runtime) 비교

| 실험    | 모델명                           |   PyTorch Latency (ms) |   PyTorch FPS |   ORT Latency (ms) |   ORT FPS | 가속 비율 (Speedup)   |
|:------|:------------------------------|-----------------------:|--------------:|-------------------:|----------:|:------------------|
| Exp 1 | YOLO11m Baseline              |                 492.36 |           2   |             427.49 |       2.3 | 1.15x             |
| Exp 2 | YOLO11m + CBAM                |                 560.8  |           1.8 |             439.51 |       2.3 | 1.28x             |
| Exp 3 | YOLO11m + BiFPN               |                 491.99 |           2   |             404.34 |       2.5 | 1.22x             |
| Exp 4 | YOLO11m + CBAM + BiFPN (Full) |                1019.69 |           1   |            1050.32 |       1   | 0.97x             |

---

## 3. 핵심 분석 및 고찰

1. **커스텀 모듈의 무손실 ONNX 표준 변환 성공**:
   - CBAM Attention의 채널/공간 축소 및 결합 연산과 BiFPN의 Fast Normalized Fusion 동적 가중치 연산이 ONNX OpSet 17 표준 연산자로 완벽히 변환되었습니다.
   - 모든 모델의 코사인 유사도가 1.00000000을 기록하고 최대 절대 오차가 1e-3 이하로 측정되어 수치적 왜곡 없이 동일한 특징 표현력을 보존함을 검증했습니다.

2. **그래프 최적화 (onnxslim)**:
   - onnxslim을 통해 불필요한 노드 제거 및 상수 폴딩이 적용되어 ONNX 런타임 엔진에 최적화된 컴팩트 그래프를 생성했습니다.

3. **향후 확장성**:
   - 변환된 ONNX 모델은 C++, Rust, TensorRT, OpenVINO, WebAssembly 등 다양한 플랫폼에서 독립적으로 고속 로드 및 배포가 가능합니다.
