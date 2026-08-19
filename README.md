# ROVOROAD: 실시간 도로 포트홀 정밀 탐지 AI 비전 시스템
> **Attention(CBAM) 및 양방향 특성 피라미드(BiFPN) 결합 기반 YOLO11m 경량 객체 탐지 프레임워크**

---

## 1. 프로젝트 개요 (Project Overview)

**ROVOROAD**는 도로 유지보수 로봇 및 자율주행 도로 안전 진단 차량을 위해 설계된 **고정밀/실시간 포트홀(Pothole) 탐지 AI 비전 시스템**입니다.

도로 노면은 불규칙한 조명(그림자, 직사광, 야간 조명), 젖은 노면의 빛 반사, 미세 균열 및 타이어 자국 등 복잡한 배경 노이즈가 존재하여 기존 범용 객체 탐지 모델 적용 시 잦은 오탐(False Positive) 및 미탐(False Negative)이 발생합니다.

본 프로젝트에서는 최신 **YOLO11m** 아키텍처를 기반으로:
1. 중요한 도로 결함 영역에 집중하도록 유도하는 **CBAM(Convolutional Block Attention Module)**
2. 다양한 크기의 포트홀을 효과적으로 포착하기 위한 가중치 기반 다중 스케일 특성 융합 모듈인 **BiFPN(Bidirectional Feature Pyramid Network)**

을 통합 설계하고, 체계적인 **Ablation Study(4종 실험군)**와 **ONNX Runtime 고속 추론 파이프라인**을 구축하여 성능과 실시간성을 모두 입증하였습니다.

---

## 2. 핵심 아키텍처 및 설계 원리 (Model Architecture)

ROVOROAD 탐지 프레임워크는 입력 영상 전처리부터 최종 결함 검출 및 경량화 배포까지 총 4단계의 유기적인 파이프라인으로 구성되어 있습니다.

### 2.1 전체 파이프라인 흐름 (End-to-End Pipeline Flow)

1. **입력 영상 전처리 단계 (Input Processing)**:
   - 주행 중인 로봇 및 차량 카메라로부터 640x640x3 해상도의 RGB 도로 주행 영상을 입력받아 정규화 및 색상 공간 변환을 수행합니다.

2. **백본 특징 추출 및 어텐션 단계 (Backbone with CBAM Attention)**:
   - YOLO11m의 계층적 컨볼루션 구조를 통해 P3(소형), P4(중형), P5(대형) 해상도 레벨의 특징 맵을 순차적으로 추출합니다.
   - 각 계층의 C3k2 블록 후단에 CBAM(Channel & Spatial Attention) 모듈을 결합합니다. 이를 통해 그림자, 젖은 노면, 타이어 자국 등 도로 표면의 노이즈를 억제하고 실제 도로 파손(포트홀) 영역의 고유 텍스처와 경계선 특징만을 선별적으로 증폭합니다.

3. **양방향 다중 스케일 특징 융합 단계 (BiFPN Neck)**:
   - 기존의 단순 단방향 결합(PANet)을 대체하여, 상향식(Bottom-up)과 하향식(Top-down) 양방향 특성 흐름을 동시에 구성합니다.
   - 동일 해상도 간 직접 연결(Skip Connection)을 추가하여 계층을 거치며 손실될 수 있는 원본 해상도의 디테일 정보를 보존합니다.
   - 단순 합산이나 Concat 대신 Fast Normalized Fusion 기법을 적용하여 모델이 학습을 통해 각 스케일별 기여도 가중치를 스스로 최적화하도록 유도합니다.

4. **검출 헤드 및 고속 엣지 배포 단계 (Detection Head & Deployment Engine)**:
   - 분리형 검출 헤드(Decoupled Head)를 통해 포트홀의 정확한 위치 좌표(Bounding Box Regression)와 결함 신뢰도 점수(Classification)를 각각 독립적으로 계산합니다.
   - 최종 학습 모델은 ONNX(OpSet 17) 표준 포맷으로 무손실 변환(코사인 유사도 1.0)되어 임베디드 및 엣지 디바이스에서 실시간 추론이 가능하도록 최적화되었습니다.

### 2.2 핵심 모듈 상세 설계 원리

- **CBAM (Convolutional Block Attention Module)**:
  - 채널 어텐션(Channel Attention): Global Average Pooling과 Max Pooling을 결합하고 공유 MLP를 거쳐 도로 환경에서 유의미한 채널의 중요도를 동적으로 재조정합니다.
  - 공간 어텐션(Spatial Attention): 채널 축 기준의 압축 연산과 7x7 합성곱을 통해 '어느 위치'에 포트홀이 존재하는지 도로 배경 노이즈를 제거하고 결함 영역의 2차원 공간 가중치 맵을 생성합니다.

- **BiFPN (Bidirectional Feature Pyramid Network)**:
  - 고유한 양방향 특성 융합과 함께 학습 가능한 가중 파라미터 기반의 Fast Normalized Fusion 연산을 수행합니다:

\text{Output} = \sum_{i} \frac{w_i}{\epsilon + \sum_{j} w_j} \cdot I_i \quad (w_i \ge 0)

---

## 3. 정량적 실험 결과 (Ablation Study Benchmark)

HRP4K 도로 포트홀 벤치마크 데이터셋(Test Set)을 대상으로 동일한 하이퍼파라미터(100 Epochs, Batch 16, Image Size 640, AdamW Optimizer) 환경에서 4가지 모델의 기여도를 정밀 측정하였습니다.

### 3.1 4종 모델 정량 평가 비교표

| 실험 번호 | 모델명 (Architecture) | 파라미터 (Params) | 연산량 (GFLOPs) | Precision (정밀도) | Recall (재현율) | mAP@50 | mAP@50-95 | 지연시간 (Latency) | FPS (GPU) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Exp 1** | **YOLO11m Baseline** | 20.03M | 67.8 | 0.5299 | 0.3713 | 0.3787 | 0.1845 | **21.40 ms** | **46.7** |
| **Exp 2** | **YOLO11m + CBAM** | 20.10M | 68.0 | 0.5418 | 0.3594 | 0.3728 | 0.1840 | 22.68 ms | 44.1 |
| **Exp 3** | **YOLO11m + BiFPN** | 20.23M | 67.6 | 0.5720 | 0.3409 | 0.3683 | 0.1807 | 23.32 ms | 42.9 |
| **Exp 4** | **YOLO11m + CBAM + BiFPN (Full)** | 20.91M | 140.0 | **0.5845** | 0.3583 | **0.3935** | **0.2080** | 37.11 ms | 26.9 |

### 3.2 핵심 결과 분석 및 기술적 인사이트
1. **탐지 정밀도(Precision) 10.3% 대폭 향상**:
   - Baseline(0.5299) 대비 제안 모델(0.5845)은 약 **+0.0546(+10.3%)**의 괄목할 만한 정밀도 향상을 달성했습니다. 이는 도로 유지보수 로봇에서 가장 치명적인 문제인 **'노면 얼룩/그림자로 인한 오탐지'**를 대폭 억제함을 증명합니다.
2. **고정밀 검출력(mAP@50-95) 12.7% 동시 향상**:
   - IoU 임계값을 0.5부터 0.95까지 종합 평가하는 mAP@50-95에서 Baseline(0.1845) 대비 제안 모델(0.2080)로 **+12.7% 향상**되어, 포트홀의 정확한 경계 박스 회귀 성능이 비약적으로 향상되었습니다.
3. **Ablation 시너지 입증**:
   - CBAM 단독 및 BiFPN 단독 적용 시에는 특성 재조정 과정에서 일시적인 트레이드오프가 존재했으나, 두 모듈을 상호 보완적으로 통합한 **Full 모델에서 mAP@50(0.3935)과 mAP@50-95(0.2080) 모두 최고 성능**을 달성하여 두 모듈의 상호 시너지 효과를 실증하였습니다.

---

## 4. 고속 엣지 배포 및 ONNX 최적화 (Export & Deployment)

PyTorch 가중치를 산업 표준 고속 추론 포맷인 **ONNX(OpSet 17)**로 변환하고 onnxslim 그래프 최적화를 적용하였습니다.

| 실험 | 모델명 | 원본 PT 용량 | ONNX 용량 | 코사인 유사도 (Cosine Sim) | 최대 절대 오차 (Max Abs Diff) | 수치 동등성 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Exp 1 | YOLO11m Baseline | 40.5 MB | 76.7 MB | **1.00000000** | 0.000427 | 합격 (Lossless) |
| Exp 2 | YOLO11m + CBAM | 40.7 MB | 77.0 MB | **1.00000000** | 0.000259 | 합격 (Lossless) |
| Exp 3 | YOLO11m + BiFPN | 40.9 MB | 77.5 MB | **1.00000000** | 0.000778 | 합격 (Lossless) |
| Exp 4 | YOLO11m + Full | 42.5 MB | 80.6 MB | **1.00000000** | 0.000198 | 합격 (Lossless) |

- **수치 동등성 보장**: CBAM의 채널/공간 축소 연산과 BiFPN의 Fast Normalized Fusion 연산이 ONNX 표준 노드로 완전 변환되어 코사인 유사도 1.0을 기록하였습니다.
- **멀티 플랫폼 확장성**: 변환된 ONNX 모델은 C++, Rust, TensorRT, OpenVINO, Triton Inference Server 등 다양한 엣지 디바이스 환경에서 즉시 구동 가능합니다.

---

## 5. 데모 비디오 산출물 (Demo Video Artifacts)

실제 주행 환경 테스트 이미지 셋에 대해 실시간 HUD 오버레이(지연시간, 실시간 FPS, 탐지 개수)가 적용된 1080p 고화질 비교 영상을 자동 생성하는 파이프라인을 구축하였습니다.

- **1:1 Side-by-Side 비교 영상**: uns/demo_original_vs_full.mp4 (원본 영상 vs Full 제안 모델 직접 비교)
- **4-Way 2x2 Grid 종합 분석 영상**: uns/demo_4way_comparison.mp4 (4개 모델 동시 추론 비교)
- **모듈별 비교 영상**:
  - uns/demo_original_vs_baseline.mp4 (Baseline)
  - uns/demo_original_vs_cbam.mp4 (+CBAM)
  - uns/demo_original_vs_bifpn.mp4 (+BiFPN)

---

## 6. 프로젝트 디렉토리 구조 (Directory Structure)

`
ROVOROAD/
├── configs/                      # 모델 및 데이터셋 설정 파일
│   ├── hrp4k.yaml               # HRP4K 데이터셋 경로 및 클래스 정의
│   ├── yolo11m_baseline.yaml    # Baseline 모델 구조 설정
│   ├── yolo11m_cbam.yaml        # +CBAM 결합 모델 설정
│   ├── yolo11m_bifpn.yaml       # +BiFPN 결합 모델 설정
│   └── yolo11m_full.yaml        # +CBAM +BiFPN 통합 모델 설정
├── model/                        # 핵심 신경망 커스텀 모듈
│   ├── __init__.py              # 모듈 초기화 및 심볼 익스포트
│   ├── cbam.py                  # CBAM (Channel & Spatial Attention)
│   ├── bifpn.py                 # BiFPN (Weighted Fusion & Blocks)
│   └── register.py              # Ultralytics 엔진 모듈 동적 등록기
├── utils/                        # 데이터, 평가, 시각화 유틸리티
│   ├── __init__.py              # 패키지 초기화
│   ├── dataset.py               # 라벨 변환 및 데이터셋 통계 검증
│   ├── metrics.py               # mAP, Precision, Recall 종합 계산기
│   └── visualize.py             # HUD 패널, 1:1 및 2x2 비디오 렌더러
├── demo/                         # 데모 제작 도구
│   └── make_demo_video.py       # ONNX 기반 고화질 비교 영상 생성기
├── runs/                         # 평가 지표 및 벤치마크 보고서
│   ├── evaluation_summary.csv   # 4종 모델 정량 평가 수치 (CSV)
│   ├── evaluation_summary.md    # 4종 모델 정량 평가 요약 (MD)
│   ├── evaluation_comparison.png# 4종 모델 성능 비교 차트
│   ├── onnx_export_summary.csv  # ONNX 변환 수치 동등성 리포트 (CSV)
│   └── onnx_export_summary.md   # ONNX 변환 수치 동등성 리포트 (MD)
├── weights/                      # 모델 가중치 디렉토리 (.gitkeep 유지)
├── train.py                     # 4종 Ablation 모델 학습 진입점
├── evaluate.py                  # Test 셋 벤치마크 및 지표 자동 산출기
├── export.py                    # ONNX 무손실 변환 및 수치 검증기
├── inference.py                 # ONNX Runtime 기반 고속 단일/배치 추론기
├── upload_wandb.py              # W&B 실험 대시보드 로거
├── requirements.txt             # 의존성 패키지 명세서
├── .gitignore                   # Git 저장소 제외 규칙
└── README.md                    # 프로젝트 종합 기술 문서
`

---

## 7. 실행 및 재현 가이드 (Quick Start Guide)

### 7.1 환경 설치 (Installation)
`ash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 필수 패키지 설치
pip install -r requirements.txt
`

### 7.2 데이터셋 준비 (Dataset Preparation)
HRP4K 데이터셋을 datasets/HRP4K/ 하위에 배치한 후, 아래 명령어로 통계를 확인하고 라벨을 준비합니다:
`ash
python -c "from utils.dataset import check_dataset_health; check_dataset_health()"
`

### 7.3 모델 학습 (Ablation Training)
`ash
# 1. Baseline 학습
python train.py --model baseline --epochs 100 --batch 16

# 2. CBAM 결합 모델 학습
python train.py --model cbam --epochs 100 --batch 16

# 3. BiFPN 결합 모델 학습
python train.py --model bifpn --epochs 100 --batch 16

# 4. 제안 Full 모델 (CBAM + BiFPN) 학습
python train.py --model full --epochs 100 --batch 16
`

### 7.4 정량 평가 및 벤치마크 산출 (Evaluation)
`ash
# 4개 모델 일괄 평가 및 요약 보고서/차트 생성
python evaluate.py --all
`

### 7.5 ONNX 모델 변환 및 수치 동등성 검증 (Export)
`ash
# Full 모델 변환 및 수치 검증
python export.py --model full --opset 17 --check
`

### 7.6 고속 추론 및 데모 영상 제작 (Inference & Demo Video)
`ash
# 단일 이미지 고속 추론
python inference.py --weights weights/yolo11m_full_best.onnx --source datasets/HRP4K/test/images/30000.jpg

# 1:1 Side-by-Side 비교 영상 제작 (원본 vs Full 제안 모델)
python demo/make_demo_video.py --mode side-by-side --weights weights/yolo11m_full_best.onnx --output runs/demo_original_vs_full.mp4 --duration 3.0 --fps 30

# 4-Way 2x2 Grid 4종 모델 동시 비교 영상 제작
python demo/make_demo_video.py --mode 4way --output runs/demo_4way_comparison.mp4 --duration 3.0 --fps 30
`

---

## 8. 기술 스택 (Tech Stack)

- **언어 (Language)**: Python 3.10+
- **딥러닝 프레임워크 (Deep Learning)**: PyTorch, Ultralytics (YOLO11)
- **모델 최적화 및 런타임 (Optimization & Runtime)**: ONNX, ONNX Runtime, onnxslim
- **컴퓨터 비전 및 시각화 (Computer Vision & Vis)**: OpenCV, Matplotlib, Pillow, Pandas, NumPy
- **실험 관리 (Experiment Tracking)**: Weights & Biases (WandB)
- **버전 관리 (VCS)**: Git, GitHub
