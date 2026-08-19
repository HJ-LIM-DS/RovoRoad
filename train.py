import os
import sys
import argparse
import shutil
from pathlib import Path

# [중요] 커스텀 모듈(CBAM, BiFPN)을 먼저 불러와 동적 레지스트리에 등록합니다.
import model
from ultralytics import YOLO

# [알림 연동] .agents 디렉터리의 slack_notifier 모듈을 불러옵니다.
sys.path.append(os.path.join(os.path.dirname(__file__), ".agents"))
try:
    from slack_notifier import send_slack_message
except ImportError:
    # 모듈이 없을 경우 안전하게 대체 함수를 정의합니다.
    def send_slack_message(msg):
        print(f"[안내] slack_notifier 모듈을 찾을 수 없어 콘솔에 출력합니다: {msg}")

def train_model(exp_name, args):
    """
    특정 설정(exp_name)으로 YOLO11m 기반 포트홀 탐지 모델을 학습하는 함수입니다.
    학습 완료 후 검증 메트릭을 추출하고 슬랙으로 알림을 전송합니다.
    """
    # 1. 환경 변수로 WandB(Weights & Biases) 로깅 모드 설정 
    # 기본적으로 오프라인(offline) 모드로 설정되어 로컬 환경에 로그를 안전하게 기록합니다.
    os.environ["WANDB_MODE"] = args.wandb_mode
    
    # 2. 실험 이름에 따른 YAML 모델 아키텍처 설정 파일 매핑
    configs = {
        "baseline": "configs/yolo11m_baseline.yaml",
        "cbam":     "configs/yolo11m_cbam.yaml",
        "bifpn":    "configs/yolo11m_bifpn.yaml",
        "full":     "configs/yolo11m_full.yaml"
    }
    
    cfg_path = configs.get(exp_name)
    if not cfg_path:
        print(f"[오류] 알 수 없는 실험 이름입니다: {exp_name}")
        return

    print(f"\n=======================================================")
    print(f"  [학습 시작] 실험명: {exp_name} | 설정 파일: {cfg_path}")
    print(f"=======================================================\n")
    
    # 3. 모델 초기화 (YAML 아키텍처 로드 및 전이 학습)
    base_weight = "yolo11m.pt"
    
    try:
        net = YOLO(cfg_path).load(base_weight)
        print("[안내] 기존 yolo11m 사전학습 가중치를 로드하여 전이 학습을 적용합니다.")
    except Exception as e:
        print(f"[안내] 사전학습 가중치 로드 실패 시 초기화된 가중치로 학습합니다. 사유: {e}")
        net = YOLO(cfg_path)

    # 4. 모델 학습(Train) 실행
    results = net.train(
        data="configs/hrp4k.yaml",   # 데이터셋 설정 파일 경로
        epochs=args.epochs,          # 학습 데이터 반복 횟수
        batch=args.batch,            # 배치 크기
        imgsz=args.imgsz,            # 이미지 해상도 (640)
        device=args.device,          # GPU 디바이스 번호
        workers=args.workers,        # 데이터 로딩 워커 수
        optimizer="AdamW",           # 옵티마이저
        lr0=args.lr0,                # 시작 학습률
        project=os.path.join(os.getcwd(), "runs", "rovoroad-pothole"),  # 저장 폴더
        name=f"yolo11m-{exp_name}",  # 실행 단위 이름
        exist_ok=True,               # 덮어쓰기 허용
        mosaic=1.0,                  # Mosaic 증강
        mixup=0.1                    # MixUp 증강
    )
    
    print(f"\n[{exp_name}] 모델 학습이 완료되었습니다.")
    
    # 5. 최적 성능 베스트 가중치(best.pt) 파일 복사 및 보관
    best_weight_path = Path(os.path.join(os.getcwd(), "runs", "rovoroad-pothole", f"yolo11m-{exp_name}", "weights", "best.pt"))
    target_weight_path = Path(f"weights/yolo11m_{exp_name}_best.pt")
    
    weight_saved = False
    if best_weight_path.exists():
        shutil.copy(best_weight_path, target_weight_path)
        print(f"[완료] 최고 성능 모델 가중치가 저장되었습니다: {target_weight_path}")
        weight_saved = True
    else:
        print(f"[경고] 학습이 완료되었으나 베스트 가중치를 찾을 수 없습니다: {best_weight_path}")

    # 6. 학습 결과 메트릭 추출 및 슬랙 알림 전송
    try:
        # results 객체에서 주요 지표 추출
        metrics_dict = {}
        if hasattr(results, 'results_dict') and results.results_dict:
            metrics_dict = results.results_dict
        elif hasattr(net, 'metrics') and net.metrics:
            metrics_dict = net.metrics.results_dict if hasattr(net.metrics, 'results_dict') else {}

        # 메트릭 파싱
        precision = metrics_dict.get('metrics/precision(B)', 0.0)
        recall = metrics_dict.get('metrics/recall(B)', 0.0)
        map50 = metrics_dict.get('metrics/mAP50(B)', 0.0)
        map50_95 = metrics_dict.get('metrics/mAP50-95(B)', 0.0)

        # 슬랙 메시지 구성
        slack_text = (
            f"[ROVOROAD 포트홀 탐지] 모델 학습 완료 보고\n"
            f"- 실험 모델: YOLO11m ({exp_name})\n"
            f"- 학습 에폭: {args.epochs} epochs\n"
            f"- 정밀도 (Precision): {precision:.4f}\n"
            f"- 재현율 (Recall): {recall:.4f}\n"
            f"- mAP@50: {map50:.4f}\n"
            f"- mAP@50-95: {map50_95:.4f}\n"
            f"- 베스트 가중치 저장 상태: {'성공 (' + str(target_weight_path) + ')' if weight_saved else '실패'}\n"
            f"- WandB 로깅 모드: {args.wandb_mode}"
        )

        send_slack_message(slack_text)
        print("[안내] 슬랙 알림 메시지 전송이 완료되었습니다.")
    except Exception as e:
        print(f"[경고] 슬랙 알림 전송 중 오류가 발생했으나 학습은 정상 완료되었습니다: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROVOROAD Phase 3 - 포트홀 탐지 커스텀 모델 학습 시스템")
    
    parser.add_argument("--exp", type=str, default="baseline", 
                        choices=["baseline", "cbam", "bifpn", "full", "compare", "all"],
                        help="실행할 실험 이름을 지정합니다. 'compare'는 비교 3개(baseline, cbam, bifpn) 실행, 'all'은 4가지 전체 실행.")
    parser.add_argument("--epochs", type=int, default=100, help="전체 데이터셋 반복 학습 횟수 (기본 100)")
    parser.add_argument("--batch", type=int, default=16, help="한 묶음당 이미지 수 (기본 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="입력 이미지 해상도 (기본 640)")
    parser.add_argument("--device", type=str, default="0", help="학습에 활용할 GPU 번호 (예: 0, 1 또는 cpu)")
    parser.add_argument("--workers", type=int, default=4, help="데이터 로더의 병렬 처리 워커 개수")
    parser.add_argument("--lr0", type=float, default=0.001, help="시작 학습률 (Learning Rate)")
    parser.add_argument("--wandb_mode", type=str, default="offline", 
                        choices=["online", "offline", "disabled"],
                        help="WandB 로깅 방식 설정 (기본값: offline으로 네트워크 없이 로컬 저장)")
    
    args = parser.parse_args()
    
    if args.exp == "all":
        experiments = ["baseline", "cbam", "bifpn", "full"]
        print(f"[전체 모드] 4가지 Ablation Study 연속 학습 파이프라인을 가동합니다: {experiments}")
        for exp in experiments:
            train_model(exp, args)
    elif args.exp == "compare":
        experiments = ["baseline", "cbam", "bifpn"]
        print(f"[비교 모델 모드] 3가지 비교 모델 연속 학습 파이프라인을 가동합니다: {experiments}")
        for exp in experiments:
            train_model(exp, args)
    else:
        train_model(args.exp, args)
