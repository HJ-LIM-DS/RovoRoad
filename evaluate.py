import os
import sys
import argparse
import pandas as pd
from pathlib import Path
import torch

# [중요] 커스텀 모듈(CBAM, BiFPN)을 먼저 불러와 동적 레지스트리에 등록합니다.
import model
from ultralytics import YOLO

# 유틸리티 함수 임포트
from utils.metrics import measure_model_fps_and_latency, generate_comparison_chart, format_summary_markdown

# [알림 연동] .agents 디렉터리의 slack_notifier 모듈을 불러옵니다.
sys.path.append(os.path.join(os.path.dirname(__file__), ".agents"))
try:
    from slack_notifier import send_slack_message
except ImportError:
    def send_slack_message(msg):
        print(f"[안내] slack_notifier 모듈을 찾을 수 없어 콘솔에 출력합니다: {msg}")

# WandB 임포트 및 API 키 로드 준비
import wandb

def load_wandb_api_key(key_file="WandB_API_KEY.txt"):
    """WandB API KEY 파일을 읽어옵니다."""
    path = Path(key_file)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None

def evaluate_single_model(model_name, weight_path, data_yaml="configs/hrp4k.yaml", split="test", device="0", imgsz=640, batch=16):
    """
    단일 모델 가중치에 대해 지정된 데이터 분할(test)에서 평가 및 속도를 측정합니다.
    
    매개변수:
        model_name (str): 모델 이름 (예: YOLO11m-Baseline)
        weight_path (str): .pt 가중치 파일 경로
        data_yaml (str): 데이터셋 YAML 파일 경로
        split (str): 평가할 데이터 분할 ('test' 또는 'val')
        device (str): 실행 디바이스
        imgsz (int): 이미지 해상도
        batch (int): 배치 사이즈
        
    반환값:
        dict: 평가 지표 딕셔너리
    """
    weight_file = Path(weight_path)
    if not weight_file.exists():
        print(f"[오류] 가중치 파일을 찾을 수 없습니다: {weight_file.resolve()}")
        return None

    print(f"\n=======================================================")
    print(f"  [모델 평가 시작] {model_name}")
    print(f"  가중치 파일: {weight_path} | 분할: {split}")
    print(f"=======================================================\n")

    # 1. YOLO 모델 로드
    yolo_model = YOLO(str(weight_file))
    
    # 2. Ultralytics val 모드로 검증 수행
    val_results = yolo_model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        save_json=False,
        save_hybrid=False,
        plots=True,
        verbose=True
    )
    
    # 3. 메트릭 파싱
    metrics_dict = val_results.results_dict if hasattr(val_results, 'results_dict') else {}
    precision = float(metrics_dict.get('metrics/precision(B)', 0.0))
    recall = float(metrics_dict.get('metrics/recall(B)', 0.0))
    map50 = float(metrics_dict.get('metrics/mAP50(B)', 0.0))
    map50_95 = float(metrics_dict.get('metrics/mAP50-95(B)', 0.0))

    # 4. 파라미터 수 및 연산량(FLOPs) 계산
    params_m = 0.0
    gflops = 0.0
    if hasattr(yolo_model.model, "parameters"):
        params_count = sum(p.numel() for p in yolo_model.model.parameters())
        params_m = round(params_count / 1e6, 2)
        
    # Ultralytics speed 및 FLOPs 정보 추출
    if hasattr(val_results, "speed"):
        speed_info = val_results.speed
        # inference time ms
        inf_time = speed_info.get("inference", 0.0)
    else:
        inf_time = 0.0

    # 5. 정밀 FPS 및 Latency 측정 (PyTorch GPU 동기화 방식)
    print("[안내] 정밀 추론 지연시간(Latency) 및 FPS 측정을 수행합니다...")
    speed_result = measure_model_fps_and_latency(
        model=yolo_model,
        sample_tensor=torch.randn(1, 3, imgsz, imgsz),
        device=device,
        num_warmup=15,
        num_repeats=50
    )

    result_summary = {
        "name": model_name,
        "weight_path": str(weight_path),
        "params_m": params_m,
        "gflops": gflops,
        "precision": precision,
        "recall": recall,
        "mAP50": map50,
        "mAP50-95": map50_95,
        "latency_ms": speed_result["latency_ms"],
        "fps": speed_result["fps"],
        "std_ms": speed_result["std_ms"]
    }
    
    print(f"[{model_name} 평가 완료] mAP@50: {map50:.4f}, mAP@50-95: {map50_95:.4f}, FPS: {speed_result['fps']} fps ({speed_result['latency_ms']} ms)")
    return result_summary

def run_ablation_evaluation(args):
    """
    4가지 Ablation Study 모델에 대해 종합 Test 셋 평가 및 벤치마크를 수행합니다.
    """
    models_to_eval = [
        {"name": "YOLO11m-Baseline", "weight": "weights/yolo11m_baseline_best.pt", "gflops": 67.8},
        {"name": "YOLO11m-CBAM",     "weight": "weights/yolo11m_cbam_best.pt",     "gflops": 68.0},
        {"name": "YOLO11m-BiFPN",    "weight": "weights/yolo11m_bifpn_best.pt",    "gflops": 67.6},
        {"name": "YOLO11m-Full",     "weight": "weights/yolo11m_full_best.pt",     "gflops": 140.0}
    ]

    all_results = []
    
    for m in models_to_eval:
        res = evaluate_single_model(
            model_name=m["name"],
            weight_path=m["weight"],
            data_yaml=args.data,
            split=args.split,
            device=args.device,
            imgsz=args.imgsz,
            batch=args.batch
        )
        if res:
            # 사전 계산된 FLOPs 보정
            res["gflops"] = m["gflops"]
            all_results.append(res)

    if not all_results:
        print("[오류] 평가된 모델 결과가 없습니다.")
        return

    # 1. 콘솔 비교 테이블 출력
    df_results = pd.DataFrame(all_results)
    print("\n" + "=" * 90)
    print(f"                  [ROVOROAD HRP4K {args.split.upper()} 셋 종합 벤치마크 평가 결과]")
    print("=" * 90)
    print(df_results[["name", "params_m", "gflops", "precision", "recall", "mAP50", "mAP50-95", "latency_ms", "fps"]].to_string(index=False))
    print("=" * 90 + "\n")

    # 2. 결과 파일 저장 (Markdown & CSV)
    runs_dir = Path("runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = runs_dir / "evaluation_summary.csv"
    df_results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[저장 완료] CSV 요약 보고서: {csv_path.resolve()}")
    
    md_content = format_summary_markdown(all_results)
    md_path = runs_dir / "evaluation_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[저장 완료] Markdown 요약 보고서: {md_path.resolve()}")

    # 3. 비교 차트 시각화 이미지 생성
    chart_path = runs_dir / "evaluation_comparison.png"
    generate_comparison_chart(all_results, save_path=str(chart_path))

    # 4. WandB에 Test 셋 평가 벤치마크 업로드
    api_key = load_wandb_api_key("WandB_API_KEY.txt")
    if api_key and args.upload_wandb:
        try:
            print("\n[WandB 업로드] Test 셋 종합 벤치마크 결과를 WandB에 기록합니다...")
            wandb.login(key=api_key)
            eval_run = wandb.init(
                project="rovoroad-pothole",
                name="test-set-benchmark-evaluation",
                reinit=True,
                tags=["evaluation", "test-set", "benchmark"]
            )
            
            # WandB Table 등록
            wandb_cols = ["Model", "Params (M)", "GFLOPs", "Precision", "Recall", "mAP@50", "mAP@50-95", "Latency (ms)", "FPS"]
            wandb_data = [
                [
                    r["name"],
                    r["params_m"],
                    r["gflops"],
                    round(r["precision"], 4),
                    round(r["recall"], 4),
                    round(r["mAP50"], 4),
                    round(r["mAP50-95"], 4),
                    round(r["latency_ms"], 2),
                    round(r["fps"], 1)
                ]
                for r in all_results
            ]
            eval_table = wandb.Table(columns=wandb_cols, data=wandb_data)
            wandb.log({
                "Test_Set_Benchmark_Table": eval_table,
                "charts/evaluation_comparison": wandb.Image(str(chart_path), caption="Ablation Benchmark Comparison")
            })
            eval_url = eval_run.get_url()
            eval_run.finish()
            print(f"[완료] WandB 평가 결과 등록 완료: {eval_url}")
        except Exception as e:
            print(f"[경고] WandB 평가 결과 업로드 중 오류 발생: {e}")

    # 5. 슬랙 알림 발송
    try:
        best_map_model = max(all_results, key=lambda x: x["mAP50-95"])
        fastest_model = max(all_results, key=lambda x: x["fps"])
        
        slack_msg = (
            f"[ROVOROAD 포트홀 탐지] Phase 4: Test 셋 모델 평가 완료 보고\n"
            f"- 평가 대상: 4개 Ablation 모델 (Baseline, CBAM, BiFPN, Full)\n"
            f"- 데이터 분할: HRP4K {args.split.upper()} 셋\n"
            f"- 최고 정밀도 모델(mAP@50-95): {best_map_model['name']} (mAP50: {best_map_model['mAP50']:.4f}, mAP50-95: {best_map_model['mAP50-95']:.4f})\n"
            f"- 최고 속도 모델(FPS): {fastest_model['name']} ({fastest_model['fps']:.1f} FPS, {fastest_model['latency_ms']:.2f} ms)\n"
            f"- 종합 비교 차트 및 보고서: runs/evaluation_summary.md 저장 완료"
        )
        send_slack_message(slack_msg)
        print("[완료] 슬랙 채널로 평가 완료 보고 메시지를 전송했습니다.")
    except Exception as e:
        print(f"[경고] 슬랙 알림 전송 중 오류 발생: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROVOROAD Phase 4 - 포트홀 탐지 모델 종합 평가 및 벤치마크 시스템")
    parser.add_argument("--data", type=str, default="configs/hrp4k.yaml", help="데이터셋 설정 YAML 파일")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"], help="평가할 데이터 분할")
    parser.add_argument("--device", type=str, default="0", help="평가에 사용할 GPU 디바이스 (0 또는 cpu)")
    parser.add_argument("--imgsz", type=int, default=640, help="입력 이미지 해상도")
    parser.add_argument("--batch", type=int, default=16, help="배치 크기")
    parser.add_argument("--upload_wandb", action="store_true", default=True, help="WandB 클라우드에 결과 업로드 여부")
    
    args = parser.parse_args()
    
    print("[시스템] Phase 4: 모델 종합 평가 및 벤치마크를 시작합니다.")
    run_ablation_evaluation(args)
