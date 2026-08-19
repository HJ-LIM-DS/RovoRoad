import os
import sys
import yaml
import pandas as pd
from pathlib import Path
import wandb

def get_wandb_api_key(key_file_path="WandB_API_KEY.txt"):
    """
    지정된 텍스트 파일 경로에서 WandB API KEY를 안전하게 읽어오는 함수입니다.
    
    매개변수:
        key_file_path (str): API 키가 저장된 텍스트 파일 경로
        
    반환값:
        str: 공백이 제거된 WandB API 키 문자열
    """
    full_path = Path(key_file_path)
    if not full_path.exists():
        raise FileNotFoundError(f"[오류] WandB API 키 파일을 찾을 수 없습니다: {full_path.resolve()}")
    
    with open(full_path, "r", encoding="utf-8") as f:
        api_key = f.read().strip()
        
    if not api_key:
        raise ValueError(f"[오류] WandB API 키 파일이 비어 있습니다: {full_path.resolve()}")
        
    return api_key

def upload_run_to_wandb(run_dir, project_name="rovoroad-pothole"):
    """
    개별 실험 폴더(runs/rovoroad-pothole/<실험명>)의 학습 로그와 결과 이미지를 WandB에 업로드하는 함수입니다.
    
    매개변수:
        run_dir (Path): 개별 실험 디렉터리 경로
        project_name (str): WandB 프로젝트 이름
    """
    run_name = run_dir.name
    print(f"\n=======================================================")
    print(f"  [WandB 업로드 시작] 실험명: {run_name}")
    print(f"  디렉터리 경로: {run_dir.resolve()}")
    print(f"=======================================================\n")
    
    # 1. args.yaml 설정 파일 로드 (하이퍼파라미터 메타데이터)
    args_file = run_dir / "args.yaml"
    config_dict = {}
    if args_file.exists():
        with open(args_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}
        print(f"[안내] {args_file.name} 하이퍼파라미터 설정 파일을 성공적으로 로드했습니다.")
    else:
        print(f"[경고] {args_file.name} 파일이 없어 기본 메타데이터로 진행합니다.")

    # 2. WandB Run 초기화
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config_dict,
        reinit=True,
        tags=["pothole-detection", "ablation-study", "yolo11m"]
    )
    
    # 3. results.csv 학습 로그 파싱 및 Step별 지표 로깅
    results_csv_file = run_dir / "results.csv"
    best_summary = {}
    
    if results_csv_file.exists():
        df = pd.read_csv(results_csv_file)
        # CSV 컬럼명의 앞뒤 불필요한 공백 제거
        df.columns = [col.strip() for col in df.columns]
        
        print(f"[안내] 총 {len(df)}개 에폭의 학습 로그를 WandB에 순차 기록합니다...")
        for _, row in df.iterrows():
            epoch = int(row.get("epoch", 0))
            log_data = {}
            for col in df.columns:
                val = row[col]
                # 숫자인 경우 float 변환하여 기록
                if pd.notnull(val):
                    try:
                        log_data[col] = float(val)
                    except (ValueError, TypeError):
                        log_data[col] = val
            
            # WandB에 에폭 단위로 지표 기록
            wandb.log(log_data, step=epoch)
            
        # 최고 성능 요약 지표 산출
        if "metrics/mAP50-95(B)" in df.columns:
            best_idx = df["metrics/mAP50-95(B)"].idxmax()
            best_row = df.loc[best_idx]
            best_summary = {
                "best_epoch": int(best_row.get("epoch", 0)),
                "best_mAP50": float(best_row.get("metrics/mAP50(B)", 0.0)),
                "best_mAP50-95": float(best_row.get("metrics/mAP50-95(B)", 0.0)),
                "best_precision": float(best_row.get("metrics/precision(B)", 0.0)),
                "best_recall": float(best_row.get("metrics/recall(B)", 0.0)),
                "final_val_box_loss": float(best_row.get("val/box_loss", 0.0)),
                "final_val_cls_loss": float(best_row.get("val/cls_loss", 0.0))
            }
            # WandB 요약(summary) 탭에 등록
            for k, v in best_summary.items():
                wandb.run.summary[k] = v
                
        print(f"[완료] {len(df)}개 에폭 지표 및 Summary 등록 완료.")
    else:
        print(f"[경고] results.csv 파일이 존재하지 않습니다: {results_csv_file}")

    # 4. 결과 차트 및 혼동 행렬 이미지 파일 WandB 업로드
    image_files = [
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "BoxPR_curve.png",
        "BoxF1_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "val_batch0_pred.jpg",
        "val_batch1_pred.jpg",
        "val_batch2_pred.jpg"
    ]
    
    images_to_log = {}
    for img_name in image_files:
        img_path = run_dir / img_name
        if img_path.exists():
            clean_key = img_name.replace(".", "_")
            images_to_log[f"charts/{clean_key}"] = wandb.Image(str(img_path), caption=img_name)
            
    if images_to_log:
        wandb.log(images_to_log)
        print(f"[완료] 총 {len(images_to_log)}개의 분석 차트 및 검증 이미지를 WandB에 업로드했습니다.")

    # 5. Run 종료
    run_url = run.get_url()
    run.finish()
    print(f"[성공] '{run_name}' 업로드가 완료되었습니다. 대시보드 URL: {run_url}\n")
    return best_summary

def upload_all_experiments(base_runs_dir="runs/rovoroad-pothole", project_name="rovoroad-pothole"):
    """
    4개 모델(baseline, cbam, bifpn, full)의 학습 로그를 순차 업로드하고,
    종합 비교 테이블(Ablation Summary)을 별도 Run으로 WandB에 생성하는 함수입니다.
    """
    # 1. WandB API 인증
    api_key = get_wandb_api_key("WandB_API_KEY.txt")
    wandb.login(key=api_key)
    print("[성공] WandB API 키 인증이 완료되었습니다.")
    
    # 2. 4가지 실험 디렉터리 정의
    experiments = [
        "yolo11m-baseline",
        "yolo11m-cbam",
        "yolo11m-bifpn",
        "yolo11m-full"
    ]
    
    base_path = Path(base_runs_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"[오류] runs 디렉터리를 찾을 수 없습니다: {base_path.resolve()}")

    comparison_records = []
    
    for exp in experiments:
        exp_dir = base_path / exp
        if exp_dir.exists():
            summary = upload_run_to_wandb(exp_dir, project_name=project_name)
            if summary:
                comparison_records.append({
                    "Model": exp,
                    "Best Epoch": summary.get("best_epoch", 0),
                    "Precision": summary.get("best_precision", 0.0),
                    "Recall": summary.get("best_recall", 0.0),
                    "mAP@50": summary.get("best_mAP50", 0.0),
                    "mAP@50-95": summary.get("best_mAP50-95", 0.0)
                })
        else:
            print(f"[경고] {exp_dir} 경로를 찾을 수 없어 건너뜁니다.")

    # 3. 종합 Ablation Study 비교 테이블 업로드
    if comparison_records:
        print("\n=======================================================")
        print("  [WandB 종합 비교 테이블 생성] ablation-comparison-summary")
        print("=======================================================\n")
        
        summary_run = wandb.init(
            project=project_name,
            name="ablation-comparison-summary",
            reinit=True,
            tags=["summary", "ablation-table"]
        )
        
        # WandB Table 생성
        columns = ["Model", "Best Epoch", "Precision", "Recall", "mAP@50", "mAP@50-95"]
        table_data = [
            [
                r["Model"],
                r["Best Epoch"],
                round(r["Precision"], 4),
                round(r["Recall"], 4),
                round(r["mAP@50"], 4),
                round(r["mAP@50-95"], 4)
            ]
            for r in comparison_records
        ]
        
        wandb_table = wandb.Table(columns=columns, data=table_data)
        wandb.log({"Ablation_Study_Comparison_Table": wandb_table})
        
        summary_url = summary_run.get_url()
        summary_run.finish()
        print(f"[성공] 종합 비교 테이블 생성이 완료되었습니다. 대시보드 URL: {summary_url}")

if __name__ == "__main__":
    print("[시스템] WandB 오프라인 학습 로그 동기화 스크립트를 가동합니다.")
    upload_all_experiments()
