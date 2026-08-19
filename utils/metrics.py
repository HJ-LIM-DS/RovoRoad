import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def measure_model_fps_and_latency(model, sample_tensor, device="cuda", num_warmup=15, num_repeats=50):
    """
    주어진 PyTorch/Ultralytics 모델의 추론 지연시간(Latency, ms) 및 초당 프레임 수(FPS)를
    GPU 동기화를 적용하여 정밀하게 측정하는 함수입니다.
    
    매개변수:
        model: 평가할 모델 인스턴스 (Ultralytics YOLO 또는 PyTorch nn.Module)
        sample_tensor (torch.Tensor): 입력 더미 또는 실제 이미지 텐서 (예: 1, 3, 640, 640)
        device (str): 실행 디바이스 ('cuda' 또는 'cpu')
        num_warmup (int): GPU 웜업 반복 횟수 (캐시 및 클럭 안정화)
        num_repeats (int): 실제 시간 측정 반복 횟수
        
    반환값:
        dict: latency_ms(평균 지연시간), fps(초당 프레임 수), std_ms(표준편차)
    """
    is_cuda = (device == "cuda" or (isinstance(device, str) and device.isdigit())) and torch.cuda.is_available()
    
    # 텐서를 해당 디바이스로 이동
    dev = torch.device(f"cuda:{device}" if (isinstance(device, str) and device.isdigit()) else device)
    
    # Ultralytics YOLO 모델의 내부 PyTorch 모델 추출
    if hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
        net = model.model.to(dev)
    elif isinstance(model, torch.nn.Module):
        net = model.to(dev)
    else:
        net = model
        
    net.eval()
    
    if not isinstance(sample_tensor, torch.Tensor):
        dummy_input = torch.randn(1, 3, 640, 640, device=dev)
    else:
        dummy_input = sample_tensor.to(dev)
        
    # 1. GPU 웜업 (Warmup)
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = net(dummy_input)
            
    if is_cuda:
        torch.cuda.synchronize()
        
    # 2. 정밀 추론 시간 측정
    latencies = []
    with torch.no_grad():
        for _ in range(num_repeats):
            if is_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                _ = net(dummy_input)
                end_event.record()
                
                torch.cuda.synchronize()
                latencies.append(start_event.elapsed_time(end_event))  # ms 단위
            else:
                t0 = time.perf_counter()
                _ = net(dummy_input)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)  # ms 단위
                
    avg_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    
    return {
        "latency_ms": round(avg_latency, 2),
        "fps": round(fps, 1),
        "std_ms": round(std_latency, 2)
    }

def generate_comparison_chart(results_list, save_path="runs/evaluation_comparison.png"):
    """
    Ablation Study 모델들의 mAP@50, mAP@50-95, 정밀도, 재현율, FPS를 비교하는
    다중 서브플롯 막대 차트를 생성하고 이미지 파일로 저장하는 함수입니다.
    
    매개변수:
        results_list (list of dict): 각 모델의 평가 메트릭 딕셔너리 리스트
        save_path (str): 저장할 이미지 파일 경로
    """
    if not results_list:
        print("[경고] 비교 차트를 생성할 데이터가 없습니다.")
        return
        
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 데이터 추출
    models = [r.get("name", "Unknown") for r in results_list]
    map50 = [r.get("mAP50", 0.0) * 100 for r in results_list]
    map50_95 = [r.get("mAP50-95", 0.0) * 100 for r in results_list]
    precision = [r.get("precision", 0.0) * 100 for r in results_list]
    recall = [r.get("recall", 0.0) * 100 for r in results_list]
    fps_list = [r.get("fps", 0.0) for r in results_list]
    params_list = [r.get("params_m", 0.0) for r in results_list]
    
    # 차트 그리기 (2행 2열 구성)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ROVOROAD Pothole Detection - Ablation Study Benchmark", fontsize=16, fontweight="bold")
    
    x = np.arange(len(models))
    bar_width = 0.35
    colors = ["#4A90E2", "#50E3C2", "#F5A623", "#E94E77"]
    
    # 1. mAP@50 & mAP@50-95 비교 (상단 좌측)
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x - bar_width/2, map50, bar_width, label="mAP@50 (%)", color="#3498db")
    bars2 = ax1.bar(x + bar_width/2, map50_95, bar_width, label="mAP@50-95 (%)", color="#2ecc71")
    ax1.set_title("Detection Accuracy (mAP@50 vs mAP@50-95)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15)
    ax1.set_ylabel("Percentage (%)")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    ax1.legend()
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8)

    # 2. Precision & Recall 비교 (상단 우측)
    ax2 = axes[0, 1]
    bars3 = ax2.bar(x - bar_width/2, precision, bar_width, label="Precision (%)", color="#9b59b6")
    bars4 = ax2.bar(x + bar_width/2, recall, bar_width, label="Recall (%)", color="#e67e22")
    ax2.set_title("Precision vs Recall", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=15)
    ax2.set_ylabel("Percentage (%)")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)
    ax2.legend()
    for bar in bars3:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8)
    for bar in bars4:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=8)

    # 3. 추론 속도 FPS 비교 (하단 좌측)
    ax3 = axes[1, 0]
    bars5 = ax3.bar(models, fps_list, color=colors, width=0.5)
    ax3.set_title("Inference Speed (FPS on RTX 3060)", fontweight="bold")
    ax3.set_ylabel("Frames Per Second (FPS)")
    ax3.grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars5:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 4. 모델 파라미터 수 (하단 우측)
    ax4 = axes[1, 1]
    bars6 = ax4.bar(models, params_list, color="#34495e", width=0.5)
    ax4.set_title("Model Parameters (Million)", fontweight="bold")
    ax4.set_ylabel("Params (M)")
    ax4.grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars6:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f"{yval:.2f}M", ha="center", va="bottom", fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_file, dpi=300)
    plt.close()
    print(f"[완료] 종합 벤치마크 비교 차트가 생성되었습니다: {save_file.resolve()}")

def format_summary_markdown(results_list):
    """
    평가 결과를 마크다운 형식의 깔끔한 비교 표 문자열로 포맷팅하는 함수입니다.
    """
    md_lines = [
        "# ROVOROAD Pothole Detection - Test Set Benchmark Summary\n",
        "| 모델명 | 파라미터(M) | 연산량(GFLOPs) | Precision | Recall | mAP@50 | mAP@50-95 | 지연시간(ms) | FPS |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    ]
    
    for r in results_list:
        md_lines.append(
            f"| **{r.get('name')}** | "
            f"{r.get('params_m', 0.0):.2f}M | "
            f"{r.get('gflops', 0.0):.1f} | "
            f"{r.get('precision', 0.0):.4f} | "
            f"{r.get('recall', 0.0):.4f} | "
            f"**{r.get('mAP50', 0.0):.4f}** | "
            f"**{r.get('mAP50-95', 0.0):.4f}** | "
            f"{r.get('latency_ms', 0.0):.2f} ms | "
            f"**{r.get('fps', 0.0):.1f}** |"
        )
        
    return "\n".join(md_lines)
