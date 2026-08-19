"""
ROVOROAD Pothole Detection 프로젝트 - Phase 5: ONNX 모델 변환 및 검증 스크립트

[ 스크립트 개요 ]
이 스크립트는 PyTorch로 학습된 커스텀 YOLO11 모델(.pt)을 범용 고속 추론 포맷인
ONNX(Open Neural Network Exchange, OpSet 17) 포맷으로 변환하고,
그래프 최적화(onnxslim/onnxsim) 및 ONNX Runtime(ORT) 기반의 수치 동등성 검증과
추론 속도(Latency, FPS) 벤치마크를 원스톱으로 수행하는 도구입니다.

[ 주요 기능 ]
1. 커스텀 모듈(CBAM Attention, BiFPN Neck) 자동 레지스트리 바인딩 및 ONNX 변환
2. ONNX Simplifier를 통한 불필요한 노드 제거 및 상수 폴딩(Constant Folding)
3. onnx.checker를 통한 ONNX 그래프 무결성 검증
4. PyTorch 모델과 ONNX Runtime 모델 간의 출력 텐서 수치 오차(Max/Mean Abs Diff, Cosine Sim) 정밀 검증
5. PyTorch vs ONNX Runtime 간의 Latency(ms/img) 및 FPS 성능 벤치마크

[ 사용 예시 ]
1. 단일 모델 변환 (검증 및 벤치마크 포함):
   python export.py --weights weights/yolo11m_full_best.pt --verify --benchmark

2. 모든 모델(4종) 일괄 변환:
   python export.py --weights all --verify --benchmark

3. 배치 동적(Dynamic batch) 옵션 적용 변환:
   python export.py --weights weights/yolo11m_baseline_best.pt --dynamic
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import torch

# 1. ROVOROAD 커스텀 모듈(CBAM, BiFPN)을 Ultralytics 프레임워크에 동적 등록
# 이 모듈이 임포트되어야 커스텀 모델 구조를 온전히 해석하고 ONNX 그래프로 추적(trace)할 수 있습니다.
import model.register

from ultralytics import YOLO
import onnx
import onnxruntime as ort


def parse_arguments() -> argparse.Namespace:
    """
    명령줄 인자(CLI Arguments)를 파싱하는 함수입니다.
    초보자도 직관적으로 옵션을 이해할 수 있도록 기본값과 도움말을 상세히 설정합니다.
    """
    parser = argparse.ArgumentParser(
        description="ROVOROAD YOLO11 Custom Model ONNX Export & Benchmark Pipeline"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/yolo11m_full_best.pt",
        help="변환할 PyTorch 가중치(.pt) 파일 경로 또는 'all' (기본값: weights/yolo11m_full_best.pt)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="모델 입력 이미지 해상도 정사각형 크기 (기본값: 640)"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX OpSet 버전 (기본값: 17, 최신 PyTorch 연산자 완벽 지원)"
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="배치 차원을 가변(Dynamic)으로 설정할지 여부 (기본값: False, 고정 shape)"
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        default=True,
        help="ONNX Simplifier(onnxslim/onnxsim)로 그래프를 최적화할지 여부 (기본값: True)"
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="FP16 반정밀도로 변환할지 여부 (기본값: False, FP32 사용)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="ONNX 유효성 검사 및 PyTorch vs ORT 수치 동등성 검증 수행 여부 (기본값: True)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=True,
        help="PyTorch vs ONNX Runtime 추론 속도 벤치마크 수행 여부 (기본값: True)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="벤치마크 수행 디바이스 ('cpu' 또는 '0', 'cuda') (기본값: 'cpu')"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="속도 벤치마크 반복 횟수 (기본값: 100)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="속도 벤치마크 웜업 횟수 (기본값: 10)"
    )
    return parser.parse_args()


def export_torch_to_onnx(
    weights_path: str,
    imgsz: int = 640,
    opset: int = 17,
    dynamic: bool = False,
    simplify: bool = True,
    half: bool = False
) -> str:
    """
    PyTorch 모델(.pt)을 Ultralytics 엔진을 통해 ONNX 포맷으로 변환합니다.

    :param weights_path: 입력 PyTorch 가중치 파일 경로
    :param imgsz: 입력 이미지 해상도
    :param opset: ONNX 연산자 세트 버전
    :param dynamic: 동적 배치 지원 여부
    :param simplify: onnxsim/onnxslim 그래프 최적화 여부
    :param half: FP16 변환 여부
    :return: 생성된 ONNX 파일 경로
    """
    weights_file = Path(weights_path)
    if not weights_file.exists():
        raise FileNotFoundError(f"[오류] 가중치 파일을 찾을 수 없습니다: {weights_path}")

    print("=" * 80)
    print(f"[단계 1] PyTorch -> ONNX 변환 시작: {weights_file.name}")
    print(f" - 입력 가중치: {weights_file.resolve()}")
    print(f" - 입력 해상도: {imgsz}x{imgsz}")
    print(f" - ONNX OpSet: {opset}")
    print(f" - Dynamic Batch: {dynamic}")
    print(f" - Graph Simplify: {simplify}")
    print(f" - Half Precision (FP16): {half}")
    print("=" * 80)

    # YOLO 모델 인스턴스 생성 (커스텀 parse_model이 적용되어 CBAM/BiFPN이 완벽히 바인딩됨)
    model = YOLO(str(weights_file))

    # Ultralytics export 함수 호출
    onnx_file_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        dynamic=dynamic,
        simplify=simplify,
        half=half
    )

    print(f"[성공] ONNX 파일 변환 완료: {onnx_file_path}")
    file_size_mb = Path(onnx_file_path).stat().st_size / (1024 * 1024)
    print(f" - ONNX 파일 크기: {file_size_mb:.2f} MB")
    return str(onnx_file_path)


def verify_onnx_model(onnx_path: str) -> bool:
    """
    생성된 ONNX 파일의 그래프 유효성과 구조 무결성을 검사합니다.

    :param onnx_path: 검증할 ONNX 파일 경로
    :return: 유효성 검사 성공 여부
    """
    print("-" * 80)
    print(f"[단계 2] ONNX 모델 그래프 유효성 검사 (onnx.checker): {Path(onnx_path).name}")
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # 입출력 텐서 정보 추출
        input_info = []
        for inp in onnx_model.graph.input:
            shape = [d.dim_value if d.dim_value > 0 else (d.dim_param or "dynamic") for d in inp.type.tensor_type.shape.dim]
            input_info.append(f"{inp.name}: {shape}")
            
        output_info = []
        for out in onnx_model.graph.output:
            shape = [d.dim_value if d.dim_value > 0 else (d.dim_param or "dynamic") for d in out.type.tensor_type.shape.dim]
            output_info.append(f"{out.name}: {shape}")

        print(" - ONNX 그래프 상태: 정상 (Valid)")
        print(f" - 입력 노드: {', '.join(input_info)}")
        print(f" - 출력 노드: {', '.join(output_info)}")
        print(f" - 프로듀서: {onnx_model.producer_name} (버전: {onnx_model.producer_version})")
        print(f" - IR 버전: {onnx_model.ir_version}")
        return True
    except Exception as e:
        print(f"[오류] ONNX 모델 유효성 검사 실패: {e}")
        return False


def verify_numerical_parity(
    pt_path: str,
    onnx_path: str,
    imgsz: int = 640
) -> Dict[str, Any]:
    """
    PyTorch 모델과 ONNX Runtime 세션에 동일한 가상 텐서를 입력하여
    출력 결과의 수치적 오차(Maximum Absolute Difference, Mean Absolute Error, Cosine Similarity)를
    계산함으로써 변환 과정에서의 수치적 정밀도 손실 여부를 검증합니다.

    :param pt_path: PyTorch 가중치 경로
    :param onnx_path: ONNX 파일 경로
    :param imgsz: 입력 해상도
    :return: 수치 오차 메트릭 딕셔너리
    """
    print("-" * 80)
    print(f"[단계 3] PyTorch vs ONNX Runtime 수치 동등성(Numerical Parity) 검증")

    # 1. 동일한 난수 시드 기반의 더미 입력 텐서 생성 (정규화된 이미지 0.0 ~ 1.0)
    np.random.seed(42)
    torch.manual_seed(42)
    dummy_input_np = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)
    dummy_input_torch = torch.from_numpy(dummy_input_np)

    # 2. PyTorch 모델 순전파 (Evaluation 모드)
    yolo_model = YOLO(pt_path)
    pt_nn_model = yolo_model.model.cpu().eval()

    with torch.no_grad():
        pt_output = pt_nn_model(dummy_input_torch)
        # Ultralytics 출력 형태가 튜플이나 리스트인 경우 첫 번째 텐서 추출
        if isinstance(pt_output, (tuple, list)):
            pt_out_tensor = pt_output[0].cpu().numpy()
        elif hasattr(pt_output, "data"):
            pt_out_tensor = pt_output.data.cpu().numpy()
        else:
            pt_out_tensor = pt_output.cpu().numpy()

    # 3. ONNX Runtime 세션 실행 (CPU)
    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = ort_session.get_inputs()[0].name
    ort_outputs = ort_session.run(None, {input_name: dummy_input_np})
    ort_out_tensor = ort_outputs[0]

    # 4. 수치 오차 계산
    max_abs_diff = float(np.max(np.abs(pt_out_tensor - ort_out_tensor)))
    mean_abs_diff = float(np.mean(np.abs(pt_out_tensor - ort_out_tensor)))

    # 코사인 유사도 계산 (평탄화된 벡터 기준)
    flat_pt = pt_out_tensor.flatten()
    flat_ort = ort_out_tensor.flatten()
    norm_pt = np.linalg.norm(flat_pt)
    norm_ort = np.linalg.norm(flat_ort)
    cosine_sim = float(np.dot(flat_pt, flat_ort) / (norm_pt * norm_ort + 1e-12))

    # 허용 오차 판정 (FP32 기준 통상 1e-3 이하이면 완벽 동등)
    passed = max_abs_diff < 1e-3

    results = {
        "pt_shape": list(pt_out_tensor.shape),
        "ort_shape": list(ort_out_tensor.shape),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "cosine_similarity": cosine_sim,
        "parity_passed": passed
    }

    print(f" - PyTorch 출력 Shape: {pt_out_tensor.shape}")
    print(f" - ONNX Runtime 출력 Shape: {ort_out_tensor.shape}")
    print(f" - 최대 절대 오차 (Max Absolute Diff): {max_abs_diff:.6e}")
    print(f" - 평균 절대 오차 (Mean Absolute Diff): {mean_abs_diff:.6e}")
    print(f" - 코사인 유사도 (Cosine Similarity): {cosine_sim:.8f}")
    if passed:
        print(" - 수치 동등성 판정: [합격] (PyTorch와 ONNX Runtime 출력이 완벽히 일치합니다)")
    else:
        print(" - 수치 동등성 판정: [주의] (오차가 기준치 1e-3을 초과하였습니다)")

    return results


def benchmark_speed(
    pt_path: str,
    onnx_path: str,
    imgsz: int = 640,
    device: str = "cpu",
    runs: int = 100,
    warmup: int = 10
) -> Dict[str, Any]:
    """
    PyTorch 모델과 ONNX Runtime 세션의 순수 추론 지연시간(Latency)과
    초당 프레임 수(FPS)를 측정하여 속도 개선 효과를 비교합니다.

    :param pt_path: PyTorch 가중치 경로
    :param onnx_path: ONNX 파일 경로
    :param imgsz: 입력 이미지 크기
    :param device: 실행 디바이스 ('cpu' 또는 'cuda'/'0')
    :param runs: 반복 횟수
    :param warmup: 워밍업 횟수
    :return: 벤치마크 결과 딕셔너리
    """
    print("-" * 80)
    print(f"[단계 4] PyTorch vs ONNX Runtime 추론 속도(Latency/FPS) 벤치마크 (반복 {runs}회)")

    # 디바이스 설정
    use_cuda = (device != "cpu") and torch.cuda.is_available()
    torch_device = torch.device("cuda:0" if use_cuda else "cpu")

    # 더미 입력 데이터 준비
    dummy_np = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
    dummy_torch = torch.from_numpy(dummy_np).to(torch_device)

    # -------------------------------------------------------------
    # 1. PyTorch 모델 속도 측정
    # -------------------------------------------------------------
    yolo_model = YOLO(pt_path)
    pt_nn = yolo_model.model.to(torch_device).eval()

    # 워밍업 (Warm-up)
    with torch.no_grad():
        for _ in range(warmup):
            _ = pt_nn(dummy_torch)
            if use_cuda:
                torch.cuda.synchronize()

    # 본 측정
    pt_latencies = []
    with torch.no_grad():
        for _ in range(runs):
            t_start = time.perf_counter()
            _ = pt_nn(dummy_torch)
            if use_cuda:
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            pt_latencies.append((t_end - t_start) * 1000.0)  # ms 단위

    pt_mean_latency = float(np.mean(pt_latencies))
    pt_fps = 1000.0 / pt_mean_latency if pt_mean_latency > 0 else 0.0

    # -------------------------------------------------------------
    # 2. ONNX Runtime 세션 속도 측정
    # -------------------------------------------------------------
    providers = ["CPUExecutionProvider"]
    if use_cuda and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    ort_session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = ort_session.get_inputs()[0].name

    # 워밍업 (Warm-up)
    for _ in range(warmup):
        _ = ort_session.run(None, {input_name: dummy_np})

    # 본 측정
    ort_latencies = []
    for _ in range(runs):
        t_start = time.perf_counter()
        _ = ort_session.run(None, {input_name: dummy_np})
        t_end = time.perf_counter()
        ort_latencies.append((t_end - t_start) * 1000.0)  # ms 단위

    ort_mean_latency = float(np.mean(ort_latencies))
    ort_fps = 1000.0 / ort_mean_latency if ort_mean_latency > 0 else 0.0

    # 속도 향상 비율 (Speedup Ratio)
    speedup = pt_mean_latency / ort_mean_latency if ort_mean_latency > 0 else 1.0

    benchmark_data = {
        "device": "CUDA (GPU)" if use_cuda else "CPU",
        "ort_provider": providers[0],
        "pt_latency_ms": pt_mean_latency,
        "pt_fps": pt_fps,
        "ort_latency_ms": ort_mean_latency,
        "ort_fps": ort_fps,
        "speedup_ratio": speedup
    }

    print(f" - 측정 디바이스: {benchmark_data['device']} (Provider: {benchmark_data['ort_provider']})")
    print(f" - PyTorch 평균 Latency: {pt_mean_latency:.2f} ms | FPS: {pt_fps:.1f}")
    print(f" - ONNX Runtime 평균 Latency: {ort_mean_latency:.2f} ms | FPS: {ort_fps:.1f}")
    print(f" - 속도 개선 비율 (Speedup): {speedup:.2f}배")
    print("=" * 80)

    return benchmark_data


def process_single_model(
    weights_path: str,
    args: argparse.Namespace
) -> Dict[str, Any]:
    """
    단일 가중치 파일에 대해 ONNX 변환, 무결성 검증, 수치 동등성 및 벤치마크를 수행합니다.
    """
    model_name = Path(weights_path).stem
    results = {
        "model_name": model_name,
        "weights_path": weights_path,
        "onnx_path": "",
        "file_size_mb": 0.0,
        "valid_graph": False,
        "parity": {},
        "benchmark": {}
    }

    # 1. PyTorch -> ONNX 변환
    onnx_path = export_torch_to_onnx(
        weights_path=weights_path,
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
        half=args.half
    )
    results["onnx_path"] = onnx_path
    results["file_size_mb"] = Path(onnx_path).stat().st_size / (1024 * 1024)

    # 2. ONNX 유효성 검사
    if args.verify:
        results["valid_graph"] = verify_onnx_model(onnx_path)
        results["parity"] = verify_numerical_parity(
            pt_path=weights_path,
            onnx_path=onnx_path,
            imgsz=args.imgsz
        )

    # 3. 속도 벤치마크
    if args.benchmark:
        results["benchmark"] = benchmark_speed(
            pt_path=weights_path,
            onnx_path=onnx_path,
            imgsz=args.imgsz,
            device=args.device,
            runs=args.runs,
            warmup=args.warmup
        )

    return results


def main():
    """
    메인 실행 함수입니다.
    """
    args = parse_arguments()

    # 'all' 옵션 지정 시 4개 모델 순차 처리
    if args.weights.lower() == "all":
        all_models = [
            "weights/yolo11m_baseline_best.pt",
            "weights/yolo11m_cbam_best.pt",
            "weights/yolo11m_bifpn_best.pt",
            "weights/yolo11m_full_best.pt"
        ]
        print(f"[ROVOROAD] 4개 모델 일괄 ONNX 변환 및 벤치마크 모드 시작: 총 {len(all_models)}개 모델")
        
        all_results = []
        for w_path in all_models:
            if os.path.exists(w_path):
                res = process_single_model(w_path, args)
                all_results.append(res)
            else:
                print(f"[경고] 파일이 존재하지 않아 건너뜁니다: {w_path}")

        print("\n" + "=" * 90)
        print("4개 모델 ONNX 변환 및 벤치마크 종합 결과")
        print("=" * 90)
        header = f"{'모델명':<25} | {'ONNX크기(MB)':<12} | {'Max Abs Diff':<14} | {'PyTorch FPS':<12} | {'ORT FPS':<10} | {'가속비율':<8}"
        print(header)
        print("-" * 90)
        for r in all_results:
            m_name = r["model_name"]
            size_mb = f"{r['file_size_mb']:.1f} MB"
            diff = f"{r['parity'].get('max_abs_diff', 0.0):.2e}" if r.get("parity") else "N/A"
            pt_fps = f"{r['benchmark'].get('pt_fps', 0.0):.1f}" if r.get("benchmark") else "N/A"
            ort_fps = f"{r['benchmark'].get('ort_fps', 0.0):.1f}" if r.get("benchmark") else "N/A"
            speedup = f"{r['benchmark'].get('speedup_ratio', 1.0):.2f}x" if r.get("benchmark") else "N/A"
            print(f"{m_name:<25} | {size_mb:<12} | {diff:<14} | {pt_fps:<12} | {ort_fps:<10} | {speedup:<8}")
        print("=" * 90)

    else:
        process_single_model(args.weights, args)


if __name__ == "__main__":
    main()
