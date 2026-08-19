"""
ROVOROAD Pothole Detection 프로젝트 - Phase 6: 고화질 비교 데모 영상 제작 스크립트 (make_demo_video.py)

[ 스크립트 설명 ]
이 스크립트는 Phase 5에서 변환된 ONNX 모델들을 사용하여 Test 셋 이미지들을 순회하며,
1) [Original 도로 이미지 vs 모델 탐지 결과] 1:1 Side-by-Side 비교 영상,
2) [Baseline / +CBAM / +BiFPN / Full] 4종 모델 2x2 Grid 동시 비교 영상
을 고화질 MP4 비디오(1920x1080 Full HD, 30 FPS)로 자동 렌더링하여 저장합니다.

[ 초보자를 위한 핵심 설계 안내 ]
1. 3초 고정 프레임 제어 (Duration Control):
   - 각 테스트 이미지 장면마다 정확히 3.0초(30 FPS 기준 90프레임) 동안 화면을 유지하도록 프레임을 복제하여 기록합니다.
2. 실시간 ONNX 추론 및 HUD 오버레이:
   - 각 모델이 실제로 이미지를 추론하는 데 걸린 지연시간(Latency, ms)과 FPS를 측정하여 상단 반투명 패널에 실시간 표시합니다.
3. 3가지 유연한 렌더링 모드:
   - mode 'side-by-side': 원본 이미지와 특정 모델의 1:1 비교 영상 생성
   - mode '4way': 4개 모델 전체의 2x2 격자 비교 영상 생성
   - mode 'all': 4개 모델 각각의 1:1 영상 4종 + 4way 영상 1종 일괄 생성
4. 포트홀 라벨 포함 이미지 스마트 필터링:
   - Test 셋 중 포트홀이 실제로 존재하는 이미지들을 우선 선별하여 데모 영상의 시각적 비교 효과를 극대화합니다.

[ 사용 예시 ]
1. Original vs Full 모델 1:1 비교 영상 생성 (기본값):
   python demo/make_demo_video.py --mode side-by-side --weights weights/yolo11m_full_best.onnx --output runs/demo_original_vs_full.mp4

2. 4종 모델 2x2 종합 비교 영상 생성:
   python demo/make_demo_video.py --mode 4way --output runs/demo_4way_comparison.mp4

3. 모든 조합 일괄 생성:
   python demo/make_demo_video.py --mode all
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np

# 프로젝트 루트 디렉토리를 파이썬 경로에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from inference import YOLOv11ONNXDetector
from utils.visualize import (
    draw_detections,
    draw_hud_panel,
    create_side_by_side_frame,
    create_2x2_grid_frame
)


def parse_arguments() -> argparse.Namespace:
    """
    데모 영상 제작을 위한 명령줄 인자를 파싱합니다.
    """
    parser = argparse.ArgumentParser(
        description="ROVOROAD Phase 6 - ONNX High-Quality Demo Video Generator"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="side-by-side",
        choices=["side-by-side", "4way", "all"],
        help="영상 제작 모드 ('side-by-side': 원본 vs 모델 1:1, '4way': 4개 모델 2x2 비교, 'all': 전체 일괄 생성)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/yolo11m_full_best.onnx",
        help="side-by-side 모드에서 사용할 ONNX 가중치 파일 경로"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="datasets/HRP4K/test/images",
        help="테스트 이미지들이 위치한 폴더 경로"
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default="datasets/HRP4K/test/labels",
        help="테스트 라벨들이 위치한 폴더 경로 (라벨 있는 이미지 선별용)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="저장할 MP4 파일 경로 (미지정 시 모드에 따라 자동 지정)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="테스트 이미지 한 장면당 유지할 시간 (초 단위, 기본값: 3.0초)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="생성할 영상의 초당 프레임 수 (FPS, 기본값: 30)"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=15,
        help="영상에 포함할 최대 테스트 이미지 수 (기본값: 15장, 0=전체)"
    )
    parser.add_argument(
        "--only-labeled",
        action="store_true",
        default=True,
        help="포트홀 라벨이 존재하는 이미지만 선별할지 여부 (기본값: True)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="탐지 신뢰도(Confidence) 임계값 (기본값: 0.25)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU 임계값 (기본값: 0.45)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="추론 디바이스 ('cpu' 또는 'cuda' / '0')"
    )
    return parser.parse_args()


def get_test_images(
    source_dir: str,
    labels_dir: str = "",
    max_images: int = 15,
    only_labeled: bool = True
) -> List[Path]:
    """
    지정된 디렉토리에서 테스트 이미지 파일 목록을 가져옵니다.
    only_labeled가 True이면 포트홀이 라벨링된 이미지를 우선적으로 필터링합니다.
    """
    src_path = Path(source_dir)
    if not src_path.exists():
        raise FileNotFoundError(f"테스트 이미지 폴더를 찾을 수 없습니다: {src_path.resolve()}")
        
    all_images = sorted(list(src_path.glob("*.jpg")) + list(src_path.glob("*.png")))
    if not all_images:
        raise FileNotFoundError(f"폴더 내에 이미지 파일(*.jpg, *.png)이 없습니다: {src_path.resolve()}")
        
    lbl_path = Path(labels_dir)
    if only_labeled and lbl_path.exists():
        labeled_images = []
        for img_p in all_images:
            txt_p = lbl_path / f"{img_p.stem}.txt"
            # 라벨 파일이 존재하고 크기가 0보다 큰 경우 (포트홀이 1개 이상 라벨링된 경우)
            if txt_p.exists() and txt_p.stat().st_size > 0:
                labeled_images.append(img_p)
                
        if labeled_images:
            all_images = labeled_images
            
    if max_images > 0:
        all_images = all_images[:max_images]
        
    return all_images


def generate_side_by_side_video(
    weights_path: str,
    image_files: List[Path],
    output_path: str,
    duration: float = 3.0,
    fps: int = 30,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "cpu",
    target_width: int = 1920,
    target_height: int = 1080
) -> str:
    """
    '원본 도로 이미지 vs 선택 모델 탐지 결과' 1:1 비교 비디오를 생성합니다.
    """
    weights_file = Path(weights_path)
    if not weights_file.exists():
        raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {weights_file.resolve()}")
        
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 75)
    print(f"  [1:1 Side-by-Side 데모 영상 제작]")
    print(f"  - 모델: {weights_file.name}")
    print(f"  - 이미지 수: {len(image_files)}장 (장면당 {duration}초 = {int(duration * fps)}프레임)")
    print(f"  - 출력 해상도: {target_width}x{target_height} @ {fps} FPS")
    print(f"  - 저장 경로: {out_file.resolve()}")
    print("=" * 75)
    
    # 1. ONNX 추론기 로드
    detector = YOLOv11ONNXDetector(
        onnx_path=str(weights_file),
        imgsz=640,
        conf_thresh=conf,
        iou_thresh=iou,
        device=device
    )
    
    # 모델명 포맷팅 (예: "YOLO11m-Full (ONNX)")
    stem = weights_file.stem.replace("_best", "").replace("yolo11m_", "").upper()
    model_title = f"YOLO11m-{stem} (ONNX)"
    
    # 2. OpenCV VideoWriter 초기화
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        str(out_file),
        fourcc,
        float(fps),
        (target_width, target_height)
    )
    
    frames_per_scene = int(duration * fps)
    
    # 3. 이미지 순회 및 렌더링
    for idx, img_p in enumerate(image_files, 1):
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            print(f"  [{idx}/{len(image_files)}] 이미지 로드 실패: {img_p.name}")
            continue
            
        # ONNX 추론 수행
        res = detector.predict(img_bgr)
        boxes = res["boxes"]
        scores = res["scores"]
        latency_ms = res["latency_ms"]
        model_fps = res["fps"]
        
        # 탐지 결과 이미지 생성
        det_img = draw_detections(img_bgr, boxes, scores, class_names=["pothole"])
        
        # 1:1 Side-by-Side 프레임 합성
        combined_frame = create_side_by_side_frame(
            original_img=img_bgr,
            detected_img=det_img,
            orig_title=f"ORIGINAL IMAGE ({img_p.name})",
            det_title=model_title,
            latency_ms=latency_ms,
            fps=model_fps,
            det_count=len(boxes),
            target_width=target_width,
            target_height=target_height
        )
        
        # 지정된 시간(3초 = 90프레임) 동안 동일 프레임 기록
        for _ in range(frames_per_scene):
            video_writer.write(combined_frame)
            
        print(f"  [{idx:02d}/{len(image_files):02d}] {img_p.name} -> 포트홀 {len(boxes)}개 탐지 | Latency: {latency_ms:.1f}ms ({model_fps:.1f} FPS) -> 렌더링 완료")
        
    video_writer.release()
    total_sec = (len(image_files) * frames_per_scene) / fps
    print(f"[성공] 1:1 Side-by-Side 데모 영상 생성 완료! (총 재생 시간: {total_sec:.1f}초)")
    return str(out_file)


def generate_4way_grid_video(
    image_files: List[Path],
    output_path: str,
    duration: float = 3.0,
    fps: int = 30,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "cpu",
    target_width: int = 1920,
    target_height: int = 1080
) -> str:
    """
    'Baseline / +CBAM / +BiFPN / Full' 4종 모델의 2x2 격자 동시 비교 영상을 생성합니다.
    """
    models_cfg = [
        {"name": "YOLO11m-Baseline", "weights": "weights/yolo11m_baseline_best.onnx"},
        {"name": "YOLO11m + CBAM",    "weights": "weights/yolo11m_cbam_best.onnx"},
        {"name": "YOLO11m + BiFPN",   "weights": "weights/yolo11m_bifpn_best.onnx"},
        {"name": "YOLO11m + Full",    "weights": "weights/yolo11m_full_best.onnx"}
    ]
    
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 75)
    print("  [4종 모델 2x2 Grid 종합 비교 데모 영상 제작]")
    print(f"  - 비교 모델: {', '.join([m['name'] for m in models_cfg])}")
    print(f"  - 이미지 수: {len(image_files)}장 (장면당 {duration}초 = {int(duration * fps)}프레임)")
    print(f"  - 출력 해상도: {target_width}x{target_height} @ {fps} FPS")
    print(f"  - 저장 경로: {out_file.resolve()}")
    print("=" * 75)
    
    # 1. 4개 모델 추론기 모두 초기화
    detectors = []
    for m in models_cfg:
        w_path = Path(m["weights"])
        if not w_path.exists():
            raise FileNotFoundError(f"필요한 가중치 파일을 찾을 수 없습니다: {w_path.resolve()}")
        det = YOLOv11ONNXDetector(
            onnx_path=str(w_path),
            imgsz=640,
            conf_thresh=conf,
            iou_thresh=iou,
            device=device
        )
        detectors.append({"name": m["name"], "detector": det})
        
    # 2. OpenCV VideoWriter 초기화
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        str(out_file),
        fourcc,
        float(fps),
        (target_width, target_height)
    )
    
    frames_per_scene = int(duration * fps)
    
    # 3. 이미지 순회 및 4개 모델 동시 추론 및 합성
    for idx, img_p in enumerate(image_files, 1):
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            print(f"  [{idx}/{len(image_files)}] 이미지 로드 실패: {img_p.name}")
            continue
            
        cell_infos = []
        log_items = []
        
        for det_dict in detectors:
            name = det_dict["name"]
            det = det_dict["detector"]
            
            # 각 모델별 ONNX 추론 실행
            res = det.predict(img_bgr)
            boxes = res["boxes"]
            scores = res["scores"]
            latency_ms = res["latency_ms"]
            model_fps = res["fps"]
            
            det_img = draw_detections(img_bgr, boxes, scores, class_names=["pothole"])
            
            cell_infos.append({
                "image": det_img,
                "title": f"{name} (ONNX)",
                "latency_ms": latency_ms,
                "fps": model_fps,
                "det_count": len(boxes)
            })
            log_items.append(f"{name}: {len(boxes)}개({latency_ms:.1f}ms)")
            
        # 2x2 Grid 프레임 합성
        grid_frame = create_2x2_grid_frame(
            frames_info=cell_infos,
            target_width=target_width,
            target_height=target_height
        )
        
        # 3초 동안 기록
        for _ in range(frames_per_scene):
            video_writer.write(grid_frame)
            
        print(f"  [{idx:02d}/{len(image_files):02d}] {img_p.name} -> {' | '.join(log_items)} -> 렌더링 완료")
        
    video_writer.release()
    total_sec = (len(image_files) * frames_per_scene) / fps
    print(f"[성공] 4종 모델 2x2 Grid 데모 영상 생성 완료! (총 재생 시간: {total_sec:.1f}초)")
    return str(out_file)


def main():
    """
    메인 실행 함수입니다.
    """
    args = parse_arguments()
    
    # 1. 테스트 이미지 목록 가져오기
    try:
        image_files = get_test_images(
            source_dir=args.source_dir,
            labels_dir=args.labels_dir,
            max_images=args.max_images,
            only_labeled=args.only_labeled
        )
    except Exception as e:
        print(f"[오류] {e}")
        sys.exit(1)
        
    print(f"[안내] Test 셋 중 포트홀이 존재하는 이미지 {len(image_files)}장을 선별하여 데모 영상을 제작합니다.")
    
    # 2. 모드별 실행
    if args.mode == "side-by-side":
        out_path = args.output if args.output else "runs/demo_original_vs_full.mp4"
        generate_side_by_side_video(
            weights_path=args.weights,
            image_files=image_files,
            output_path=out_path,
            duration=args.duration,
            fps=args.fps,
            conf=args.conf,
            iou=args.iou,
            device=args.device
        )
    elif args.mode == "4way":
        out_path = args.output if args.output else "runs/demo_4way_comparison.mp4"
        generate_4way_grid_video(
            image_files=image_files,
            output_path=out_path,
            duration=args.duration,
            fps=args.fps,
            conf=args.conf,
            iou=args.iou,
            device=args.device
        )
    elif args.mode == "all":
        print("\n[모드: ALL] 모든 모델별 1:1 비교 영상 4종 및 4way 종합 영상을 일괄 제작합니다.")
        
        # 1. 4개 모델 1:1 Side-by-Side 영상 제작
        all_weights = [
            ("Baseline", "weights/yolo11m_baseline_best.onnx", "runs/demo_original_vs_baseline.mp4"),
            ("CBAM",     "weights/yolo11m_cbam_best.onnx",     "runs/demo_original_vs_cbam.mp4"),
            ("BiFPN",    "weights/yolo11m_bifpn_best.onnx",    "runs/demo_original_vs_bifpn.mp4"),
            ("Full",     "weights/yolo11m_full_best.onnx",     "runs/demo_original_vs_full.mp4")
        ]
        
        for name, w_path, o_path in all_weights:
            if Path(w_path).exists():
                generate_side_by_side_video(
                    weights_path=w_path,
                    image_files=image_files,
                    output_path=o_path,
                    duration=args.duration,
                    fps=args.fps,
                    conf=args.conf,
                    iou=args.iou,
                    device=args.device
                )
            else:
                print(f"[경고] 가중치가 없어 건너뜁니다: {w_path}")
                
        # 2. 4way 종합 그리드 영상 제작
        generate_4way_grid_video(
            image_files=image_files,
            output_path="runs/demo_4way_comparison.mp4",
            duration=args.duration,
            fps=args.fps,
            conf=args.conf,
            iou=args.iou,
            device=args.device
        )
        
    print("\n" + "=" * 75)
    print("  [Phase 6 데모 영상 제작 파이프라인 전체 완료]")
    print("=" * 75)


if __name__ == "__main__":
    main()
