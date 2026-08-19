"""
ROVOROAD Pothole Detection 프로젝트 - Phase 6: 고속 추론 CLI 스크립트 (inference.py)

[ 스크립트 설명 ]
이 스크립트는 Phase 5에서 변환된 ONNX 모델(.onnx) 또는 학습된 PyTorch 가중치(.pt)를 로드하여
단일 이미지 또는 폴더 내의 테스트 이미지들을 대상으로 객체 탐지(포트홀 검출)를 수행합니다.

[ 초보자를 위한 핵심 파이프라인 안내 ]
1. 전처리 (Preprocessing):
   - 이미지 크기를 모델 입력 규격(640x640)에 맞추되, 원본 왜곡을 방지하기 위해 Letterbox(여백 패딩)를 적용합니다.
   - BGR 색상을 RGB로 변환하고 0~255 픽셀값을 0.0~1.0 실수 범위로 정규화합니다.
2. 추론 (Inference):
   - ONNX Runtime(ORT) 고속 엔진을 통해 신경망 연산을 수행하고 순수 추론 시간(ms)을 정밀 측정합니다.
3. 후처리 (Postprocessing):
   - 출력된 8400개의 앵커 박스 중 신뢰도(Confidence) 기준치 이상의 박스만 추출합니다.
   - NMS(Non-Maximum Suppression, 비최대 억제)를 적용하여 중복된 박스를 하나로 통합합니다.
   - Letterbox 좌표를 원본 이미지 크기 좌표계로 역변환합니다.
4. 시각화 (Visualization):
   - 원본 이미지 위에 바운딩 박스와 신뢰도, 그리고 상단 정보창(HUD)을 오버레이하여 저장합니다.

[ 사용 예시 ]
1. 단일 이미지 ONNX 추론:
   python inference.py --weights weights/yolo11m_full_best.onnx --source datasets/HRP4K/test/images/30000.jpg

2. 폴더 내 테스트 이미지 일괄 추론:
   python inference.py --weights weights/yolo11m_full_best.onnx --source datasets/HRP4K/test/images --max-images 20
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np

# 커스텀 모듈 레지스트리 임포트 (PyTorch .pt 로드 시 필요)
try:
    import model.register
    from ultralytics import YOLO
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ONNX Runtime 임포트
import onnxruntime as ort

# 시각화 유틸리티 임포트
from utils.visualize import draw_detections, draw_hud_panel


class YOLOv11ONNXDetector:
    """
    ONNX Runtime 기반의 YOLOv11 포트홀 탐지 고속 추론기 클래스입니다.
    """
    def __init__(
        self,
        onnx_path: str,
        imgsz: int = 640,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        device: str = "cpu"
    ):
        """
        추론기 초기화 함수입니다.
        
        매개변수:
            onnx_path (str): ONNX 가중치 파일 경로
            imgsz (int): 입력 이미지 해상도 (기본값: 640)
            conf_thresh (float): 신뢰도 임계값
            iou_thresh (float): NMS IoU 임계값
            device (str): 실행 디바이스 ('cpu' 또는 'cuda' / '0')
        """
        self.onnx_path = onnx_path
        self.imgsz = imgsz
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        
        # 1. 실행 프로바이더(Execution Provider) 설정
        providers = ["CPUExecutionProvider"]
        if device.lower() in ["cuda", "gpu", "0"] and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            
        # 2. ONNX Runtime 세션 생성
        self.session = ort.InferenceSession(self.onnx_path, providers=providers)
        
        # 입출력 메타데이터 파싱
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        
        # 모델 클래스 이름 (HRP4K 데이터셋은 pothole 단일 클래스)
        self.class_names = ["pothole"]
        
        # 3. 워밍업 (Warm-up): 첫 추론 시 초기화 지연 방지
        dummy_input = np.zeros((1, 3, self.imgsz, self.imgsz), dtype=np.float32)
        for _ in range(3):
            self.session.run([self.output_name], {self.input_name: dummy_input})

    def letterbox(
        self,
        img: np.ndarray,
        new_shape: Tuple[int, int] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114)
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        이미지의 종횡비를 유지하면서 지정된 크기(new_shape)로 리사이즈하고 회색 여백(Padding)을 채웁니다.
        
        반환값:
            padded_img (np.ndarray): 여백이 추가된 리사이즈 이미지
            ratio (float): 원본 대비 축소/확대 비율
            (dw, dh): 좌우, 상하에 추가된 패딩 크기
        """
        shape = img.shape[:2]  # [height, width]
        
        # 스케일 비율 계산 (너비 비율, 높이 비율 중 작은 값 선택)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # 새로운 크기 계산
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = (new_shape[1] - new_unpad[0]) / 2  # 가로 패딩 분할
        dh = (new_shape[0] - new_unpad[1]) / 2  # 세로 패딩 분할
        
        # 리사이즈
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        # 패딩 추가
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        return padded_img, r, (dw, dh)

    def preprocess(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        입력 BGR 이미지를 ONNX Runtime 추론용 텐서 포맷으로 전처리합니다.
        
        반환값:
            tensor (np.ndarray): (1, 3, imgsz, imgsz) 형태의 float32 배열
            ratio (float): 리사이즈 비율
            pad (tuple): (dw, dh) 패딩 크기
        """
        # 1. Letterbox 패딩 적용
        padded_img, ratio, pad = self.letterbox(img_bgr, (self.imgsz, self.imgsz))
        
        # 2. BGR -> RGB 변환
        img_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        
        # 3. 0~255 정수 -> 0.0~1.0 실수 정규화
        img_norm = img_rgb.astype(np.float32) / 255.0
        
        # 4. HWC -> CHW 축 변환 (Height, Width, Channel -> Channel, Height, Width)
        img_transposed = np.transpose(img_norm, (2, 0, 1))
        
        # 5. 배치 차원 추가: (3, H, W) -> (1, 3, H, W)
        tensor = np.expand_dims(img_transposed, axis=0)
        
        return tensor, ratio, pad

    def postprocess(
        self,
        raw_output: np.ndarray,
        orig_shape: Tuple[int, int],
        ratio: float,
        pad: Tuple[float, float]
    ) -> Tuple[List[List[float]], List[float], List[int]]:
        """
        ONNX 모델의 출력 텐서에서 바운딩 박스를 추출하고 NMS를 적용하여 최종 탐지 결과를 도출합니다.
        
        매개변수:
            raw_output (np.ndarray): 모델 출력 텐서 (shape: [1, 5, 8400] 등)
            orig_shape (tuple): 원본 이미지의 [height, width]
            ratio (float): 전처리 시 적용된 스케일 비율
            pad (tuple): (dw, dh) 패딩 크기
            
        반환값:
            final_boxes (list): [[x1, y1, x2, y2], ...]
            final_scores (list): [score, ...]
            final_cls_ids (list): [class_id, ...]
        """
        # 출력 텐서 차원 정돈 (1, 4+num_classes, num_boxes) -> (num_boxes, 4+num_classes)
        predictions = raw_output[0]
        if predictions.shape[0] < predictions.shape[1]:
            predictions = np.transpose(predictions, (1, 0))  # (8400, 5) 형태로 전치
            
        boxes_list = []
        confidences = []
        class_ids = []
        
        dw, dh = pad
        orig_h, orig_w = orig_shape
        
        # 1. 앵커 박스 순회 및 신뢰도 임계값 필터링
        for row in predictions:
            cx, cy, w, h = row[0:4]
            # 클래스 점수 추출 (단일 클래스인 경우 row[4]가 pothole 신뢰도)
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])
            
            if score >= self.conf_thresh:
                # [cx, cy, w, h] -> [x1, y1, x2, y2] 변환 (Letterbox 좌표계)
                x1 = cx - w / 2.0
                y1 = cy - h / 2.0
                x2 = cx + w / 2.0
                y2 = cy + h / 2.0
                
                # Letterbox 패딩 제거 및 원본 좌표계로 스케일 복원
                x1 = (x1 - dw) / ratio
                y1 = (y1 - dh) / ratio
                x2 = (x2 - dw) / ratio
                y2 = (y2 - dh) / ratio
                
                # 원본 이미지 경계선 클리핑 (좌표 이탈 방지)
                x1 = max(0.0, min(float(orig_w), x1))
                y1 = max(0.0, min(float(orig_h), y1))
                x2 = max(0.0, min(float(orig_w), x2))
                y2 = max(0.0, min(float(orig_h), y2))
                
                boxes_list.append([x1, y1, x2 - x1, y2 - y1])  # OpenCV NMS는 [x, y, w, h] 형식 사용
                confidences.append(score)
                class_ids.append(class_id)
                
        # 2. OpenCV NMS(Non-Maximum Suppression) 적용
        final_boxes = []
        final_scores = []
        final_cls_ids = []
        
        if len(boxes_list) > 0:
            indices = cv2.dnn.NMSBoxes(
                bboxes=boxes_list,
                scores=confidences,
                score_threshold=self.conf_thresh,
                nms_threshold=self.iou_thresh
            )
            
            if len(indices) > 0:
                for idx in indices.flatten():
                    bx, by, bw, bh = boxes_list[idx]
                    # [x, y, w, h] -> [x1, y1, x2, y2]
                    final_boxes.append([bx, by, bx + bw, by + bh])
                    final_scores.append(confidences[idx])
                    final_cls_ids.append(class_ids[idx])
                    
        return final_boxes, final_scores, final_cls_ids

    def predict(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """
        단일 BGR 이미지에 대해 전체 추론 파이프라인을 실행합니다.
        
        반환값:
            dict: {
                'boxes': list,
                'scores': list,
                'class_ids': list,
                'class_names': list,
                'latency_ms': float,
                'fps': float,
                'orig_shape': tuple
            }
        """
        orig_shape = img_bgr.shape[:2]  # [H, W]
        
        # 1. 전처리
        tensor, ratio, pad = self.preprocess(img_bgr)
        
        # 2. 순수 ONNX 추론 및 시간 측정
        t_start = time.perf_counter()
        raw_outputs = self.session.run([self.output_name], {self.input_name: tensor})
        t_end = time.perf_counter()
        
        latency_ms = (t_end - t_start) * 1000.0
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
        
        # 3. 후처리 및 NMS
        boxes, scores, cls_ids = self.postprocess(raw_outputs[0], orig_shape, ratio, pad)
        
        return {
            "boxes": boxes,
            "scores": scores,
            "class_ids": cls_ids,
            "class_names": self.class_names,
            "latency_ms": latency_ms,
            "fps": fps,
            "orig_shape": orig_shape
        }


def parse_arguments() -> argparse.Namespace:
    """
    명령줄 인자를 파싱합니다.
    """
    parser = argparse.ArgumentParser(description="ROVOROAD Pothole Detection - Phase 6 High-speed Inference")
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/yolo11m_full_best.onnx",
        help="추론에 사용할 모델 가중치 (.onnx 또는 .pt)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="datasets/HRP4K/test/images",
        help="추론 대상 이미지 파일 또는 폴더 경로"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="입력 이미지 해상도 (기본값: 640)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="신뢰도(Confidence) 임계값 (기본값: 0.25)"
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
    parser.add_argument(
        "--save-dir",
        type=str,
        default="runs/inference",
        help="추론 결과 저장 디렉토리 (기본값: runs/inference)"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="폴더 추론 시 최대 처리 이미지 수 (기본값: 20, 0=전체)"
    )
    return parser.parse_args()


def main():
    """
    메인 실행 함수입니다.
    """
    args = parse_arguments()
    
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[오류] 가중치 파일을 찾을 수 없습니다: {weights_path.resolve()}")
        sys.exit(1)
        
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("  ROVOROAD Pothole Detection - Phase 6 Inference Pipeline")
    print(f"  - 가중치: {weights_path.name}")
    print(f"  - 소스: {args.source}")
    print(f"  - 임계값: Conf={args.conf}, IoU={args.iou}")
    print(f"  - 디바이스: {args.device}")
    print(f"  - 결과 저장: {save_dir.resolve()}")
    print("=" * 70)
    
    # 1. 추론 엔진 초기화
    if weights_path.suffix.lower() == ".onnx":
        detector = YOLOv11ONNXDetector(
            onnx_path=str(weights_path),
            imgsz=args.imgsz,
            conf_thresh=args.conf,
            iou_thresh=args.iou,
            device=args.device
        )
    elif weights_path.suffix.lower() == ".pt":
        if not TORCH_AVAILABLE:
            print("[오류] PyTorch(.pt) 모델을 로드하려면 torch 및 ultralytics 패키지가 필요합니다.")
            sys.exit(1)
        print("[안내] PyTorch 모델을 로드하여 ONNX 검증 모드로 실행합니다.")
        yolo_pt = YOLO(str(weights_path))
    else:
        print(f"[오류] 지원하지 않는 가중치 확장자입니다: {weights_path.suffix}")
        sys.exit(1)
        
    # 2. 입력 소스 수집
    source_path = Path(args.source)
    if source_path.is_file():
        image_files = [source_path]
    elif source_path.is_dir():
        image_files = sorted(list(source_path.glob("*.jpg")) + list(source_path.glob("*.png")))
        if args.max_images > 0:
            image_files = image_files[:args.max_images]
    else:
        print(f"[오류] 입력 소스를 찾을 수 없습니다: {source_path}")
        sys.exit(1)
        
    print(f"[안내] 총 {len(image_files)}장의 이미지에 대해 추론을 진행합니다.\n")
    
    total_latencies = []
    
    # 3. 이미지 순회 추론
    for idx, img_p in enumerate(image_files, 1):
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            print(f"[{idx}/{len(image_files)}] 이미지 로드 실패: {img_p.name}")
            continue
            
        if weights_path.suffix.lower() == ".onnx":
            result = detector.predict(img_bgr)
            boxes = result["boxes"]
            scores = result["scores"]
            latency_ms = result["latency_ms"]
            fps = result["fps"]
        else:
            # PyTorch fallback 추론
            t0 = time.perf_counter()
            pt_res = yolo_pt.predict(img_bgr, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)[0]
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0
            fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
            boxes = pt_res.boxes.xyxy.cpu().numpy().tolist() if len(pt_res.boxes) > 0 else []
            scores = pt_res.boxes.conf.cpu().numpy().tolist() if len(pt_res.boxes) > 0 else []
            
        total_latencies.append(latency_ms)
        
        # 4. 시각화 (바운딩 박스 + HUD 패널)
        det_img = draw_detections(img_bgr, boxes, scores, class_names=["pothole"])
        final_vis = draw_hud_panel(
            det_img,
            title=f"Model: {weights_path.stem}",
            latency_ms=latency_ms,
            fps=fps,
            det_count=len(boxes)
        )
        
        # 5. 결과 이미지 저장
        save_path = save_dir / f"pred_{img_p.name}"
        cv2.imwrite(str(save_path), final_vis)
        
        print(f"[{idx:02d}/{len(image_files):02d}] {img_p.name} -> 탐지 {len(boxes)}개 | 지연시간: {latency_ms:.1f}ms ({fps:.1f} FPS) -> {save_path.name}")
        
    # 종합 요약
    avg_latency = float(np.mean(total_latencies)) if total_latencies else 0.0
    avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    
    print("\n" + "=" * 70)
    print("  [추론 완료 요약]")
    print(f"  - 처리 이미지 수: {len(total_latencies)}장")
    print(f"  - 평균 추론 지연시간: {avg_latency:.2f} ms")
    print(f"  - 평균 추론 속도: {avg_fps:.1f} FPS")
    print(f"  - 결과 저장 경로: {save_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
