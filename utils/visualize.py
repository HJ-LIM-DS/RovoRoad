"""
ROVOROAD Pothole Detection 프로젝트 - Phase 6: 시각화 및 영상 합성 유틸리티 모듈 (visualize.py)

[ 모듈 설명 ]
이 모듈은 객체 탐지(Object Detection) 모델의 추론 결과를 시각화하고,
원본 이미지와의 1:1 비교(Side-by-Side) 및 4개 모델 동시 비교(2x2 Grid) 화면을
고화질 비디오 프레임으로 렌더링하기 위한 다양한 그래픽 유틸리티 함수들을 제공합니다.

[ 초보자를 위한 주요 기능 안내 ]
1. draw_detections: 감지된 포트홀 좌표에 바운딩 박스와 신뢰도(Confidence) 라벨을 그립니다.
2. draw_hud_panel: 화면 상단에 모델명, 추론 시간(ms), FPS, 탐지 개수 정보를 깔끔한 반투명 패널로 오버레이합니다.
3. create_side_by_side_frame: 좌측(원본 이미지)과 우측(탐지 결과 이미지)을 나란히 배치하여 1:1 비교 프레임을 만듭니다.
4. create_2x2_grid_frame: 4개 모델(Baseline, CBAM, BiFPN, Full)의 탐지 결과를 2x2 격자로 합성합니다.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


def draw_hud_panel(
    image: np.ndarray,
    title: str,
    latency_ms: Optional[float] = None,
    fps: Optional[float] = None,
    det_count: Optional[int] = None,
    bg_color: Tuple[int, int, int] = (20, 20, 20),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    accent_color: Tuple[int, int, int] = (0, 215, 255),
    alpha: float = 0.65,
    panel_height: int = 44
) -> np.ndarray:
    """
    이미지 상단에 반투명 HUD(Heads-Up Display) 정보 패널을 그립니다.
    
    매개변수:
        image (np.ndarray): 대상 BGR 이미지
        title (str): 패널 좌측에 표시할 타이틀 (예: 'Model: YOLO11m-Full (ONNX)' 또는 'Original Image')
        latency_ms (float, optional): 순수 추론 지연시간 (단위: 밀리초)
        fps (float, optional): 초당 프레임 수
        det_count (int, optional): 탐지된 포트홀 개수
        bg_color (tuple): 배경 색상 (B, G, R)
        text_color (tuple): 기본 텍스트 색상 (B, G, R)
        accent_color (tuple): 강조 텍스트 색상 (B, G, R) - 노란색/금색 계열
        alpha (float): 패널의 투명도 (0.0: 완전투명 ~ 1.0: 불투명)
        panel_height (int): 상단 패널 높이(픽셀)
        
    반환값:
        np.ndarray: HUD 패널이 합성된 이미지
    """
    img_h, img_w = image.shape[:2]
    overlay = image.copy()
    
    # 1. 상단 반투명 직사각형 배경 그리기
    cv2.rectangle(overlay, (0, 0), (img_w, panel_height), bg_color, -1)
    # 하단 얇은 포인트 테두리 선 추가
    cv2.line(overlay, (0, panel_height), (img_w, panel_height), accent_color, 2)
    
    # 원본 이미지와 반투명 블렌딩
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # 2. 폰트 및 스케일 설정
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65 if img_w >= 640 else 0.45
    thickness = 2
    
    # 3. 좌측 타이틀 텍스트 출력
    cv2.putText(
        image,
        title,
        (14, int(panel_height * 0.68)),
        font,
        font_scale,
        accent_color,
        thickness,
        cv2.LINE_AA
    )
    
    # 4. 우측 메트릭 정보 텍스트 조합 (추론 시간, FPS, 탐지 개수)
    metric_items = []
    if latency_ms is not None:
        metric_items.append(f"Latency: {latency_ms:.1f}ms")
    if fps is not None:
        metric_items.append(f"FPS: {fps:.1f}")
    if det_count is not None:
        metric_items.append(f"Potholes: {det_count}")
        
    if metric_items:
        metric_text = " | ".join(metric_items)
        # 텍스트 크기 계산하여 우측 정렬
        text_size, _ = cv2.getTextSize(metric_text, font, font_scale, thickness)
        text_x = img_w - text_size[0] - 14
        if text_x > 0:
            cv2.putText(
                image,
                metric_text,
                (text_x, int(panel_height * 0.68)),
                font,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA
            )
            
    return image


def draw_detections(
    image: np.ndarray,
    boxes: List[List[float]],
    scores: List[float],
    class_names: Optional[List[str]] = None,
    box_color: Tuple[int, int, int] = (0, 140, 255),
    text_bg_color: Tuple[int, int, int] = (0, 140, 255),
    text_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    이미지에 탐지된 바운딩 박스(Bounding Box)와 신뢰도(Confidence) 라벨을 그립니다.
    
    매개변수:
        image (np.ndarray): 대상 BGR 이미지
        boxes (list): [x1, y1, x2, y2] 좌표 리스트
        scores (list): 각 박스의 신뢰도 점수 (0.0 ~ 1.0)
        class_names (list, optional): 클래스 이름 리스트 (기본값: ['pothole'])
        box_color (tuple): 바운딩 박스 색상 (B, G, R)
        text_bg_color (tuple): 텍스트 라벨 배경 색상
        text_color (tuple): 텍스트 글자 색상
        
    반환값:
        np.ndarray: 바운딩 박스가 그려진 이미지
    """
    result_img = image.copy()
    if class_names is None:
        class_names = ["pothole"]
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        
        # 1. 바운딩 박스 테두리 그리기
        cv2.rectangle(result_img, (x1, y1), (x2, y2), box_color, 2)
        
        # 박스 네 모서리에 강조 코너 라인 추가 (시인성 강화)
        corner_len = min(15, max(5, (x2 - x1) // 4, (y2 - y1) // 4))
        # 좌상단
        cv2.line(result_img, (x1, y1), (x1 + corner_len, y1), (0, 255, 255), 3)
        cv2.line(result_img, (x1, y1), (x1, y1 + corner_len), (0, 255, 255), 3)
        # 우상단
        cv2.line(result_img, (x2, y1), (x2 - corner_len, y1), (0, 255, 255), 3)
        cv2.line(result_img, (x2, y1), (x2, y1 + corner_len), (0, 255, 255), 3)
        # 좌하단
        cv2.line(result_img, (x1, y2), (x1 + corner_len, y2), (0, 255, 255), 3)
        cv2.line(result_img, (x1, y2), (x1, y1 - corner_len if y1 - corner_len > y1 else y2 - corner_len), (0, 255, 255), 3)
        # 우하단
        cv2.line(result_img, (x2, y2), (x2 - corner_len, y2), (0, 255, 255), 3)
        cv2.line(result_img, (x2, y2), (x2, y2 - corner_len), (0, 255, 255), 3)
        
        # 2. 라벨 텍스트 생성 (예: "pothole 0.89")
        cls_name = class_names[0] if class_names else "pothole"
        label_text = f"{cls_name} {score:.2f}"
        
        # 라벨 배경 직사각형 크기 계산
        (tw, th), baseline = cv2.getTextSize(label_text, font, 0.5, 1)
        label_y1 = max(0, y1 - th - 6)
        label_y2 = y1
        label_x1 = x1
        label_x2 = x1 + tw + 6
        
        # 라벨 배경 채우기
        cv2.rectangle(result_img, (label_x1, label_y1), (label_x2, label_y2), text_bg_color, -1)
        # 라벨 텍스트 출력
        cv2.putText(
            result_img,
            label_text,
            (label_x1 + 3, label_y2 - 3),
            font,
            0.5,
            text_color,
            1,
            cv2.LINE_AA
        )
        
    return result_img


def create_side_by_side_frame(
    original_img: np.ndarray,
    detected_img: np.ndarray,
    orig_title: str = "ORIGINAL IMAGE (INPUT)",
    det_title: str = "YOLO11m-Full (ONNX)",
    latency_ms: Optional[float] = None,
    fps: Optional[float] = None,
    det_count: Optional[int] = None,
    target_width: int = 1920,
    target_height: int = 1080
) -> np.ndarray:
    """
    원본 이미지와 탐지 결과 이미지를 1:1 좌우로 배치하여 Full HD 해상도의 Side-by-Side 프레임을 만듭니다.
    
    매개변수:
        original_img (np.ndarray): 원본 입력 이미지
        detected_img (np.ndarray): 탐지 결과가 그려진 이미지
        orig_title (str): 좌측 원본 화면 타이틀
        det_title (str): 우측 탐지 모델 화면 타이틀
        latency_ms (float, optional): 모델 추론 지연시간 (ms)
        fps (float, optional): 초당 프레임 수
        det_count (int, optional): 탐지된 포트홀 개수
        target_width (int): 출력 프레임 가로 해상도 (기본값: 1920)
        target_height (int): 출력 프레임 세로 해상도 (기본값: 1080)
        
    반환값:
        np.ndarray: 좌우가 합성된 1920x1080 비디오 프레임
    """
    # 1. 각각의 하위 뷰 크기 계산 (가운데 구분선 4px 고려)
    divider_w = 4
    half_w = (target_width - divider_w) // 2
    view_h = target_height
    
    # 2. 이미지 리사이즈 (가로세로 비율 유지 및 레터박스 패딩)
    resized_orig = fit_into_box(original_img, half_w, view_h)
    resized_det = fit_into_box(detected_img, half_w, view_h)
    
    # 3. 각각의 이미지 상단에 HUD 패널 부착
    # 원본 이미지에는 추론 시간 없이 깔끔한 안내 타이틀만 부착
    hud_orig = draw_hud_panel(
        resized_orig,
        title=orig_title,
        bg_color=(30, 30, 30),
        accent_color=(180, 180, 180),
        panel_height=48
    )
    # 탐지 결과 이미지에는 모델명, 지연시간, FPS, 탐지 개수 부착
    hud_det = draw_hud_panel(
        resized_det,
        title=det_title,
        latency_ms=latency_ms,
        fps=fps,
        det_count=det_count,
        bg_color=(20, 20, 20),
        accent_color=(0, 215, 255),
        panel_height=48
    )
    
    # 4. 중앙 구분선(Divider) 생성
    divider = np.zeros((view_h, divider_w, 3), dtype=np.uint8)
    divider[:] = (0, 215, 255)  # 노란색 구분선
    
    # 5. 좌우 결합하여 1개의 프레임 완성
    combined_frame = np.hstack([hud_orig, divider, hud_det])
    
    # 최종 해상도 보정
    if combined_frame.shape[1] != target_width or combined_frame.shape[0] != target_height:
        combined_frame = cv2.resize(combined_frame, (target_width, target_height))
        
    return combined_frame


def create_2x2_grid_frame(
    frames_info: List[Dict[str, Any]],
    target_width: int = 1920,
    target_height: int = 1080
) -> np.ndarray:
    """
    4개 모델의 추론 결과를 2x2 격자로 배치하여 Full HD 해상도의 종합 비교 프레임을 만듭니다.
    
    매개변수:
        frames_info (list): 4개 모델의 정보가 담긴 딕셔너리 리스트
            각 딕셔너리는 다음 키를 포함해야 합니다:
            - 'image': np.ndarray (탐지 결과 이미지)
            - 'title': str (모델명)
            - 'latency_ms': float (지연시간)
            - 'fps': float (초당 프레임)
            - 'det_count': int (탐지 개수)
        target_width (int): 출력 프레임 가로 해상도 (기본값: 1920)
        target_height (int): 출력 프레임 세로 해상도 (기본값: 1080)
        
    반환값:
        np.ndarray: 2x2 격자로 합성된 1920x1080 비디오 프레임
    """
    if len(frames_info) < 4:
        raise ValueError("2x2 그리드를 만들기 위해서는 정확히 4개의 모델 정보가 필요합니다.")
        
    grid_w = target_width // 2
    grid_h = target_height // 2
    
    processed_cells = []
    
    # 각 셀 색상 테마 지정
    themes = [
        {"accent": (255, 180, 0),   "bg": (25, 25, 25)},   # Baseline: 청록/하늘
        {"accent": (0, 200, 255),   "bg": (25, 25, 25)},   # +CBAM: 골드
        {"accent": (100, 255, 100), "bg": (25, 25, 25)},   # +BiFPN: 연두
        {"accent": (0, 100, 255),   "bg": (25, 25, 25)}    # Full: 주황
    ]
    
    for i in range(4):
        item = frames_info[i]
        theme = themes[i % len(themes)]
        
        # 1. 셀 크기에 맞게 이미지 리사이즈
        cell_img = fit_into_box(item["image"], grid_w, grid_h)
        
        # 2. 각 셀 상단에 HUD 부착
        hud_cell = draw_hud_panel(
            cell_img,
            title=item.get("title", f"Model {i+1}"),
            latency_ms=item.get("latency_ms"),
            fps=item.get("fps"),
            det_count=item.get("det_count"),
            bg_color=theme["bg"],
            accent_color=theme["accent"],
            panel_height=42
        )
        processed_cells.append(hud_cell)
        
    # 3. 상단 2개(좌, 우) 및 하단 2개(좌, 우) 가로 결합
    top_row = np.hstack([processed_cells[0], processed_cells[1]])
    bottom_row = np.hstack([processed_cells[2], processed_cells[3]])
    
    # 4. 상하 결합하여 2x2 그리드 완성
    combined_grid = np.vstack([top_row, bottom_row])
    
    # 5. 중앙 십자선 구분선 추가
    cv2.line(combined_grid, (grid_w, 0), (grid_w, target_height), (0, 215, 255), 3)
    cv2.line(combined_grid, (0, grid_h), (target_width, grid_h), (0, 215, 255), 3)
    
    return combined_grid


def fit_into_box(
    image: np.ndarray,
    box_w: int,
    box_h: int,
    bg_color: Tuple[int, int, int] = (15, 15, 15)
) -> np.ndarray:
    """
    이미지의 종횡비(Aspect Ratio)를 그대로 유지하면서 목표 직사각형 영역(box_w, box_h)에
    검은색/어두운 배경 패딩(Letterbox)을 추가하여 정확히 맞춥니다.
    
    매개변수:
        image (np.ndarray): 입력 BGR 이미지
        box_w (int): 목표 영역 너비
        box_h (int): 목표 영역 높이
        bg_color (tuple): 패딩 배경 색상
        
    반환값:
        np.ndarray: (box_h, box_w, 3) 크기의 패딩 처리된 이미지
    """
    h, w = image.shape[:2]
    scale = min(box_w / w, box_h / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    canvas = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    canvas[:] = bg_color
    
    # 가운데 정렬 배치
    start_x = (box_w - new_w) // 2
    start_y = (box_h - new_h) // 2
    
    canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized
    return canvas
