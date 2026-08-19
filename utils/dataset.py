import os
import sys
import yaml
from pathlib import Path

# Windows 터미널에서 한글 출력 깨짐(CP949 인코딩 문제)을 방지하기 위해 표준 출력을 UTF-8로 설정
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

class HRP4KDatasetChecker:
    """
    HRP4K 포트홀 탐지 데이터셋의 구조, 무결성, 통계를 분석하는 검증 클래스입니다.
    교육용 목적으로 작성되어 각 단계별 설명과 예외 처리가 포함되어 있습니다.
    """

    def __init__(self, dataset_dir="E:/ROVOROAD/datasets/HRP4K", config_path="E:/ROVOROAD/configs/hrp4k.yaml"):
        """
        초기화 메서드: 데이터셋 경로 및 설정 파일 경로를 받아 Path 객체로 변환합니다.
        
        :param dataset_dir: HRP4K 데이터셋 루트 폴더 경로
        :param config_path: Ultralytics YOLO 전용 hrp4k.yaml 설정 파일 경로
        """
        self.dataset_dir = Path(dataset_dir)
        self.config_path = Path(config_path)
        self.splits = ["train", "valid", "test"]

    def check_yaml_config(self):
        """
        [단계 1] configs/hrp4k.yaml 설정 파일의 로드 가능 여부 및 내용 구조를 검증합니다.
        """
        print("[1. hrp4k.yaml 설정 파일 검증]")
        if not self.config_path.exists():
            print(f"  [오류] YAML 설정 파일이 존재하지 않습니다: {self.config_path}")
            return False

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            
            print(f"  - Dataset Root Path: {cfg.get('path')}")
            print(f"  - Train Path       : {cfg.get('train')}")
            print(f"  - Valid Path       : {cfg.get('val')}")
            print(f"  - Test Path        : {cfg.get('test')}")
            print(f"  - Number of Classes: {cfg.get('nc')}")
            print(f"  - Class Names      : {cfg.get('names')}")
            print("  -> YAML 설정 파일 정상 확인 완료.\n")
            return True
        except Exception as e:
            print(f"  [오류] YAML 파일 로드 중 예외가 발생했습니다: {e}\n")
            return False

    def check_dataset_structure(self):
        """
        [단계 2] HRP4K 데이터셋의 디렉터리 구조, 이미지-라벨 1:1 매칭 무결성 및 바운딩 박스 개수를 검증합니다.
        """
        print("[2. HRP4K 데이터셋 무결성 및 통계 점검]")
        if not self.dataset_dir.exists():
            print(f"  [오류] 데이터셋 경로가 존재하지 않습니다: {self.dataset_dir}")
            return False

        summary_data = {}

        for split in self.splits:
            split_dir = self.dataset_dir / split
            img_dir = split_dir / "images"
            label_dir = split_dir / "labels"
            json_file = self.dataset_dir / f"{split}.json"

            # 이미지 및 라벨 파일 수집
            img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) if img_dir.exists() else []
            label_files = list(label_dir.glob("*.txt")) if label_dir.exists() else []
            has_json = json_file.exists()

            # 확장자 제외 파일명(stem) 수집
            img_map = {f.stem: f for f in img_files}
            label_map = {f.stem: f for f in label_files}

            img_stems = set(img_map.keys())
            label_stems = set(label_map.keys())

            # 이미지와 라벨의 매칭 관계 조사
            matched_stems = img_stems & label_stems
            missing_labels = img_stems - label_stems
            orphan_labels = label_stems - img_stems

            # 유효한 이미지에 해당하는 라벨 파일(matched_stems)에 대한 분석 진행
            positive_count = 0
            negative_count = 0
            total_bboxes = 0

            for stem in matched_stems:
                lbl_file = label_map[stem]
                if lbl_file.stat().st_size == 0:
                    # 라벨 파일 크기가 0B인 경우 (배경/Negative 이미지)
                    negative_count += 1
                else:
                    # 라벨 파일 내 바운딩 박스가 존재하는 경우 (Positive 이미지)
                    positive_count += 1
                    with open(lbl_file, 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f if line.strip()]
                        total_bboxes += len(lines)

            summary_data[split] = {
                "images_count": len(img_files),
                "labels_count": len(label_files),
                "matched_count": len(matched_stems),
                "missing_labels": len(missing_labels),
                "orphan_labels": len(orphan_labels),
                "has_json": has_json,
                "positive_samples": positive_count,
                "negative_samples": negative_count,
                "total_bboxes": total_bboxes
            }

            print(f"  [{split.upper()} Split 분석 리포트]")
            print(f"    * 이미지 파일 총 개수: {len(img_files):,} 장")
            print(f"    * 라벨 파일 총 개수  : {len(label_files):,} 개")
            print(f"    * 이미지-라벨 1:1 매칭 : {len(matched_stems):,} 장 (매칭률 100.0%)" if len(missing_labels) == 0 else f"    * 라벨 누락 이미지    : {len(missing_labels):,} 장")
            if len(orphan_labels) > 0:
                print(f"    * 이미지 없는 고아 라벨: {len(orphan_labels):,} 개 (학습 시 무시됨)")
            print(f"    * COCO JSON 메타데이터 : {'존재함' if has_json else '없음'}")
            print(f"    * 양성 이미지 (포트홀 O): {positive_count:,} 장")
            print(f"    * 음성 이미지 (포트홀 X): {negative_count:,} 장")
            print(f"    * 총 포트홀 바운딩박스 수: {total_bboxes:,} 개\n")

        return summary_data

def run_dataset_check():
    """
    데이터셋 검증 실행 함수 메인 엔트리 포인트
    """
    checker = HRP4KDatasetChecker()
    print("==========================================================")
    print("        ROVOROAD HRP4K 데이터셋 검증 파이프라인           ")
    print("==========================================================\n")
    
    yaml_status = checker.check_yaml_config()
    summary = checker.check_dataset_structure()
    
    print("==========================================================")
    print("                 데이터셋 최종 요약 리포트                 ")
    print("==========================================================")
    print(f"{'SPLIT':<6} | {'이미지 수':<8} | {'매칭 라벨':<8} | {'양성 (포트홀O)':<12} | {'음성 (포트홀X)':<12} | {'총 객체수':<8}")
    print("----------------------------------------------------------")
    for split, data in summary.items():
        print(f"{split.upper():<6} | {data['images_count']:>8,}장 | {data['matched_count']:>8,}개 | {data['positive_samples']:>12,}장 | {data['negative_samples']:>12,}장 | {data['total_bboxes']:>8,}개")
    print("==========================================================\n")

if __name__ == "__main__":
    run_dataset_check()
