import sys
from copy import deepcopy
import torch
import torch.nn as nn
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules

from .cbam import CBAM, ChannelAttention, SpatialAttention
from .bifpn import BiFPN_Concat, BiFPN_Add

_original_parse_model = tasks.parse_model

def custom_parse_model(d, ch, verbose=True):
    """
    Ultralytics parse_model 래퍼 함수
    
    YOLO YAML 파일 파싱 시, CBAM 및 BiFPN_Concat/Add 등 커스텀 모듈의 
    입력 채널(c1)과 멀티스케일 가중 융합(Fast Normalized Fusion)이 
    완벽하게 동작하도록 동적 텐서 그래프 및 파라미터를 보정합니다.
    """
    d_copy = deepcopy(d)
    bifpn_indices = []
    cbam_indices = []
    
    all_layers = d_copy.get("backbone", []) + d_copy.get("head", [])
    for idx, layer in enumerate(all_layers):
        module_name = layer[2]
        if module_name in ("BiFPN_Concat", "BiFPN_Add"):
            bifpn_indices.append((idx, module_name))
            layer[2] = "Concat"
        elif module_name == "CBAM":
            cbam_indices.append((idx, layer[3]))
            layer[2] = "nn.Identity"
            layer[3] = []

    # 표준 parse_model 실행으로 모델 뼈대 구축
    model_list, save_list = _original_parse_model(d_copy, ch, verbose)
    
    # 1. BiFPN 모듈 치환 및 속성 전파
    for idx, module_name in bifpn_indices:
        old_m = model_list[idx]
        if module_name == "BiFPN_Concat":
            new_m = BiFPN_Concat(dimension=1)
        else:
            new_m = BiFPN_Add()
            
        new_m.i = old_m.i
        new_m.f = old_m.f
        new_m.type = f"model.bifpn.{module_name}"
        new_m.np = old_m.np
        model_list[idx] = new_m

    # 2. CBAM 모듈 동적 인스턴스화 및 입출력 채널(c1) 동기화
    from .cbam import CBAM
    if cbam_indices:
        dummy = torch.zeros(1, ch, 640, 640)
        y = []
        for i, m in enumerate(model_list):
            f = m.f
            if f == -1:
                x = dummy if i == 0 else y[-1]
            elif isinstance(f, int):
                x = y[f]
            else:
                x = [y[j] for j in f]
            
            if i in [idx for idx, _ in cbam_indices]:
                orig_args = dict(cbam_indices)[i]
                c1 = x.shape[1]
                
                # 인자 파싱 보호: YAML에 c1이 첫 번째 인자로 포함된 케이스 예방
                if len(orig_args) >= 3:
                    # [c1, ratio, kernel_size]
                    ratio = orig_args[1]
                    k_size = orig_args[2]
                elif len(orig_args) == 2:
                    # [ratio, kernel_size]
                    ratio = orig_args[0]
                    k_size = orig_args[1]
                elif len(orig_args) == 1:
                    # [ratio]
                    ratio = orig_args[0]
                    k_size = 7
                else:
                    ratio = 16
                    k_size = 7
                    
                cbam_m = CBAM(c1=c1, ratio=ratio, kernel_size=k_size)
                cbam_m.i = m.i
                cbam_m.f = m.f
                cbam_m.type = "model.cbam.CBAM"
                cbam_m.np = sum(p.numel() for p in cbam_m.parameters())
                model_list[i] = cbam_m
                x = cbam_m(x)
            else:
                x = m(x)
            y.append(x)
        
    return model_list, save_list


def register_custom_modules():
    """
    Ultralytics YOLO 프레임워크 레지스트리에 커스텀 모듈 등록
    """
    custom_classes = [
        CBAM,
        ChannelAttention,
        SpatialAttention,
        BiFPN_Concat,
        BiFPN_Add
    ]
    
    for cls in custom_classes:
        name = cls.__name__
        setattr(nn, name, cls)
        setattr(tasks, name, cls)
        setattr(modules, name, cls)
        if hasattr(tasks, 'parse_model') and hasattr(tasks.parse_model, '__globals__'):
            tasks.parse_model.__globals__[name] = cls

    tasks.parse_model = custom_parse_model
    print("[ROVOROAD] Ultralytics 커스텀 모듈 동적 레지스트리 및 parse_model 패치 등록 완료")

# 모듈 로드 시 자동 등록 실행
register_custom_modules()
