"""
ROVOROAD Model Package
포트홀 탐지 AI 로봇을 위한 커스텀 딥러닝 모듈 패키지
"""

from .cbam import CBAM, ChannelAttention, SpatialAttention
from .bifpn import BiFPN_Concat, BiFPN_Add
from .register import register_custom_modules

# 패키지 로드 시 레지스트리 자동 실행
register_custom_modules()

__all__ = [
    "CBAM",
    "ChannelAttention",
    "SpatialAttention",
    "BiFPN_Concat",
    "BiFPN_Add",
    "register_custom_modules"
]
