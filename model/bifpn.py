import torch
import torch.nn as nn
import torch.nn.functional as F

class BiFPN_Concat(nn.Module):
    """
    BiFPN (Bi-directional Feature Pyramid Network) Fast Normalized Fusion Concat 모듈
    
    원리 (EfficientDet 논문 참고):
    기존 YOLO의 Concat 레이어는 여러 레벨(P3, P4, P5 등)에서 넘어온 특징 맵(Feature Map)을 단순 결합합니다.
    BiFPN_Concat은 학습 가능한 가중치(w_i >= 0)를 각 입력 텐서마다 부여하여, 포트홀 탐지에 
    더 중요한 해상도/스케일의 특징 맵에 고가중치를 부여하도록 동적으로 융합합니다.
    
    공식:
    Fast Normalized Fusion: Output = Concat( (w_i / (epsilon + sum(w_j))) * Tensor_i )
    """
    def __init__(self, dimension: int = 1):
        """
        초기화 메서드
        
        :param dimension: Concat을 수행할 차원 축 (기본값: 1, 채널 방향)
        """
        super(BiFPN_Concat, self).__init__()
        self.d = dimension
        
        # 최대 8개까지의 입력 텐서 가중치를 수용할 수 있는 학습 가능 파라미터 선언
        # torch.ones로 초기화하여 초기에는 동일한 비중으로 융합 시작
        self.w = nn.Parameter(torch.ones(8, dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-4 # 분모 0 방지용 미소값

    def forward(self, x: list) -> torch.Tensor:
        """
        순전파 (Forward Pass) 연산
        
        :param x: Ultralytics 파서로부터 전달받은 입력 텐서 리스트 [Tensor_1, Tensor_2, ...]
        :return: 가중치가 적용되어 채널 축으로 결합된 텐서 (B, C_total, H, W)
        """
        num_inputs = len(x)
        
        # 1. 가중치가 음수가 되지 않도록 ReLU 적용 (w_i >= 0 보장)
        w = F.relu(self.w[:num_inputs])
        
        # 2. Fast Normalized Fusion 가중치 정규화: sum(w_i)로 나누어 합이 1에 가깝게 만듦
        weight_norm = w / (torch.sum(w, dim=0) + self.epsilon)

        # 3. 기준 해상도 설정 (리스트의 첫 번째 텐서 H, W 크기 기준)
        target_size = x[0].shape[2:]
        
        fused_tensors = []
        for i in range(num_inputs):
            xi = x[i]
            # 4. 해상도(H, W)가 다를 경우 Nearest Interpolation으로 동일하게 크기 맞춤
            if xi.shape[2:] != target_size:
                xi = F.interpolate(xi, size=target_size, mode='nearest')
            
            # 5. 정규화된 가중치를 텐서에 요소별 곱셈(Element-wise Multiplications)
            fused_tensors.append(weight_norm[i] * xi)

        # 6. 지정된 차원(d=1, 채널)으로 가중 융합된 텐서들을 Concat
        return torch.cat(fused_tensors, self.d)


class BiFPN_Add(nn.Module):
    """
    BiFPN Fast Normalized Fusion Add 모듈
    
    채널 수가 동일한 복수 특징 맵을 Concat으로 채널을 늘리지 않고,
    학습 가능 가중치를 곱해 Element-wise Sum(요소별 합산)으로 피처를 융합하는 모듈입니다.
    """
    def __init__(self, c1: int = None, c2: int = None):
        """
        초기화 메서드
        
        :param c1: 입력 채널 수 (Ultralytics 호환용 인자)
        :param c2: 출력 채널 수 (Ultralytics 호환용 인자)
        """
        super(BiFPN_Add, self).__init__()
        self.w = nn.Parameter(torch.ones(8, dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-4

    def forward(self, x: list) -> torch.Tensor:
        """
        순전파 연산
        
        :param x: 동일한 채널 수를 가진 입력 텐서 리스트 [Tensor_1, Tensor_2, ...]
        :return: 가중치가 적용되어 요소별 합산된 텐서 (B, C, H, W)
        """
        num_inputs = len(x)
        w = F.relu(self.w[:num_inputs])
        weight_norm = w / (torch.sum(w, dim=0) + self.epsilon)

        target_size = x[0].shape[2:]
        out = 0.0
        for i in range(num_inputs):
            xi = x[i]
            if xi.shape[2:] != target_size:
                xi = F.interpolate(xi, size=target_size, mode='nearest')
            out = out + weight_norm[i] * xi
            
        return out
