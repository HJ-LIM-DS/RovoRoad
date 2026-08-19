import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """
    CBAM의 채널 어텐션 모듈 (Channel Attention Module)
    
    입력 특징 맵의 어떤 채널(Channel)에 중요한 정보가 집중되어 있는지를 학습합니다.
    기능 요약:
    1. 입력 텐서(B, C, H, W)에 대해 공간 축(H, W)으로 평균 풀링(AvgPool)과 최대 풀링(MaxPool)을 각각 수행합니다.
    2. 생성된 두 텐서(B, C, 1, 1)를 공유 MLP(1x1 합성곱 레이어)에 통과시킵니다.
    3. 두 결과를 합산(Element-wise Sum)한 후 Sigmoid 활성화 함수를 적용하여 채널별 가중치(0 ~ 1)를 생성합니다.
    4. 원래 입력 텐서와 생성된 가중치를 곱하여 중요한 채널 정보는 강화하고 불필요한 채널 정보는 억제합니다.
    """
    def __init__(self, channels: int, reduction: int = 16):
        """
        초기화 메서드
        
        :param channels: 입력 특징 맵의 채널 수 (C)
        :param reduction: 공유 MLP 내부 차원 축소 비율 (기본값: 16)
        """
        super(ChannelAttention, self).__init__()
        
        # 최소 차원을 1 이상으로 유지하기 위해 max 함수 사용
        reduced_channels = max(1, channels // reduction)
        
        # 평균 풀링과 최대 풀링 텐서가 공통으로 통과하는 Shared MLP 구성을 1x1 Conv로 구현
        self.fc1 = nn.Conv2d(in_channels=channels, out_channels=reduced_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels=reduced_channels, out_channels=channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # 풀링 레이어 정의
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파 (Forward Pass) 연산
        
        :param x: 입력 텐서 (배치 크기 B, 채널 C, 높이 H, 너비 W)
        :return: 채널 어텐션 가중치가 적용된 텐서 (B, C, H, W)
        """
        # 1. 평균 풀링 수행: Shape (B, C, H, W) -> (B, C, 1, 1)
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        
        # 2. 최대 풀링 수행: Shape (B, C, H, W) -> (B, C, 1, 1)
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        
        # 3. 두 풀링 결과 합산 및 Sigmoid 적용으로 가중치 맵 생성: (B, C, 1, 1)
        out = self.sigmoid(avg_out + max_out)
        
        # 4. 입력 텐서와 어텐션 가중치 요소별 곱(Element-wise Multiplication)
        return x * out


class SpatialAttention(nn.Module):
    """
    CBAM의 공간 어텐션 모듈 (Spatial Attention Module)
    
    입력 특징 맵의 어떤 위치(Spatial location: H, W)에 중요한 객체(포트홀 등)가 존재하는지를 학습합니다.
    기능 요약:
    1. 입력 텐서(B, C, H, W)의 채널 축(C)을 따라 평균(Mean) 및 최대값(Max)을 계산하여 2채널 텐서(B, 2, H, W)를 만듭니다.
    2. 커널 크기 7x7 합성곱(Conv2d)을 적용하여 1채널 텐서(B, 1, H, W)로 변환합니다.
    3. Sigmoid 활성화 함수를 통해 공간상의 중요도를 나타내는 가중치 맵(0 ~ 1)을 얻습니다.
    4. 입력 텐서에 위치별 가중치를 곱해 객체가 위치한 주요 영역을 강조합니다.
    """
    def __init__(self, kernel_size: int = 7):
        """
        초기화 메서드
        
        :param kernel_size: 공간 어텐션을 위한 합성곱 커널 크기 (기본값: 7)
        """
        super(SpatialAttention, self).__init__()
        
        # 커널 크기에 맞춘 패딩 계산 (7x7 커널 시 padding=3)
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파 (Forward Pass) 연산
        
        :param x: 입력 텐서 (B, C, H, W)
        :return: 공간 어텐션 가중치가 적용된 텐서 (B, C, H, W)
        """
        # 1. 채널 축(dim=1) 기준 평균 텐서 추출: Shape (B, 1, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        
        # 2. 채널 축(dim=1) 기준 최대값 텐서 추출: Shape (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 3. 평균 및 최대값 텐서를 채널 방향으로 결합: Shape (B, 2, H, W)
        scale = torch.cat([avg_out, max_out], dim=1)
        
        # 4. 7x7 Conv 및 Sigmoid 연산으로 공간 어텐션 맵 생성: Shape (B, 1, H, W)
        scale = self.sigmoid(self.conv(scale))
        
        # 5. 입력 텐서와 공간 어텐션 가중치 곱셈
        return x * scale


class CBAM(nn.Module):
    """
    CBAM (Convolutional Block Attention Module) 통합 모듈
    
    채널 어텐션(Channel Attention)과 공간 어텐션(Spatial Attention)을 직렬(Sequential)로 연결하여
    특징 맵의 중요한 채널과 공간 위치를 동시에 적응적으로 강조합니다.
    
    Ultralytics parse_model 호환 시그니처: (c1, ratio=16, kernel_size=7)
    parse_model이 입력 채널 c1(ch[f])을 첫 번째 인자로 전달합니다.
    """
    def __init__(self, c1: int, ratio: int = 16, kernel_size: int = 7):
        """
        초기화 메서드
        
        :param c1: 입력 특징 맵 채널 수 (parse_model에서 ch[f] 전달)
        :param ratio: Channel Attention의 채널 축소 비율 (기본값: 16)
        :param kernel_size: Spatial Attention의 커널 크기 (기본값: 7)
        """
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels=c1, reduction=ratio)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파 (Forward Pass) 연산
        
        입력 x -> 채널 어텐션 통과 -> 공간 어텐션 통과 -> 최종 출력 (입출력 채널 동일)
        :param x: 입력 텐서 (B, C, H, W)
        :return: 어텐션이 적용된 출력 텐서 (B, C, H, W)
        """
        x = self.ca(x)
        x = self.sa(x)
        return x
