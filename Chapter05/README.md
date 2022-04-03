# Convolution 기법들

## 그전에...
2차원 컨볼루션 세 가지 문제점이 존재한다.
- Expensive Cost
- Dead Channels
- Low Correlation between channels

또한 영상 내의 객체에 대한 정확한 판단을 하기 위해서는 **Contextual Information**이 중요하다. Object Detection이나 Object Segment에서는 충분한 Contextual Information을 확보하기 위해 상대적으로 넓은 Receptive Field를 고려할 필요가 있다.

단순히 더 많은 convolution layer를 많이 쌓거나 kernel size를 확장하는 방법은 연산량을 크게 증가하므로 적절하지 않다.

**연산량을 경량화하면서 정보 손실이 일어나지 않게끔 유의미한 정보만을 추출하기 위해 다양한 convolution 기법들이 등장하였다.**

## Convolution 기법들
### 1. Convlolution
### 2. Dilated Convolutions
### 3. Transpose Convolutions
### 4. Separable Convolution
### 5. Depthwise Convolution
### 6. Depthwise Separable Convolution
### 7. Pointwise Convolution
### 8. Grouped Convolution
### 9. Deformable Convolution



