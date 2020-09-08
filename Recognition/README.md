# Korea Landmarks Dataset

본 저장소는 한국정보화진흥원 `랜드마크 이미지 데이터`의 Instance-Level Recognition task 검수용으로 사용된 인공지능 모델을 구현하기 위한 저장소입니다.


## Dataset Download

랜드마크 이미지 데이터는 공공 데이터 구축을 목적으로 하는 [AI Hub](http://www.aihub.or.kr/)에서 제공됩니다.



## Introduction

Large-Scale에서의 Instance-Level Recognition task는 컴퓨터 비전 분야에서 활발히 연구되고 있는 분야중 하나입니다. 000하였습니다.
이를 구현하기 위한 소스코드를 공개합니다.

## Installation

[Install Instruction](INSTALL_INSTRUCTION)

## Model
![ResNet 101](docs/imgs/resnet101.png)

[![Paper](http://img.shields.io/badge/paper-arXiv.2004.01804-B3181B.svg)](https://arxiv.org/abs/2004.01804)

## Evaluate

### matrix
모델 평가를 위한 지표는 Global Average Precision(GAP) 를 사용합니다. 


$$GAP=\frac{1}{M}\sum_{i=1}^N P(i) rel(i)$$

- N is the total number of predictions returned by the system, across all queries
- M is the total number of queries with at least one landmark from the training set visible in it (note that some queries may not depict landmarks)
- P(i) is the precision at rank i
- rel(i) denotes the relevance of prediciton i: it’s 1 if the i-th prediction is correct, and 0 otherwise

[![Paper](http://img.shields.io/badge/paper-arXiv.2004.01804-B3181B.svg)](https://arxiv.org/abs/2004.01804) -- 수정필요
```
"A Family of Contextual Measures of Similarity between Distributions with Application to Image Retrieval," 
F. Perronnin, Y. Liu, and J.-M. Renders, 
Proc. CVPR'09"
```

### Performance

|GAP



## Reference
[![Paper](http://img.shields.io/badge/paper-arXiv.2004.01804-B3181B.svg)](https://arxiv.org/abs/2004.01804)

```
"Google Landmarks Dataset v2 - A Large-Scale Benchmark for Instance-Level Recognition and Retrieval"
T. Weyand*, A. Araujo*, B. Cao, J. Sim,
Proc. CVPR'20
```

## 응용 서비스
응용 서비스는 [페이지](http://15.165.113.21:8080)에서 확인 가능합니다.
