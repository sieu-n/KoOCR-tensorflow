
# DCGAN-CelebA-tensorflow (Korean README)

Tensorflow 딥러닝 기반의 오픈 소스 한글 OCR 엔진.
Open-source Korean OCR engine based on Tensorflow, deep-learning.

## 개요
- 중국어, 일본어 등 유사한 언어의 뛰어난 OCR 인식 성능에 비해 한글 인식에 대해서는 활발한 연구가 이루어지 않았다.
- 쉽게 사용 가능한 고성능의 한글 OCR 프로젝트, 라이브러리가 많지 않았다. 
- 중국어 인식(HCCR)등에 사용된 학습 방법, Model Architecture를 한글 인식에 적용하고 성능을 비교하였다.
- 한글의 특수한 구조에 기인해 초성, 중성, 종성을 각각 따로 예측하는 Multi-output 모델을 구성했다. 
- 학습 완료된 모델(logs/weights.h5)과 학습에 사용된 코드 전체를 공개하고 있다. 


##  Method and Plans


- [x]  DirectMap: Online and Offline Handwritten Chinese Character Recognition: A Comprehensive Study and New Benchmark
- [x]  Fire-module based model, GWAP: Building Efficient CNN Architecture for Offline Handwritten Chinese Character Recognition
- [x] High-performance network architecture, CAM, GAP/GWAP/GWOAP: A High-Performance CNN Method for Offline Handwritten Chinese Character Recognition and Visualization
- [ ] Hybrid learning loss: Improving Discrimination Ability of Convolutional Neural Networks by Hybrid Learning
- [ ] Adaptive Drop Weight, GSLRE: Building Fast and Compact Convolutional Neural Networks for Offline Handwritten Chinese Character Recognition
- [ ] Adversarial Feature Learning: Robust offline handwritten character recognition through exploring writer-independent features under the guidance of printed data
- [ ] DenseRAN style model: DenseRAN for Offline Handwritten Chinese Character Recognition
- [ ] Iterative Refinemet: Improving Offline Handwritten Chinese Character Recognition by Iterative Refinement

위 논문들에서 제시하는 method를 구현하고 한글 인식에 대한 성능 개선의 정도를 평가하는 논문을 작성할 계획이다. 

## 프로젝트 사용

### load_data.py
```
!python load_data.py --sevenzip=true
```
Google Drive에 업로드된 데이터셋을 다운로드 받아 `./data`, `./val_data` 에 저장한다. 데이터셋은 300여개의 .pickle 패치로 이루어져 있고, 데이터의 특성에 따라 `handwritten_*.pickle`, `printed_*.pickle`, `clova_*.pickle` 으로 이름지어져있다. 각각 손글씨, 인쇄체, 손글씨 폰트의 이미지를 나타낸다. 

`sevenzip` 변수는 압축된 데이터를 .7z 파일로 받을지 .zip 파일로 받을지를 나타낸다. True 값이 다운로드와 압축 속도가 빠르다. 

### crawl_data.py

```
!python crawl_data.py 	--AIHub=true 
						--clova=true
						--image_size=96
						--x_offset=8
						--y_offset=8
						--char_size=80   
```
데이터셋을 크롤링해서 다운로드받는다. load_data.py와 같은 역할을 한다. `x_offset`, `y_offset`, `char_size` 변수는 폰트를 이미지에 그릴 때 위치의 offset과 문자의 크기를 지정한다. 아래 표는 실험에서 사용한 이미지 크기에 따른 변수 설정값이다. 
 
image size | x_offset | y_offset | char_size
---------- | -------- | -------- | ---------
64         |         5|         5|50
96         |8         |8         |80
128        |14        |10        |100
256        |50        |10        |200


### model. py
```
import model
OCR_model=model.KoOCR(weights='C:\\...', split_components=True, ...)

OCR_model.model.summary()
OCR_model.train(epochs=10, lr=0.01, ...)

pred=OCR_model.predict(image, n=5)
```
모델을 정의하는 모듈으로 `KoOCR` class가 정의되어 있다. **추론에 사용되는 메소드 `KoOCR.predict`는 이미지 혹은 이미지의 배치를 입력받아 가능성이 가장 높은 top-n 개의 한글 글자를 반환한다.** 모델의 추가적인 학습에 사용되는 메소드는 `KoOCR.train`으로, 입력받은 Hyperparameter를 바탕으로 학습을 진행한다. 

### train. py
```
!python train.py 	--split_components=true 
					--network=melnyk
					--image_size=96
					--direct_map=true
					--epochs=10
					...
```
 학습을 진행하는 파이썬 모듈인지만, 모델을 정의하고 `KoOCR.train`을 호출하는 역할을 할 뿐, 직접 `model.py`를 import 하고 훈련하는 것과 차이가 없다. 학습결과와 과정에 대한 모든 정보는 `./logs`에 저장되고, 가중치는 매 에포크마다 `./logs/weights.h5`에 저장된다. 
 
### evaluate. py
```
python evaluate.py	--weights='./logs/weights.h5'
					--accuracy=true
					--confusion_matrix=true
					--class_activation=true
```
모델을 정확도, confusion matrix, CAM 3가지 방법으로 분석한다. 각 방법을 선택 해제하거나 top-n 정확도 등 각 방법의 세부적인 parameter 또한 설정할 수 있다. 
