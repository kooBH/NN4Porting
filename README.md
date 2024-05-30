# NN4Porting  
신경망 포팅을 위한 더미 라이브러리


## requirements
- pytorch
- librosa(test)

## model.py  

### Net

STFT 도메인에서 동작하는 masking 네트워크 

B : batch  
C : channels == 2(real, imag)  
F : frequency  
T : time  

입력 :   
+ x[B,C,F,T] : STFT 도메인 입력  
+ h[B,1,144] : RNN의 hidden state   

출력 :
+ y[B,C,F,T] : 입력 x에 모델 출력 z를 마스킹한 결과  
+ h1 : 업데이트된 hidden state  


### NetHelper    
STFT 연산을 추적하기 위해 래핑한 클래스  

B : batch    
L : samples  

입력 : 
+ x[B,L] : 단일 채널 웨이브 입력
+ state[B,1,144] : RNN의 hidden state   

출력 :
+ y[B,L] : 모델 출력을 적용한 결과  
+ h : 업데이트된 hidden state  

### 더미 학습   
입력 == 출력을 목표로 하여 학습한 결과를 ```dummy.pt```에 저장   
해당 모델의 ONNX 버전은 ```dummy.onnx```에 저장  
  
## test.py   
dummy 모델을 테스트하는 코드    
```dummy.pt```의 가중치를 모델에 불러와 사용   
```input.wav```를 입력으로 받아서 ```output.wav```로 저장    
