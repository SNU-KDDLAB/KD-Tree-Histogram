-English-

Python3 implementation of *An Improvement of KD-tree base Metric Similarity Joins using MapReduce*, KSC2018 (Korea Software Congress 2018)

(There was a typo in the english title; "base"->"based")


Ubuntu version : Ubuntu 14.04.5 LTS (GNU/Linux 4.4.0-81-generic x86_64)

Hadoop version : Hadoop 2.6.0-cdh5.11.2


Usage :


1.In partition_s.py, edit input file names to your string data in lines 28 and 31.
```
28  with open('datasets/word/Q_11', 'rb') as f:
29      Q = pickle.load(f)
30
31  with open('datasets/word/O_11', 'rb') as f:
32      O = pickle.load(f)
```

2.Run test_s.sh


Word data used in my experiments is from http://dbgroup.cs.tsinghua.edu.cn/ligl/simjoin/


-한국어-

2018 한국소프트웨어종합학술대회 (KSC2018, Korea Software Congress 2018) 에 제출한 *맵리듀스를 이용한 KD-트리 기반 거리 유사도 조인의 개선*을 파이썬3로 구현한 코드입니다.


우분투 버전 : Ubuntu 14.04.5 LTS (GNU/Linux 4.4.0-81-generic x86_64)

하둡 버전 : Hadoop 2.6.0-cdh5.11.2


실행 방법 : 

1. partition_s.py의 28, 31번 줄에서 파일 이름을 가지고 있는 문자열 데이터의 파일 이름으로 바꾼다.

```
28  with open('datasets/word/Q_11', 'rb') as f:
29      Q = pickle.load(f)
30
31  with open('datasets/word/O_11', 'rb') as f:
32      O = pickle.load(f)
```


2.test_s.sh 실행


실험에 사용한 단어 데이터의 출처 : http://dbgroup.cs.tsinghua.edu.cn/ligl/simjoin/
