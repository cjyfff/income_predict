## 收入预测程序   
   
   
#### 概述   
基于UCI的这一份[美国成年人收入数据](http://archive.ics.uci.edu/ml/datasets/Adult)，使用机器学习算法对美国成年人的收入进行预测。目前使用的分类模型是Logistic Regression和SVM。   
   
   
#### 依赖   
python==2.7   
numpy==1.10.1   
matplotlib==1.5.1   
pandas==0.17.1   
   
   
#### 使用方法   
在代码主目录执行python main.py即可   

#### 代码逻辑说明   
1. LR的优化方法采用随机梯度下降    
2. SVM使用RBF作为核函数   
   

#### 总结   
1. LR的预测准确率并不稳定，在74%～79%之间，SVM的预测准确率比较稳定，为84%左右。在对age、education_num、hours_per_week等连续数据进行离散化后，LR的预测准确率提升至84%，SVM的准确率基本没变化。   
2. SVM的建立模型时所需要的参数需要使用交叉验证得到。该过程耗时很长，对32000条数据进行交叉验证，在单进程的情况下，在我的机器上耗费了超过20小时。    
3. 两种算法建模并且预测的时间相差也比较大，在我的机器上，LR训练模型以及预测共耗时42秒，SVM耗时337秒。   
   
