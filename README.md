# Data-Driven Ultra-Long-Term Early Prediction of Lithium-Ion Battery State of Health Under Different Charge-Discharge Strategies

This is a PyTorch implementation of LRPEformer model and discussion experiments proposed by our paper "Data-Driven Ultra-Long-Term Early Prediction of Lithium-Ion Battery State of Health Under Different Charge-Discharge Strategies".

## 1.Overview
![Example Image](imgs/fig1.png)
Fig.1 An overview of LRPEformer model.

Our study presents an Ultra-Long-Term Early Prediction model, LRPEformer (LSTM Relative Position Encoding Informer), to predict battery State of Health and significantly reduce the time cost of experiments. We perform unified sequence modeling for lithium-ion batteries under various charge-discharge strategies.We introduce complex temporal feature extraction and optimize the formula for calculating attention scores, effectively modeling ultra-long and complex temporal information. We also propose a two-stage training method, modeling and predicting dependent features and target features separately, reducing error accumulation during iterative prediction.


## 2.Requirements
- pyecharts==2.0.3
- matplotlib==3.5.1
- numpy==1.21.5
- pandas==1.3.5
- scikit-learn==1.0.2
- torch==1.7.1
- ....

You can install it directly from the environment pack version file:
```
pip install -r requirements.txt
or
conda env create -f torch_lts.yaml
```

## 3.Data
You can find the processed data for this work in the data folder, or obtain the original data from the following table using the reference URL.

<table>
  <tr>
    <th style="width:20%;">Data Source</th>
    <th style="width:50%;">Description</th>
    <th style="width:30%;">Reference URL</th>
  </tr>
  <tr>
    <td>Nature Battery(Closed-loop optimization of extreme fast charging for batteries using machine learning) </td>
    <td>The objective of this work is to optimize fast charging for lithium-ion batteries. It consists of commercial lithium-ion batteries cycled under fast-charging conditions.</td>
    <td><a href="https://data.matr.io/1">https://data.matr.io/1</a></td>
  </tr>
</table>

## 4.Training and Testing
You can perform univariate autoregressive prediction on dependent features and multivariate autoregressive prediction on the target feature (Qd) in the four files: 'natureMain(Target_min,Parameter_min).ipynb', 'natureMain(Target_Qd,Parameter_min_Qd).ipynb', 'SDMain(Target_Δη,Parameter_Δη).ipynb', and 'SDMain(Target_Qd,Parameter_Δη_Qd).ipynb'. Some experimental results are shown below, demonstrating that our proposed LRPEformer significantly outperforms Informer.



