#Regularized Linear Regression and Bias v.s. Variance<br>
In **machineLearningStanford/ex5**, executing following command.<br>
```
  python3 -m text.ex5
```
##Linear Regression with λ = 0<br>
![preview](https://cloud.githubusercontent.com/assets/5163329/18840178/4adf2c5c-8440-11e6-8284-aa6f32e7e143.png)<br>
##Linear Regression Learning Curve<br>
![preview](https://cloud.githubusercontent.com/assets/5163329/18840181/4ae0538e-8440-11e6-808c-06b80dfa12fc.png)<br>
You can observe that both the train error and cross validation error are high when the number of training examples
is increased. This reflects a high bias problem in the model – the linear regression model is too simple and is unable
to fit our dataset well.<br>
##Polynomial Regression with λ = 3<br>
![preview](https://cloud.githubusercontent.com/assets/5163329/18840179/4adf6370-8440-11e6-8382-98fb38a2bdb5.png)<br>
##Polynomial Regression Learning Curve<br>
![preview](https://cloud.githubusercontent.com/assets/5163329/18840182/4ae314fc-8440-11e6-9dac-27c62c7a3e1a.png)<br>
You can observe that train and cross validation error converge to relatively low value. This shows that λ = 3 regularized
polynomial regression model does not have the high-bias or high-variance problems. In effect, it achieves a good trade-off
between bias and variance.<br>
##Train and Cross Validation Error v.s. λ<br>
![preview](https://cloud.githubusercontent.com/assets/5163329/18840180/4ae05668-8440-11e6-9cbe-7aef23290ee8.png)<br>
We want the lowest point of cross validation. In this figure, we can see that the best of λ is around 1. Due to randomness
in the training and validation splits of the dataset, the cross validation error can sometimes be lower than the the training
error.<br>
