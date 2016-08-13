#Logistic Regression Classification
##Linear Regression
If h<sub>&#952;</sub>(x) &ge; 0.5, predict y = 1<br>
If h<sub>&#952;</sub>(x) &lt; 0.5, predict y = 0<br>
Problem: There is no bound for h<sub>&#952;</sub>(x), h<sub>&#952;</sub>(x) can be > 1 or < 0.<br>

##Logistic:
We want 0 &le; h<sub>&#952;</sub>(x) &le; 1, h<sub>&#952;</sub>(x) = g(Z) = g(&#952;<sup>T</sup>x) = (1+e<sup>-&#952;<sup>T</sup>x</sup>)<sup>-1</sup><br>
h<sub>&#952;</sub>(x) = P(y = 1 | x ; &#952;), probability that y = 1, given x, parameterized by &#952;.<br>
1 - h<sub>&#952;</sub>(x) = P(y = 0 | x ; &#952;), estimated probability that y = 0 on input x.<br>
###Decision Boundry
h<sub>&#952;</sub>(x) = g(&#952;<sub>0</sub> + &#952;<sub>1</sub>x<sub>1</sub> + &#952;<sub>2</sub>x<sub>2</sub>) ,eg.: &#952; = [-3, 1, 1]<br>
predict y = 1, if -3 + x<sub>1</sub> + x<sub>2</sub> &ge; 0 --> x<sub>1</sub> + x<sub>2</sub> &ge; 3<br>

This module implements assignments of machine learning coursera Stanford by PYTHON.

If you want to use this module, please read the *.py file in test folder. You should get a good understanding of using this module by reading test file. If not, please contact me. Any judgement and recommendation are welcomed.
Thank you.
