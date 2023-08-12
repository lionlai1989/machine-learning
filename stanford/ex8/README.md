#Anomaly Detection and Recommender Systems<br>
In **machineLearningStanford/ex8**, executing following command.<br>
```
  python3 -m text.ex8
```
##Anomaly detection<br>
![preview](https://cloud.githubusercontent.com/assets/5163329/18869663/dae960f2-84df-11e6-8d41-15e77f7d77ee.png)<br>
First, you should draw the Gaussian distribution contours of the distribution fit to the dataset. Then, selecting the
threshold ε. Finally, you can see the anomalies are noted by red circles.<br> 
```
  python3 -m text.ex8_cofi
```
##Recommender systems<br>
There is a movie list below.<br>
1 Toy Story (1995)<br>
2 GoldenEye (1995)<br>
3 Four Rooms (1995)<br>
4 Get Shorty (1995)<br>
5 Copycat (1995)<br>
6 Shanghai Triad (Yao a yao yao dao waipo qiao) (1995)<br>
7 Twelve Monkeys (1995)<br>
8 Babe (1995)<br>
9 Dead Man Walking (1995)<br>
10 Richard III (1995)<br>
11 Seven (Se7en) (1995)<br>
.<br>
.<br>
.<br>
Then I randomly give my ratings below.<br>
myRatings[0] = 4<br>
myRatings[97] = 2<br>
myRatings[6] = 3<br>
myRatings[11] = 5<br>
myRatings[53] = 4<br>
myRatings[63] = 5<br>
myRatings[65] = 3<br>
myRatings[68] = 5<br>
myRatings[182] = 4<br>
myRatings[225] = 5<br>
myRatings[354] = 5<br>
Finally, my recommender system can recommend movies that I may like :)
![preview](https://cloud.githubusercontent.com/assets/5163329/18869660/d46104ce-84df-11e6-9307-f43f80ddca36.png)<br>
