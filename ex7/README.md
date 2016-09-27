#K-means Clustering and Principal Component Analysis<br>
In **machineLearningStanford/ex7**, executing following command.<br>
```
  python3 -m text.ex7
```
##Using K-means to find centroids<br>
![preview](https://cloud.githubusercontent.com/assets/5163329/18864927/3fce40b0-84cb-11e6-9bd4-e86b71928658.png)
We can observe the progress of the algorithm how it converges to final centroids.<br>
##Image compression with K-means<br>
In a straightforward 24-bit color representation of an image, each pixel is represented as three 8-bit unsigned
integers (ranging from 0 to 255) that specify the red, green and blue intensity values. This encoding is often 
refered to as the **RGB** encoding. Our image contains thousands of colors, and in this exercise, you will reduce
the number of colors to **16** colors. By doing this, it is possible to represent (compress) the photo in an efficient
way. Specifically, you only need to store the RGB values of the 16 selected colors, and for each pixel in the image
you now need to only store the index of the color at that location (where only 4 bits are necessary to represent 16 
possibilities).<br>
**Original Bird**<br>
![preview](https://cloud.githubusercontent.com/assets/5163329/18864909/2be278dc-84cb-11e6-8a22-a467ec5bb414.png)<br>
**Compressed Bird**<br>
![preview](https://cloud.githubusercontent.com/assets/5163329/18864906/2bdca646-84cb-11e6-8a4b-7d1ea03b8e33.png)<br>
**Original Me**<br>
![preview](https://cloud.githubusercontent.com/assets/5163329/18864910/2be38e34-84cb-11e6-8b7b-4e5d68138664.png)<br>
**Compressed Me**<br>
![preview](https://cloud.githubusercontent.com/assets/5163329/18864911/2be5282a-84cb-11e6-8879-02525f1199e3.png)<br>

##Principal Component Analysis<br>
###Computed eigenvectors of the dataset<br>
![preview](https://cloud.githubusercontent.com/assets/5163329/18868576/50c64556-84db-11e6-8291-e76cdc85f053.png)<br>
###The normalized and projected data after PCA<br>
![preview](https://cloud.githubusercontent.com/assets/5163329/18868580/50cabde8-84db-11e6-8188-5340aaf91568.png)<br>
###Face dataset<br>
![preview](https://cloud.githubusercontent.com/assets/5163329/18868579/50ca0254-84db-11e6-9a64-f2f908f570ab.png)<br>
###Principal components on the face dataset<br>
![preview](https://cloud.githubusercontent.com/assets/5163329/18868582/50f72590-84db-11e6-827a-04851aed9883.png)<br>
###Original images of faces<br>
![preview](https://cloud.githubusercontent.com/assets/5163329/18868583/50f7ce6e-84db-11e6-932c-73f5dd76ed46.png)<br>
###Reconstructed images of faces<br>
![preview](https://cloud.githubusercontent.com/assets/5163329/18868584/50f8a942-84db-11e6-9918-1d30f8b984bf.png)<br>
From the reconstruction, you can observe that the general structure and appearance of the face are kept while the fine
details are lost. This is a remarkable reduction (more than 10×) in the dataset size that can help speed up your learning
algorithm significantly. For example, if you were training a neural network to perform person recognition (gven a face
image, predict the identitfy of the person), you can use the dimension reduced input of only a 100 dimensions instead of
the original pixels.<br>
