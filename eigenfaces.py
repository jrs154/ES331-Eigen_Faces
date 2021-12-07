import numpy as np
import cv2 as cv
from PIL import Image
from numpy import linalg as LA
import matplotlib.pyplot as plt
import math

image = []

#Function to train faces from database
def train_data():
    
    sum_images = np.zeros((250*250,)) # All image dimensions must be 250x250 pixels

    #Viola Jones algorithm detected 151 faces out of 165 images with
    #a minimum of 9 images per person. 6 images out of them are used
    #to train the eigenfaces algorithm, and three will be passed as test faces 
    for i in range(1,16):
        add = "useful/subject" + str(i) + "_"
        image.append((cv.imread(add+"1.jpg", 0)).flatten())
        image.append((cv.imread(add+"2.jpg", 0)).flatten())
        #image.append((cv.imread(add+"3.jpg", 0)).flatten())
        #image.append((cv.imread(add+"4.jpg", 0)).flatten())
        #image.append((cv.imread(add+"5.jpg", 0)).flatten())
        image.append((cv.imread(add+"6.jpg", 0)).flatten())
        image.append((cv.imread(add+"7.jpg", 0)).flatten())
        image.append((cv.imread(add+"8.jpg", 0)).flatten())
        image.append((cv.imread(add+"9.jpg", 0)).flatten())
    for img in image:
        sum_images = sum_images + img
    return image, sum_images


#Function to test new faces
def test_face(mean_img,img,K,image):
    test_img = img.flatten()            #Flattening the test image
    diff_test = test_img - mean_img 
    wgt = []                            #Initialising weight matrix
    for i in range(K):
	    wgt.append(np.dot(np.transpose(efacesn[i]), diff_test)) 

    mag_dist = []                       #Eigen distance of test image from all trained images
    for i in range(len(image)):
	    mag_dist.append(LA.norm(wgt - weight[i]))
    #print(max(mag_dist))
    #print(min(mag_dist))
    temp2 = np.resize(diff_test, (250,250))
    
    #plt.subplot(133)       #Uncomment to display the difference between test image and mean image
    #plt.imshow(temp2, cmap = 'gray')
    
    if min(mag_dist) > 20000:
        return -1                       #-1 means face not found in database (distance from images is too high)
    else:
        return mag_dist.index(min(mag_dist))//(M//15) + 1    #Subject number based on number of training samples


image, sum_images = train_data()
train_img = np.array(image)     #Array of flattened images
M = len(train_img)              #Number of training samples
N = len(train_img[0])
mean_img = sum_images/M         #Mean image matrix


A=[]                            #The difference matrix 
for i in range(M):
	diff_img = train_img[i] - mean_img
	A.append(diff_img.tolist())
A = np.array(A)

temp1 = np.resize(mean_img, (250,250))

#plt.subplot(132)       
#plt.imshow(temp1, cmap = 'gray')           #Uncomment to display the mean image

M_cov = np.dot(A, np.transpose(A))          #Small covariance matrix of MxM dimension
evl, evc = LA.eig(M_cov)                    #evl = eigenvalues and evc = eigenvectors of covariance matrix
srt = evl.argsort()[::-1]
evl_sort = evl[srt]
evc_sort = evc[srt]                         #Sorted eigenvectors in order of decreasing eigenvalues


efaces = np.dot(np.transpose(A),evc_sort)   #Eigenvector matrix of covarince matrix
efaces = np.transpose(efaces)               #Creating eigenfaces
efacesn = np.zeros((M,62500))
for i in range(M):                          #Normalizing the eigenfaces
    efacesn[i] = efaces[i]/np.linalg.norm(efaces[i], ord=2, axis=0, keepdims=True)


weight = []                 
K = 30                      #Random number of top features chosen
for i in range(M):
    wgt =[]
    for j in range(K):
	    wgt.append(np.dot(np.transpose(efacesn[j]) ,A[i]))
    weight.append(wgt)
weight = np.array(weight)   #Matrix containing weights of the features



#main code for subjects existing in database:
accuracy = 0 
observed = []
correct = []
for i in range(1,16):
    add = "useful/subject" + str(i) + "_"
    for j in range(1,10):
        if j==5: #or j==4 or j==5:
            test_img = cv.imread(add+str(j)+".jpg", 0)
            observed.append(test_face(mean_img,test_img,K,train_img))
            correct.append(i)
#plt.subplot(131)
#plt.imshow(test_img, cmap = 'gray')    #Uncomment to show the test image

for i in range(len(observed)):
    if correct[i]==observed[i]:
        accuracy += 1

print("\nFor images existing in database:")
print("\nNumber of test cases = " + str(len(observed)))
print("\nSubject labels of images: ")
print(correct)
print("\nThe face labels given by the system: ")
print(observed)
print("\nNumber of incorrect labels = " + str(len(observed)-accuracy))
print("Accuracy of Eigenface algorithm for positive samples = " + str(accuracy*100/len(observed)) + " %")
plt.show()


'''
#main code for subjects outside the database:
accuracy = 0
labels = []
observed = []
img = []
for i in range(1,8):
    for j in range(1,4):
        img.append(cv.imread("new_faces/s" + str(i) + "_" + str(j) + ".jpg",0))
        labels.append("s" + str(i))

print("\n\n\n\nFor images outside database:")
for i in range(len(img)):
    ac = test_face(mean_img,img[i],K,train_img)
    if ac == -1:
        observed.append("Face not found")
        accuracy += 1
    else:
        observed.append("Face matched to subject" + str(ac) + "(incorrect)")
print("\nNumber of test cases = " + str(len(observed)))
print("\nSubject labels of images: ")
print(labels)
print("\nOutput of the system: ")
print(observed)
print("\nNumber of false outputs = " + str(len(observed)-accuracy))
print("Accuracy of Eigenface algorithm for negative samples = " + str(accuracy*100/len(img)) + " %")
'''
