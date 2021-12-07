import numpy as np
import cv2 as cv

face_cascade=cv.CascadeClassifier('cascade.xml')
image = []

def srtsec(val):  
    return val[2]


#Test for positive samples
for i in range(1,16):          #Reading the positive images from a folder
    if i<10:
        add = r"positive_images\subject0" + str(i)
    else:
        add = r"positive_images\subject" + str(i)
    image.append(cv.imread(add+".centerlight.jpg", 1))
    image.append(cv.imread(add+".rightlight.jpg", 1))
    image.append(cv.imread(add+".leftlight.jpg", 1))
    image.append(cv.imread(add+".happy.jpg", 1))
    image.append(cv.imread(add+".normal.jpg", 1))
    image.append(cv.imread(add+".sad.jpg", 1))
    image.append(cv.imread(add+".surprised.jpg", 1))
    image.append(cv.imread(add+".sleepy.jpg", 1))
    image.append(cv.imread(add+".glasses.jpg", 1))
    image.append(cv.imread(add+".noglasses.jpg", 1))
    image.append(cv.imread(add+".wink.jpg", 1))


count = 1
detected = []
index = []

for i in range(len(image)):      #Checking for faces in each image
    imgtest = image[i]
    faces = np.array(face_cascade.detectMultiScale(imgtest,1.001,7))
    faces = list(sorted(faces, key = srtsec, reverse = True))
    #print(faces)
    if faces:
        x,y,w,h = faces[0]
    else:
        x,y,w,h = 0,0,0,0
    if w >= 130:
        #detected.append(imgtest)
        index.append((i//11) + 1)
        imgtest = cv.rectangle(imgtest, (x,y), (x+w,y+h), (255,0,0), 2)
        #cropped = imgtest[y:y+h, x:x+w]
        #filename = 'subject' + str((i//11) + 1) + '_' + str(count) + '.jpg'
        #dim = (250,250)
        #cropped = cv.resize(cropped, dim, interpolation = cv.INTER_AREA)
        #cv.imwrite(filename, cropped)        

print("\nRunning the algorithm for images with faces: ")
print("Total number of test cases = " + str(len(image)))
print("Number of images in which the algorithm detected a face = " + str(len(index)))
print("\nThe subjects whose faces are detected: ")
print(index)
accuracy = len(index)*100/len(image)
print("\nAccuracy of Viola Jones algorithm for positive samples = " + str(accuracy) + " %")


#Test for negative samples
image = []
output = []
accuracy = 0
for i in range(0,11):          #Reading the negative images from a folder
    image.append(cv.imread("negative_images/neg-000" + str(i) + ".jpg", 1))

for i in range(len(image)):     #Checking for faces in each image
    imgtest = image[i]
    faces = np.array(face_cascade.detectMultiScale(imgtest,1.001,7))    #This gives the coordinates and dimensions of bounding box, if any
    faces = list(sorted(faces, key = srtsec, reverse = True))
    if faces:
        x,y,w,h = faces[0]
    else:
        x,y,w,h = 0,0,0,0
    
    if w >= 130:           #Face only considered if rectangle is appropriately big
        output.append("Face detected")
        imgtest = cv.rectangle(imgtest, (x,y), (x+w,y+h), (255,0,0), 2)
    else:
        output.append("No face")
        accuracy += 1
    
#cv.imshow('False', imgtest)

print("\n\n\n\nRunning the algorithm for images without faces: ")
print("Total number of test cases = " + str(len(image)))
print("Number of images in which the algorithm detected a face = " + str(len(image)-accuracy))
print("\nOutput of algorithm: ")
print(output)
accuracy = accuracy*100/len(image)
print("\nAccuracy of Viola Jones algorithm for negative samples = " + str(accuracy) + " %")

cv.waitKey(0)
cv.destroyAllWindows()

