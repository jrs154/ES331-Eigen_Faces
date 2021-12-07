import cv2 as cv

image = []
count = 2
dim = (250, 250)

image.append(cv.imread("1.pgm", 0))
image.append(cv.imread("2.pgm", 0))
image.append(cv.imread("3.pgm", 0))

for i in range(0,3):
    filename = 's' + str(count) + '_' + str(i+1) + '.jpg'
    img = cv.resize(image[i], dim, interpolation = cv.INTER_AREA)
    cv.imwrite(filename, img)
