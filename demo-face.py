import numpy as np

import cv2 as cv

frame = cv.imread('temp/frame.jpg')  # [431, 342, 649, 625]

img4 = cv.imread('temp/2.jpg')
img5 = cv.imread('temp/face1.jpg')

frame[342:625, 431:649, :] = img4

cv.imwrite('temp/frame-4.jpg', frame)  # 68, 169,  57

for i in range(0, img5.shape[0]):

    for j in range(0, img5.shape[1]):
        #
        if abs(img5[i][j][0] - 68) <= 5 and abs(img5[i][j][1] - 169) <= 5 and (img5[i][j][2] - 57) <= 5:
            #
            img4[i][j][0] = 68
            img4[i][j][1] = 169
            img4[i][j][2] = 57

frame[342:625, 431:649, :] = img4

cv.imwrite('temp/frame-5.jpg', frame)  # 68, 169,  57

img1 = cv.imread('temp/demo-001.jpg')

frame[342:625, 431:649, :] = img1

cv.imwrite('temp/frame-1.jpg', frame)

# 缩放图像，后面的其他程序都是在这一行上改动
dst = cv.resize(img1, (96, 96))

img2 = cv.resize(dst, (img1.shape[1], img1.shape[0]))

frame[342:625, 431:649, :] = img2

cv.imwrite('temp/frame-2.jpg', frame)

img3 = cv.imread('temp/1.jpg')

img3 = cv.resize(img2, (img1.shape[1], img1.shape[0]))

frame[342:625, 431:649, :] = img3

cv.imwrite('temp/frame-3.jpg', frame)

# for i in range(0, img1.shape[0]):
#
#     for j in range(0, img1.shape[1]):
#         #
#         print('{}\t{}\t{}'.format(
#             img1[i][j][0] - img2[i][j][0],
#             img1[i][j][1] - img2[i][j][1],
#             img1[i][j][2] - img2[i][j][2]
#         ))


# 显示图像
cv.imshow("dst: %d x %d" % (dst.shape[0], dst.shape[1]), dst)

cv.waitKey(0)

cv.destroyAllWindows()
