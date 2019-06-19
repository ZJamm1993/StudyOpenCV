from cv2 import cv2

img = cv2.imread('resource/inpaint_test.jpg')
mask = cv2.imread('resource/inpaint_test_mask.jpg', 0)

dst1 = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
dst2 = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()
