import cv2


from seam import SeamCarve

img = cv2.imread('Example/dog.png')
height, width, channel = img.shape
new_img = cv2.resize(img, (400, 400))
masking=cv2.imread('Example/big.png')
new_mask = cv2.resize(masking, (400, 400))
cv2.imwrite('new_mask.png', new_mask)
mask = cv2.imread('new_mask.png', 0) != 255

## explicit converted to 400*400 size
# img = cv2.imread('Example/dog_explicit.png')
# mask = cv2.imread('Example/dog_mask_explicit.png', 0) != 255



sc_img = SeamCarve(new_img)
sc_img.remove_mask(mask)

new_img = cv2.resize(sc_img.image(), ( 450,height))
cv2.imshow('original', img)
cv2.imshow('removed', new_img)
cv2.waitKey(0)
