import imageio
import cv2


#im = imageio.imread( '/home/feng/Downloads/Session4-Job 010- Setup 006-face_1.png' )
im = imageio.imread( '/home/feng/Downloads/test_1.png' )

start = 2
step = 1
n = 16

for idx in range( n ):
    grid = start + idx*step
    clahe = cv2.createCLAHE(clipLimit=6, tileGridSize=(grid, grid))
    im = clahe.apply((im))
    imageio.imwrite( f'/home/feng/Downloads/test_1_{grid}.png', im )


