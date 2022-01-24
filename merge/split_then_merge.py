import imageio
import numpy as np
import os

#
# @brief Partitioning input to an image array with 256x256 pixels, padding size ( (64, 64), (64, 64) )
# @param image_gray 2D Gray image, a numpy array.
# @return partitioned image with shape [r, c, 256, 256]
#
def partition_gray( image_gray ):
    row, col = image_gray.shape

    nrow = (row // 256) * 256 + 512
    rl_padding = (nrow - row) // 2
    rr_padding = nrow - row - rl_padding

    ncol = (col // 256) * 256 + 512
    cl_padding = (ncol - col) // 2
    cr_padding = ncol - col - cl_padding

    nimg = np.pad( image_gray, [(rl_padding, rr_padding), (cl_padding, cr_padding)], mode='reflect' )

    images_gray = []
    rows, cols = (nrow//128) - 1, (ncol//128) - 1
    images_gray = np.zeros( (rows, cols, 256, 256) )
    for r in range( rows ):
        for c in range( cols ):
            images_gray[r][c] = nimg[r*128:r*128+256, c*128:c*128+256]

    return images_gray


#
# @brief Merge
# @param images_rgb The colorized rgb image, 3 channels. shape [r, c, 256, 256, 3], padding size ( (64, 64), (64, 64) ).
# @param row The row resolution to the gray image
# @param col The column resolution to the gray image
# @return Merged image, shape [row, col, 3]
#
def merge( images_rgb, row, col ):
    rows, cols, *_ = images_rgb.shape
    tmp_img = np.zeros( (rows*128, cols*128, 3) )
    for r in range( rows ):
        for c in range( cols ):
            tmp_img[r*128:r*128+128, c*128:c*128+128] = images_rgb[r, c, 64:192, 64:192, :]
    pr = (rows*128 - row) // 2
    pc = (cols*128 - col) // 2
    ans = tmp_img[pr:row+pr, pc:col+pc]
    return ans


#
# @brief partition-colorization-then-merge
# @param image_gray A gray image, a numpy array
# @param gray_output_dir The input dir for the model to read
# @param rgb_input_dir The output dir for the model to produce
# @return A RGB image, a numpy array
#
def do_colorization( image_gray, gray_output_dir='./img_dir', rgb_input_dir='./store_dir/final' ):
    row, col = image_gray.shape
    images_gray = partition_gray( image_gray )

    # serialize gray images
    r, c, *_ = images_gray.shape
    _ig = images_gray.reshape( (r*c, 256, 256) )
    for idx in range( r*c ):
        imageio.imwrite( f'{gray_output_dir}/{str(idx).zfill(8)}.png', _ig[idx] )

    # gray -> rgb
    # command python3 ./custom_colorize.py --config=configs/colorizer.py --logdir=./coltran/colorizer --img_dir=./img_dir --store_dir=./store_dir --mode=colorize && python3 ./custom_colorize.py --config=./configs/color_upsampler.py --logdir=./coltran/color_upsampler --img_dir=./img_dir --store_dir=./store_dir --gen_data_dir=./store_dir/stage1 --mode=colorize && python ./custom_colorize.py --config=./configs/spatial_upsampler.py --logdir=./coltran/spatial_upsampler --img_dir=./img_dir --store_dir=./store_dir --gen_data_dir=./store_dir/stage2 --mode=colorize
    command = f'python3 ./custom_colorize.py --config=configs/colorizer.py --logdir=./coltran/colorizer --img_dir=./img_dir --store_dir=./store_dir --mode=colorize && python3 ./custom_colorize.py --config=./configs/color_upsampler.py --logdir=./coltran/color_upsampler --img_dir=./img_dir --store_dir=./store_dir --gen_data_dir=./store_dir/stage1 --mode=colorize && python ./custom_colorize.py --config=./configs/spatial_upsampler.py --logdir=./coltran/spatial_upsampler --img_dir=./img_dir --store_dir=./store_dir --gen_data_dir=./store_dir/stage2 --mode=colorize'
    os.system( command )

    # deserialize rgb image
    rgb_images = np.zeros( (r*c, 256, 256, 3 ) )

    for idx in range( r*c ):
        rgb_images[idx] = imageio.imread( f'{rgb_input_dir}/{str(idx).zfill(8)}.png' )

    return merge( np.reshape( rgb_images, (r, c, 256, 256, 3) ), row, col )


def colorize( input_gray_image_path, output_rgb_image_path ):
    imageio.imwrite( output_rgb_image_path, do_colorization( imageio.imread( input_gray_image_path ) )


if __name__ == '__main__':
    colorize(  '/home/feng/Downloads/Session4-Job 010- Setup 006-face_1.png',  '/home/feng/Downloads/Session4-Job 010- Setup 006-face_1_rgb.png' )

