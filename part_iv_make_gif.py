# Make an animated gif from the /images/1982_06/ folder
import imageio
import os
import glob


if __name__ == '__main__':
    # image_path = '/Users/jeewantha/Graduate Studies/Pinsky Lab/ROMS code/roms/images/images_i'
    image_path = 'images/images_i/'
    if os.path.exists(image_path):
        print('Yup')
        # glob the surface images
        surface_image_list = glob.glob(image_path + '/*surface_0*.png')
        print(surface_image_list)
        surface_image_list = sorted(surface_image_list)
        surface_image_io = [imageio.imread(x) for x in surface_image_list]
        print(surface_image_io)
        imageio.mimsave('surface_gif_2.gif', surface_image_io)
