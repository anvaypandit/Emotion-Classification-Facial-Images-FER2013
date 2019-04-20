# Saves the image files on disk as required by the keras api
import data_loader
from keras.preprocessing.image import save_img
from keras.preprocessing.image import array_to_img
import os
import numpy as np
import pandas as pd

IMAGE_WIDTH = 48
IMAGE_HEIGHT = 48

def extract_and_save_img(df, type_subdir, index, file_prefix='Image_', image_dir='data'):
    type_fp = os.path.join(os.getcwd(), image_dir)
    type_fp = os.path.join(type_fp, type_subdir + str(index))
    if not os.path.exists(type_fp):
        os.mkdir(type_fp)

    # Iterate over data frame
    basetype_fp = type_fp
    for row in df.itertuples():
        pixel_arr = np.array(row.pixels.split())

        emotion = 0
        if row.emotion != index:
            emotion = 1
        
        if type_subdir != 'Test':
            type_fp = os.path.join(type_fp, str(emotion))
        else:
            type_fp = os.path.join(type_fp, 'Test_folder')
        if not os.path.exists(type_fp):
            os.mkdir(type_fp)

        # Reshape to channel_last_dimensions
        pixel_arr = np.reshape(pixel_arr, (IMAGE_WIDTH, IMAGE_HEIGHT, 1))

        if type_subdir != 'Test':
            type_fp = os.path.join(type_fp, file_prefix + str(row.Index))
        else:
            type_fp = os.path.join(type_fp, str(emotion) + '_' + file_prefix + str(row.Index))

        save_img(type_fp + '.png', pixel_arr)
        
        #Reset type_fp to basetype
        type_fp = basetype_fp


train_df,test_df,val_df= data_loader.load_all_data()

for i in range(7):
    extract_and_save_img(train_df, 'Train', i)
    extract_and_save_img(val_df, 'Valid', i)
    extract_and_save_img(test_df, 'Test', i)
