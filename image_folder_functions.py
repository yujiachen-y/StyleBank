import os
import shutil
import numpy as np
from PIL import Image


def is_photo(name):
    return name[0] != '.' and name.split('.')[-1] in ['jpg', 'png']


def rename_file_str(file, new_name):
    file = file.split('.')
    return new_name + '.' + file[-1]


zfill_pad = 6


def handle_folder(dir):
    length, data = 0, []
    files = os.listdir(dir)
    for file in files:
        if is_photo(file):
            # length = length + 1
            file_name, file_suffix = file.split('.')
            file_name = file_name.zfill(zfill_pad)
            old_file = os.path.join(dir, file)
            new_file = os.path.join(dir, file_name+'.'+file_suffix)
            os.renames(old_file, new_file)
            data.append(file_name+'.'+file_suffix)
    data.sort()
    for i, file in enumerate(data):
        new_file = rename_file_str(file, str(i).zfill(zfill_pad))
        data[i] = new_file
        old_file = os.path.join(dir, file)
        new_file = os.path.join(dir, new_file)
        os.renames(old_file, new_file)
    # return length, data


if __name__ == '__main__':
    os.chdir('C:\\Users\\yjc56\\Desktop\\photosketch_clear')
    dataset_list = ['AR', 'CUHK', 'CUHK_FERET', 'XM2VTS',
                    'VIPSL0', 'VIPSL1', 'VIPSL2', 'VIPSL3', 'VIPSL4']
    for dir in dataset_list:
        handle_folder(os.path.join(dir, 'photos'))
        handle_folder(os.path.join(dir, 'sketches'))

# def divide_VIPSLdataset():
#     dir = os.path.join('datasets', 'VIPSL_FaceSketch')
#     if not os.path.exists(dir):
#         raise ValueError('path: ' + dir + ' does not exits!')
#     for i in range(5):
#         new_dir = dir+str(i)
#         os.makedirs(new_dir)
#
#         new_photo_dir = os.path.join(new_dir, 'photos')
#         old_photo_dir = os.path.join(dir, 'photos')
#         shutil.copytree(old_photo_dir, new_photo_dir)
#
#         new_sketch_dir = os.path.join(new_dir, 'sketches')
#         old_sketch_dir = os.path.join(dir, 'sketches' + str(i))
#         shutil.copytree(old_sketch_dir, new_sketch_dir)
#
#
# def count_style():
#     # print 'count_style start'
#     main_dir = os.path.join('.', 'datasets')
#     dir_list = os.listdir(main_dir)
#
#     # print dir_list
#     style_info = {}
#
#     for folder in dir_list:
#         dir = os.path.join(main_dir, folder)
#         # print 'deal ' + dir + '...'
#         if os.path.isdir(dir):
#             lengths = {}
#             result = {}
#
#             for sub_folder in ['photos', 'sketches']:
#                 sub_dir = os.path.join(dir, sub_folder)
#                 length, data = handle_folder(sub_dir)
#                 result[sub_folder] = data
#                 lengths[sub_folder] = length
#
#             if lengths['photos'] != lengths['sketches']:
#                 raise ValueError(folder + ' size does not match !')
#             for (pname, sname) in zip(result['photos'], result['sketches']):
#                 if pname.split('.')[0] != sname.split('.')[0]:
#                     raise ValueError(folder + ' name does not match !')
#
#             result['length'] = lengths['photos']
#             style_info[folder] = result
#
#     return style_info


# def divide_style_info(style_info, style_list, ratio=0.8):
#     train_info,  test_info = {}, {}
#     for style in style_list:
#         train, test = {}, {}
#         train['length'], test['length'] = 0, 0
#         train['photos'], train['sketches'] = [], []
#         test['photos'], test['sketches'] = [], []
#
#         info = style_info[style]
#         indexes = range(info['length'])
#         np.random.shuffle(indexes)
#         bound = int(np.floor(info['length']*ratio + 0.5))
#
#         for i in indexes:
#             if train['length'] < bound:
#                 train['photos'].append(info['photos'][i])
#                 train['sketches'].append(info['sketches'][i])
#                 train['length'] = train['length'] + 1
#             else :
#                 test['photos'].append(info['photos'][i])
#                 test['sketches'].append(info['sketches'][i])
#                 test['length'] = test['length'] + 1
#
#         train_info[style], test_info[style] = train, test
#
#     return train_info, test_info
