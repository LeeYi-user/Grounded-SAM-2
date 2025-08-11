import glob

# 存放圖片的路徑
train_image_path = r"D:/Git/Grounded-SAM-2/customdata_830/images/train/"
valid_image_path = r"D:/Git/Grounded-SAM-2/customdata_830/images/valid/"
# 生成的txt的路徑
txt_path = r"D:/Git/Grounded-SAM-2/customdata_830/"

def generate_train_and_val(image_path, txt_file):
    with open(txt_file, 'w') as tf:  # 修正引號
        for jpg_file in glob.glob(image_path + '*.jpg'):  # 修正引號
            tf.write(jpg_file + '\n')  # 修正引號

generate_train_and_val(train_image_path, txt_path + 'train.txt')  # 修正引號
generate_train_and_val(valid_image_path, txt_path + 'valid.txt')  # 修正引號
