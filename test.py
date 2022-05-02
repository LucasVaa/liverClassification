import os
import glob
import json

from PIL import Image
import numpy as np
import cv2

from model import GoogLeNet


def main():
    im_height = 224
    im_width = 224

    data_root = os.path.abspath(os.path.join(os.getcwd(), ""))  # get data root path
    image_path = os.path.join(data_root, "data")  # flower data set path
    test_dir = os.path.join(image_path, "test_set")


    liver_class = [cla for cla in os.listdir(test_dir)
                    if os.path.isdir(os.path.join(test_dir, cla))]

    print(liver_class)

    model = GoogLeNet(class_num=6, aux_logits=False)
    # model.load_weights("./save_weights/myGoogLenet.h5", by_name=True)  # h5 format
    weights_path = "./save_weights/myGoogLeNet50.ckpt"
    assert len(glob.glob(weights_path + "*")), "cannot find {}".format(weights_path)
    model.load_weights(weights_path)

    num = 0
    right = 0
    for cla in liver_class:
        cla_path = os.path.join(test_dir, cla)
        images = os.listdir(cla_path)
        pre = {
            "123": 0,
            "1234": 0,
            "4": 0,
            "5678": 0,
            "58": 0,
            "67": 0
        }
        for image in images:
            num = num + 1
            image_path_1 = os.path.join(cla_path, image)


            img = cv2.imread(image_path_1)  # 填要转换的图片存储地址
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(image_path, "convert", image), img)

            img_path = os.path.join(image_path, "convert", image)
            assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            img = Image.open(img_path)
            # resize image to 224x224
            img = img.resize((im_width, im_height))

            # scaling pixel value and normalize
            img = ((np.array(img) / 255.) - 0.5) / 0.5

            # Add the image to a batch where it's the only member.
            img = (np.expand_dims(img, 0))

            # read class_indict
            json_path = './class_indices.json'
            assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

            with open(json_path, "r") as f:
                class_indict = json.load(f)



            result = np.squeeze(model.predict(img))
            predict_class = np.argmax(result)

            pre[class_indict[str(predict_class)]] = pre[class_indict[str(predict_class)]] + 1
            if (class_indict[str(predict_class)] == cla):
                right = right + 1
        print(pre)
    print(right/num)

if __name__ == "__main__":
    main()

# 30 0.9439461883408071

# 40 0.95254110612855

# 50 0.9745889387144993