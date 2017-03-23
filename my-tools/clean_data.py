#coding:utf8
# tools
# author: keith
# num eng: 2225 num chinese: 97018

import requests
import os
from matplotlib.image import imread
import numpy as np
import codecs
import re


def read_from_txt(file_name):
    """
    read data form txt
    :param file_name: file name
    :return: txt, string
    """
    with codecs.open(file_name, mode='r', encoding='utf8') as file:
        for line in file:
            # get rid of the url from txt
            line = re.sub(r"^\s*[a-zA-z]+://[^\s]*\s*", "", line)
            line = re.sub(r"\s{2,}", "", line)
            #print(line)  #test the result
            return line # string


def img_names2txt_names(image_names):
    """
    :param root: the root directory of txt files
    :param image_names:  the name of bad images , list
    :return: list of txt names
    """
    txt_names = []
    for image_name in image_names:
        txt_name = re.sub(r"\..*", ".txt", image_name)
        txt_names.append(txt_name)
    return txt_names


def img_name2txt_name(img_name):
    """
    generate the corresponding txt name from img name
    :param img_name: img name
    :return: txt name
    """
    txt_name = re.sub(r"\..*", ".txt", img_name)
    return txt_name


def delete_bad_txt_and_img(root_txt, root_img, bad_image_names):
    """
    delete bad txt and img
    :param root_txt: the root dir of txt
    :param root_img:  the root dir of img
    :param bad_image_names: the names of bad image names
    :return: Nothing
    """
    root_txt = re.sub(r"/$", "", root_txt)
    root_img = re.sub(r"/$", "", root_img)

    bad_txt_names = img_names2txt_names(bad_image_names)
    for txt, img in zip(bad_txt_names, bad_image_names):
        txt_path = root_txt+"/"+txt
        img_path = root_img+"/"+img
        if os.path.exists(txt_path):
            os.remove(txt_path)
        if os.path.exists(img_path):
            os.remove(img_path)


def get_url_from_txt(root, txt_name):
    """
    get url from txt
    :param root: the root of txt
    :param txt_name: the name of txt file
    :return: url
    """
    root = re.sub(r"/$", "", root)
    with codecs.open(root+"/"+txt_name, encoding="utf8") as file:
        for line in file:
            try:
                url = re.findall(r"^\s*[a-zA-z]+://[^\s]*", line)[0]
            except BaseException as e:
                raise "error"
            return url


def get_img_url_pairs(root, txt_names, bad_image_names):
    """
    get pairs (image_name:url)
    :param root:  the root dir of txt
    :param txt_names: bad names
    :param bad_image_names: bad names
    :return:  dict (image_names, url) no root
    """
    root = re.sub(r"/$", "", root)
    image_url_pairs = {}
    for img_name, txt_name in zip(bad_image_names, txt_names):
        image_url_pairs[img_name] = get_url_from_txt(root, txt_name)
    return image_url_pairs


def download_image(root, pairs):
    """
    download image
    :param root: image root
    :param pairs: (image: url) pairs
    :return: Nothing
    """
    print("downloading images")
    root = re.sub(r"/$", "", root)
    for i, key in enumerate(pairs.keys()):
        if i%10 == 0: # trace the process
            print(i)

        url = pairs[key]
        print(url)
        try:
            req = requests.get(url)
            content = req.content
            ##############
            if not os.path.exists(root+"_temp"):
                os.mkdir(root+"_temp")
            ##############

            relative_path = root+"_temp"+"/"+ key
            # remove the bad images
            if os.path.exists(relative_path):
                os.remove(relative_path)
            # write the good images
            with open(relative_path, mode='wb') as file:
                file.write(content)
        except BaseException as e:
            raise "url error"


def replace_the_bad_images(root_txt, root_img, bad_image_names):
    """
    replace the bad img from the good img
    :param root_txt:
    :param root_img:
    :param bad_image_names:
    :return:
    """
    txt_names = img_names2txt_names(bad_image_names)
    pairs = get_img_url_pairs(root_txt, txt_names, bad_image_names)
    download_image(root_img, pairs)


def get_news_title(txt_string):
    """
    doesn't work
    :param txt_string:
    :return:
    """
    is_eng = re.match(r"[a-zA-Z]", txt_string)

    if is_eng is not None:
        print("is eng")
        matched = re.findall(r"^[^\.]*\.", txt_string)
    else:
        print("is chi")
        # doesn't work
        matched = re.findall(r"^[^\s]+\s?\s{2}", txt_string)
    print(matched)
    return matched[0]


def check_data_set(root):
    """
    checking if there are some bad imgs
    :param root: the root dir of img
    :return: list of names of bad imgs
    """
    files = os.listdir(root)
    files = sorted(files)
    bad_data = []
    for file in files:
        try:
            img = imread(root+"/"+file)
        except BaseException as e:
            bad_data.append(file)
            return bad_data
    return bad_data


def clean_data(root_txt, root_img):
    """
    remove the bad (img, txt) from the raw data
    :param root_txt: the root dir of txt
    :param root_img: the root dir of img
    :return: Nothing
    """
    root_txt = re.sub(r"/$", "", root_txt)
    root_img = re.sub(r"/$", "", root_img)

    files = os.listdir(root_img)
    files = sorted(files)
    for file in files:
        try:
            img = imread(root_img + "/" + file)
        except BaseException as e:
            relative_path_img = root_img + "/" + file
            txt_name = img_name2txt_name(file)
            relative_path_txt = root_txt + "/" + txt_name
            if os.path.exists(relative_path_img):
                print("delete img ", relative_path_img)
                os.remove(relative_path_img)
            if os.path.exists(relative_path_txt):
                print("delete txt ", relative_path_txt)
                os.remove(relative_path_txt)


def count_eng_chinese(root_txt):
    """
    count the quantity of english file and chinese file separately
    :param root_txt: the root of txt files
    :return: None
    """
    root = re.sub(r"/$", "", root_txt)
    file_names = os.listdir(root_txt)
    num_eng = 0
    num_chinese = 0
    for file_name in file_names:
        line = read_from_txt(root_txt+"/"+file_name)
        if is_eng(line):
            num_eng += 1
        else:
            num_chinese += 1
    print("num eng:", num_eng, "num chinese:", num_chinese)


def is_eng(txt):
    """
    is the txt english or not?
    :param txt: string
    :return: boolean
    """
    is_en = re.match(r"[a-zA-Z]", txt)
    if is_en is None:
        return False
    else:
        return True


def max_length_of_txts_word_wised(root_txt):
    """
    print the max length of txt of english and chinese separately
    :param root_txt: the root dir of txt
    :return: None
    """
    def eng_txt_length(txt):
        matched = re.findall(r"\S+\s+", txt)
        return len(matched)

    def chinese_txt_length(txt):
        #txt_ = re.sub(r"(\s+|。|，|“|”|【|】|！|？|（|）)","", txt)
        line_ = re.sub(r"\s*[a-zA-z]+://[^\s]*\s*", "", txt)
        txt_ = re.sub(r"(\s+|。|，|“|”|【|】|！|？|（|）|\d|\.|-|\(|\))", "", line_)
        return len(txt_)
    root_txt = re.sub(r"/$", "", root_txt)
    file_names = os.listdir(root_txt)
    max_length_eng = 0
    max_chi_name = None
    max_length_chi = 0
    for file_name in file_names:
        txt = read_from_txt(root_txt+"/"+file_name)
        if is_eng(txt):
            length = eng_txt_length(txt)
            if length > max_length_eng:
                max_length_eng = length
        else:
            length = chinese_txt_length(txt)
            if length > max_length_chi:
                max_chi_name = file_name
                max_length_chi = length
    print("max eng:", max_length_eng, "max chi:", max_length_chi)
    print("max name", max_chi_name)

def max_length_of_txts_letter_wised(file_names):
    pass


def test():
    txt = "hello what is your name"
    txt = re.findall(r"(\S+)\s",txt)
    print(txt)


def main():
    root_img = "../formalCompetition4/News_pic_info_train"
    root_txt = "../formalCompetition4/News_info_train"
    #test()
    #max_length_of_txts_word_wised(root_txt)
    #count_eng_chinese(root_txt)
    # line = read_from_txt(root_txt+"/"+"2016690798.txt")
    # line_ = re.sub(r"\s*[a-zA-z]+://[^\s]*\s*", "", line)
    # txt_ = re.sub(r"(\s+|。|，|“|”|【|】|！|？|（|）|\d|\.|-|\(|\))", "", line_)
    # print(txt_[0])
    # print(len(txt_))
    #test()
    txt = "你好"
    matched = re.findall(r"你",txt)
    print(matched)

if __name__ == '__main__':
    main()
