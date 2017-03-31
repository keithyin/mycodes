#coding:utf8
#author: keithyin

import pandas as pd

action_file_name_02 = "JData_Action_201602.csv"
action_file_name_03 = "JData_Action_201603.csv"
action_file_name_03_e = "JData_Action_201603.csv"
action_file_name_04 = "JData_Action_201604.csv"

product_file_name = 'JData_Product.csv'
user_info_file_name = 'JData_User.csv'
comment_file_name = 'JData_Comment(修正版).csv'
def process_user_info(data_frame):
    """
    process user info
    :param data_frame:  DataFrame
    :return: processed user info
    about age:
    -1: -1    there is reason for them to hide their age. so i am going to preserve this information.
    6~15: 0
    16~25: 1
    26~35: 2
    36~45: 3
    46~55: 4
    """
    print(type(pd.to_datetime(data_frame["user_reg_dt"])))
    st = "hello"

    data_frame["user_reg_dt"] = pd.to_datetime(data_frame["user_reg_dt"])
    age_list = []
    for age in data_frame["age"]:
        if age.startswith("1"):
            age_list.append(1)
        elif age.startswith("2"):
            age_list.append(2)
        elif age.startswith("3"):
            age_list.append(3)
        elif age.startswith("4"):
            age_list.append(4)
        elif age.startswith("5"):
            age_list.append(5)
        elif age.startswith("-1"):
            age_list.append(-1)
    age_series = pd.Series(age_list)
    data_frame["age"] = age_series
    return data_frame

def process_action_info(data_frame):

    print(data_frame)


def main():
    user_file_name = 'JData_User.csv'
    # print(data_files)
    #user_data = pd.read_csv(user_file_name, encoding="GBK")
    #process_user_info(user_data)
    action_data = pd.read_csv(action_file_name_02, encoding="GBK")
    process_action_info(action_data)

if __name__ == '__main__':
    main()