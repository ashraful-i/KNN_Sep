import sys
# import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from IPython.display import display
import json


def read_data(dataset: pd.DataFrame, sep, header):


    if not header:
        df = pd.read_csv(dataset, sep=sep, header=None)
    else:
        df = pd.read_csv(dataset, sep=sep)
    df = pd.DataFrame(df)
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    return df


def split_dataset(df, x):
    mask_test = int(df.shape[0] * (10 / 100))
    test_start = x * mask_test
    test_end = test_start + mask_test
    test_set = df[test_start:test_end]
    train_set = pd.concat([df[:test_start], df[test_end:]], ignore_index=True)
    return train_set, test_set


def euclidean_distance(row1, row2, col):
    distance = 0
    for c in col:
        row1_val = getattr(row1, c)
        row2_val = getattr(row2, c)
        distance += (row1_val - row2_val) ** 2
    return sqrt(distance)


def get_avg_point(tl, n):
    df_list = [[] for y in range(n)]
    avg_points = [0 for y in range(n)]
    # print (df_list)

    for x in range(n):
        _list = []
        for l in tl[x]:
            _list.append(l.to_dict())
        avg_points[x] = pd.DataFrame(_list).mean()

    # print(avg_points)
    return avg_points


def get_grp_point(dataset, n, test_points, diff):
    if dataset.empty:
        return
    avg_dis = 0
    global test_list
    global divided_grps
    cols = dataset.columns.values.tolist()
    cols.pop(0)
    test_list1 = [[] for y in range(n)]
    for cnt_df, i in dataset.iterrows():
        min_dist = sys.maxsize
        min_idx = -1
        for cnt_points, j in enumerate(test_points.iterrows()):
            dis = euclidean_distance(i, j[1], cols)
            if dis < min_dist:
                # min_dis_point = i.to_dict()
                min_dist = dis
                min_idx = cnt_points
        test_list1[min_idx].append(i)
    # print(test_list)
    divided_grps = test_list1

    avg_pts = get_avg_point(test_list1, n)
    _list = []
    for item in avg_pts:
        _list.append(item.to_dict())
    avg_pts_tt = pd.DataFrame(_list)
    total_dis = 0
    for x in range(n):
        for index, row1 in test_points.iloc[[0]].iterrows():
            for index2, row2 in avg_pts_tt.iloc[[x]].iterrows():
                dis = euclidean_distance(row1, row2, cols)
                total_dis += dis
                # print(dis)
    #print(total_dis)
    if abs(total_dis - diff) < 1:
        test_list = avg_pts_tt
        return
    get_grp_point(dataset, n, avg_pts_tt, total_dis)



def knn(df_knn: pd.DataFrame, group_cnt):
    global test_list
    # print(df_knn[:][group_cnt:])
    get_grp_point(df_knn[:][group_cnt:], group_cnt, df_knn[:][:group_cnt], 10000)
    optimum_points = test_list
    #print(optimum_points)
    return optimum_points


def check_test_set(df_knn_test: pd.DataFrame, points: pd.DataFrame, class_col, col_name):
    cols = df_knn_test.columns.values.tolist()
    cols = cols[1:]
    print(cols)
    print(points)
    point_grp = points.groupby(col_name)
    total_instance = 0
    correct = 0
    for cnt_df, i in df_knn_test.iterrows():
        distance_min = sys.maxsize
        result_grp = -1
        result_found = -1
        crnt_grp = i[col_name]
        for pname, pt_group in point_grp:
            for cnt_points, j in pt_group.iterrows():
                dis = euclidean_distance(i, j, cols)
                if dis < distance_min:
                    distance_min = dis
                    result_grp = pname
                    result_found = crnt_grp
        total_instance += 1
        if result_grp == result_found:
            correct += 1
    percentage = (correct / total_instance) * 100
    print("Accuracy " + str(percentage) + "%")


def check_val(t_set:pd.DataFrame, g_points):
    print("Total test_set length = "+str(len(t_set)))
    print(test_set)
    cols = t_set.columns.values.tolist()
    while True:
        distance_min = sys.maxsize
        val = input("Enter test_set index: ")
        i_val = int(val)
        #print(val)
        if i_val < 0:
            break
        print(t_set.iloc[[val]])

        for cnt_points, j in g_points.iterrows():
            dis = euclidean_distance(t_set.iloc[[val]], j, cols)
            print(cnt_points, dis)
            if dis < distance_min:
                distance_min = dis
                result_grp = j
                result_idx = cnt_points

        print(result_idx)



if __name__ == '__main__':
    config_file = open('config.json')
    config = json.load(config_file)
    dataset = config['dataset_name']
    seperator = config['sep']
    header_d = config['head']
    if header_d == "None":
        df = read_data(dataset, seperator, None)
        #print(len(df.columns))
        df_col = []
        for e in range(len(df.columns)):
            df_col.append('A' + str(e))
        df.columns = df_col
    else:
        df = read_data(dataset, seperator, "Y")


    split_times = 1
    num_of_grp_v = input("Enter Number of groups: ")
    num_of_grp = int(num_of_grp_v)

    # num_of_grp = 10
    for x in range(split_times):
        print("Start")
        train_set, test_set = split_dataset(df, x)
        #classification = config['classification']
        df_all_points = pd.DataFrame()
        group_pnts = knn(train_set, num_of_grp)
        print(group_pnts)
        check_val(test_set, group_pnts)

        #check_test_set(test_set, df_all_points, 0, class_col_name)
    print("End")
