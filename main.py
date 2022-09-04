import sys
# import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from IPython.display import display
import json


def read_data(dataset: pd.DataFrame, sep, names=None):
    df = pd.read_csv(dataset, sep=sep, names=names)
    df = pd.DataFrame(df)
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    return df


def split_dataset(df, x):
    mask_test = int(df.shape[0] * (20 / 100))
    test_start = x * mask_test
    test_end = test_start + mask_test
    test_set = df[test_start:test_end]
    train_set = pd.concat([df[:test_start], df[test_end:]])
    return train_set, test_set


def euclidean_distance(row1, row2, col):
    distance = 0
    for c in col[:-1]:
        row1_val = getattr(row1, c)
        row2_val = getattr(row2, c)
        distance += (row1_val - row2_val) ** 2
    # print(distance)
    return sqrt(distance)


def get_relevant_points(dataset, points, dis_min):
    # print(points)
    # distance = [[-1 for x in range(n)]for y in range(len(dataset))]
    df = pd.concat([dataset, points], sort=False)
    df = df.drop_duplicates(keep=False)
    min_dist = dis_min
    min_dis_point = -1
    avg_dis = 0
    for cnt_df, i in df.iterrows():

        for cnt_points, j in points.iterrows():
            dis = euclidean_distance(i, j, dataset.columns.values.tolist())
            if dis < min_dist:
                min_dis_point = i.to_dict()
                min_dist = dis

    # print(min_dist, '\n', min_dis_point)
    _list = []
    _list.append(min_dis_point)
    min_dis_point = pd.DataFrame(_list)
    print(min_dis_point)


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


def get_grp_point(dataset, n, test_points, div=False):
    avg_dis = 0
    global test_list
    global divided_grps
    test_list1 = [[] for y in range(n)]
    for cnt_df, i in dataset.iterrows():
        min_dist = sys.maxsize
        min_idx = -1
        for cnt_points, j in enumerate(test_points.iterrows()):
            #print(i[0])
            dis = euclidean_distance(i, j[1], dataset.columns.values.tolist())
            if dis < min_dist:
                # min_dis_point = i.to_dict()
                min_dist = dis
                min_idx = cnt_points
        # print(min_idx)
        test_list1[min_idx].append(i)
    # print(test_list)
    divided_grps = test_list1
    if div:
        return
    avg_pts = get_avg_point(test_list1, n)
    _list = []
    for item in avg_pts:
        _list.append(item.to_dict())
    avg_pts_tt = pd.DataFrame(_list)

    for x in range(n):
        for index, row1 in test_points.iloc[[0]].iterrows():
            for index2, row2 in avg_pts_tt.iloc[[x]].iterrows():
                dis = euclidean_distance(row1, row2, dataset.columns.values.tolist())
                if dis < 1000:
                    test_list = avg_pts_tt
                    return
    get_grp_point(dataset, n, avg_pts_tt)


def divide_sets():
    pass


def knn(df_knn: pd.DataFrame, group_cnt):
    global test_list
    get_grp_point(df_knn[:][group_cnt:], group_cnt, df_knn[:][:group_cnt])
    optimum_points = test_list
    #print(optimum_points)
    return optimum_points


def check_test_set(df_knn_test: pd.DataFrame, points: pd.DataFrame, match_attr):
    #test_list1 = [[] for y in range(len(df_knn_test))]
    print(df_knn_test.columns.values.tolist())
    print(points.columns.values.tolist())
    for cnt_df, i in df_knn_test.iterrows():
        print("\nst")
        print(i)
        min_dist = sys.maxsize
        min_idx = -1
        min_item = 0
        for cnt_points, j in enumerate(points.iterrows()):
            # print(cnt_points)

            dis = euclidean_distance(i, j[1], df_knn_test.columns.values.tolist())
            if dis < min_dist:
                min_dist = dis
                min_idx = cnt_points
                min_item = j

        #test_list1[min_idx].append(i)
        print(min_item)
    # print(test_list)
    #divided_grps = test_list1


if __name__ == '__main__':
    config_file = open('config.json')
    config = json.load(config_file)
    dataset = config['dataset_name']
    seperator = config['sep']
    df = read_data(dataset, seperator)
    # print(df)
    split_times = 1
    num_of_grp = len(df.columns)
    for x in range(split_times):
        print("Start")
        train_set, test_set = split_dataset(df, x)
        # death_set = train_set.loc[train_set['DEATH_EVENT'] == 1]

        classification = config['classification']
        class_col_name = train_set.columns[config['class_col']]
        if classification:
            df_class = train_set.groupby(train_set.columns[config['class_col']])
            df_test_class = test_set.groupby(test_set.columns[config['class_col']])
            print(df_class.groups.keys())
            df_all_points = pd.DataFrame()
            for name, df_group in df_class:
                group_pnts = knn(df_group, num_of_grp - 1)
                df_all_points = pd.concat([df_all_points, group_pnts], ignore_index = True)

            #print(df_all_points)
            '''df_test_class = train_set.groupby(test_set.columns[config['class_col']])
            for name, df_test_group in df_test_class:
                #print(name,df_test_group)
                check_test_set(df_test_group, df_all_points, name)'''
            #check_test_set(test_set, df_all_points)

        '''num_of_grp = 13
        global test_list
        global divided_grps
        get_grp_point(train_set[:][num_of_grp:], num_of_grp, train_set[:][:num_of_grp])

        optimum_points = test_list

        d_l = optimum_points


        get_grp_point(test_set, len(d_l), d_l, True)

        d_grp = divided_grps
        _df_lst_1 = []
        for df_lists_1 in d_grp:
            _ll = []
            for df_s in df_lists_1:
                _ll.append(df_s.to_dict())
            _df_lst_1.append(pd.DataFrame(_ll))

        print(_df_lst_1)
        for cnt, d_f_plt in enumerate(_df_lst_1):
            d_f_plt.to_csv("ans_"+str(x)+"_"+str(cnt)+".csv")'''

    print("End")
