import numpy as np
import pandas as pd
import os
import pickle

from math import sin, cos, sqrt, asin

def save_as_csv(data_dir, file_name, data):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    np.savetxt(data_dir+file_name, data, delimiter=',')

def save_as_npy(data_dir, file_name, data):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    np.save(data_dir+file_name, data)

def save_as_pkl(data_dir, filename, data):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    filehandler = open(data_dir + "/" + filename + ".pkl", "wb")
    pickle.dump(data, filehandler)
    filehandler.close()

# def save_as_pt(data_dir, filename, data):
#     if not os.path.exists(data_dir):
#         os.makedirs(data_dir)
#     torch.save(data, data_dir + filename)

def read_from_npy_dict(data_dir, file_name):
    return np.load(data_dir + file_name).item()

def read_from_npy(data_dir, file_name):
    return np.load(data_dir + file_name)

def read_from_pkl(save_dir, filename):
    return pickle.load(open(save_dir + filename + '.pkl' , 'rb'))
#
# def read_from_pt(save_dir, filename):
#     return torch.load(save_dir + filename)


# 切分数据，将数据分为辅助域与目标域
def split_original_data(data, percent):
    '''
    data: 原始数据
    percent: 第一个返回值数据占比
    '''
    data_num = len(data)

    auxiliary_domain_data_num = int(data_num * percent)
    auxiliary_domain_data = data[:auxiliary_domain_data_num]
    target_domain_data = data[auxiliary_domain_data_num:]

    return auxiliary_domain_data, target_domain_data


def cal_dis(lat1, lon1, lat2, lon2, dd, dist_num):
    """
    Haversine公式: 计算两个latitude-longitude点之间的距离. [ http://www.cnblogs.com/softidea/p/6925673.html ]
    二倍角公式：cos(2a) = 1 - 2sin(a)sin(a)，即sin(a/2)*sin(a/2) = (1 - cos(a))/2
    dd = 25m，距离间隔。
    """
    d = 12742                           # 地球的平均直径。
    p = 0.017453292519943295            # math.pi / 180, x*p由度转换为弧度。
    a = (lat1 - lat2) * p
    b = (lon1 - lon2) * p
    # c = pow(sin(a / 2), 2) + cos(lat1 * p) * cos(lat2 * p) * pow(sin(b / 2), 2)     # a/b别混了。
    c = (1.0 - cos(a)) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1.0 - cos(b)) / 2     # 二者等价，但这个更快。
    dist = d * asin(sqrt(c))            # 球面上的弧线距离(km)

    interval = int(dist / dd)    # 该距离落在哪个距离间隔区间里。
    interval = min(interval, dist_num)
    # 间隔区间范围是[0, 379+1]。即额外添加一个idx=380, 表示两点间距>=38km。
    # 对应的生成分析计算出来的各区间概率，也添加一个位置来表示1520的概率，就是0.
    return interval


def load_data(dataset, mode, split, dd, dist_num, process_folder, percent):
    if not os.path.exists(process_folder + 'user_item_num.pkl'):
        """
        加载购买记录文件，生成数据。
        dd = 25m, dt = 60min,
        """
        # 用户购买历史记录，原纪录. 嵌套列表, 元素为一个用户的购买记录(小列表)
        print('Original data ...')
        pois = pd.read_csv(dataset, sep=' ')

        all_user_pois = [[i for i in upois.split('/')] for upois in pois['u_pois']]
        all_user_cods = [[i.split(',') for i in ucods.split('/')] for ucods in pois['u_coordinates']]       # string
        # 删除Gowalla里的空值：'LOC_null'和['null', 'null']
        if 'Gowalla' in dataset:
            tmp_pois, tmp_cods = [], []
            for upois in all_user_pois:
                if 'LOC_null' in upois:
                    tmp = [poi for poi in upois if poi != 'LOC_null']
                    tmp_pois.append(tmp)
                else:
                    tmp_pois.append(upois)
            for ucods in all_user_cods:
                if ['null', 'null'] in ucods:
                    tmp = [cod for cod in ucods if cod != ['null', 'null']]
                    tmp_cods.append(tmp)
                else:
                    tmp_cods.append(ucods)
            all_user_pois, all_user_cods = tmp_pois, tmp_cods
        all_user_cods = [[[float(ucod[0]), float(ucod[1])] for ucod in ucods] for ucods in all_user_cods]   # float
        all_trans = [item for upois in all_user_pois for item in upois]
        all_cordi = [ucod for ucods in all_user_cods for ucod in ucods]
        poi_cordi = dict(zip(all_trans, all_cordi))  # 每个poi都有一个对应的的坐标。
        tran_num, user_num, item_num = len(all_trans), len(all_user_pois), len(set(all_trans))
        print('\tusers, items, trans:  = {v1}, {v2}, {v3}'.format(v1=user_num, v2=item_num, v3=tran_num))
        print('\tavg. user check:      = {val}'.format(val=1.0 * tran_num / user_num))
        print('\tavg. poi checked:     = {val}'.format(val=1.0 * tran_num / item_num))
        print('\tdistance interval     = [0, {val}]'.format(val=dist_num))

        # 选取训练集、验证集(测试集)，并对test去重。不管是valid还是test模式，统一用train，test表示。
        # print('Split the training set, test set: mode = {val} ...'.format(val=mode))
        tra_pois, tes_pois = [], []
        tra_dist, tes_dist = [], []                 # idx0 = max_dist, idx1 = 0/1的间距并划分到距离区间里。
        for upois, ucods in zip(all_user_pois, all_user_cods):
            left, right = upois[:split], [upois[split]]   # 预测最后一个。此时right只有一个idx，加[]变成list。

            # 两个POI之间距离间隔落在哪个区间。
            dist = []
            for i, cord in enumerate(ucods[1:]):    # 从idx=1和idx=0的距离间隔开始算。
                pre = ucods[i]
                dist.append(cal_dis(cord[0], cord[1], pre[0], pre[1], dd, dist_num))
            dist = [dist_num] + dist                # idx=0的距离间隔，就用最大的。
            dist_lf, dist_rt = dist[:split], [dist[split]]

            # 保存
            tra_pois.append(left)
            tes_pois.append(right)
            tra_dist.append(dist_lf)
            tes_dist.append(dist_rt)

        # # 去重后的基本信息。只预测最后一个，这个用不到。
        # all_trans = []
        # for utra, utes in zip(tra_pois, tes_pois):
        #     all_trans.extend(utra)
        #     all_trans.extend(utes)
        # tran_num, user_num, item_num = len(all_trans), len(tra_pois), len(set(all_trans))
        # temp = tra_dist
        # temp.extend(tes_dist)
        # all_dists = [item for upois in temp for item in upois]
        # print('\tusers, items, trans:    = {v1}, {v2}, {v3}'.format(v1=user_num, v2=item_num, v3=tran_num))
        # print('\tavg. user poi:          = {val}'.format(val=1.0 * tran_num / user_num))
        # print('\tavg. item bought:       = {val}'.format(val=1.0 * tran_num / item_num))
        # print('\tdistance interval     = [0, {val}]'.format(val=max_dist))

        # 建立商品别名字典。更新购买记录，替换为0~len(se)-1的别名。
        print('Use aliases to represent pois ...')
        all_items = set(all_trans)
        aliases_dict = dict(zip(all_items, range(item_num)))    # 将poi转换为[0, n)标号。
        tra_pois = [[aliases_dict[i] for i in utra] for utra in tra_pois]
        tes_pois = [[aliases_dict[i] for i in utes] for utes in tes_pois]
        # 根据别名对应关系，更新poi-坐标的表示，以list表示，并且坐标idx就是poi别名替换后的idx。
        cordi_new = dict()
        for poi in poi_cordi.keys():
            cordi_new[aliases_dict[poi]] = poi_cordi[poi]       # 将poi和坐标转换为：poi的[0, n)标号、坐标。
        pois_cordis = [cordi_new[k] for k in sorted(cordi_new.keys())]

        # 目标域与辅助域的切分
        auxiliary_domain_tra_pois, target_domain_tra_pois = split_original_data(tra_pois, percent)
        auxiliary_domain_tes_pois, target_domain_tes_pois = split_original_data(tes_pois, percent)
        auxiliary_domain_tra_dist, target_domain_tra_dist = split_original_data(tra_dist, percent)
        auxiliary_domain_tes_dist, target_domain_tes_dist = split_original_data(tes_dist, percent)

        # 存储对应的数据
        print('saving processed data......')
        save_as_pkl(process_folder, 'user_item_num', [user_num, item_num])
        save_as_pkl(process_folder, 'pois_cordis', pois_cordis)
        save_as_pkl(process_folder + 'auxiliary_domain/', 'tra_tes_pois', (auxiliary_domain_tra_pois, auxiliary_domain_tes_pois))
        save_as_pkl(process_folder + 'auxiliary_domain/', 'tra_tes_dist', (auxiliary_domain_tra_dist, auxiliary_domain_tes_dist))
        save_as_pkl(process_folder + 'target_domain/', 'tra_test_pois', (target_domain_tra_pois, target_domain_tes_pois))
        save_as_pkl(process_folder + 'target_domain/', 'tra_tes_dist', (target_domain_tra_dist, target_domain_tes_dist))

    else:
        # 若已存在处理好的数据直接读取即可
        print('Original data has been preprocessed, load the processed data......')

        user_item_num = read_from_pkl(process_folder, 'user_item_num')
        user_num, item_num = user_item_num[0], user_item_num[1]

        pois_cordis = read_from_pkl(process_folder, 'pois_cordis')

        (auxiliary_domain_tra_pois, auxiliary_domain_tes_pois) = read_from_pkl(process_folder + 'auxiliary_domain/', 'tra_tes_pois')
        (auxiliary_domain_tra_dist, auxiliary_domain_tes_dist) = read_from_pkl(process_folder + 'auxiliary_domain/', 'tra_tes_dist')

        (target_domain_tra_pois, target_domain_tes_pois) = read_from_pkl(process_folder + 'target_domain/', 'tra_test_pois')
        (target_domain_tra_dist, target_domain_tes_dist) = read_from_pkl(process_folder + 'target_domain/', 'tra_tes_dist')

        print('\tusers, items:  = {v1}, {v2}'.format(v1=user_num, v2=item_num))

    return [(user_num, item_num), pois_cordis, (auxiliary_domain_tra_pois, auxiliary_domain_tes_pois),
            (auxiliary_domain_tra_dist, auxiliary_domain_tes_dist),
            (target_domain_tra_pois, target_domain_tes_pois), (target_domain_tra_dist, target_domain_tes_dist)]


def compute_distance(pois, pois_cordis, dd, dist_num):
    all_dist = []
    for u_pois in pois:
        u_pos_cordis = [pois_cordis[u] for u in u_pois]
        dist = []
        for i, cord in enumerate(u_pos_cordis[1:]):
            pre = u_pos_cordis[i]
            dist.append(cal_dis(cord[0], cord[1], pre[0], pre[1], dd, dist_num))
        dist = [dist_num] + dist
        all_dist.append(dist)
    return all_dist