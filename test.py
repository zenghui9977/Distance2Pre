from utils import *
from Geo_indistinguishability import *
import os
import kd_tree
from collections import OrderedDict


WHOLE = './poidata/'
PATH_f = os.path.join(WHOLE, './Foursquare')
PATH_g = os.path.join(WHOLE, './Gowalla')
PATH = PATH_f

FILE_NAME = 'test.txt'

PROCESS_DATA_FOLDER = 'process_data/Foursquare/'

dd = 150 / 1000.0
UD = 5


dist_num = int(UD/dd)  # 1520 = 38*1000/25。idx=[0, 1519+1]


percent = 0.7
[(user_num, item_num), pois_cordis, (auxiliary_domain_tra_pois, auxiliary_domain_tes_pois),
            (auxiliary_domain_tra_dist, auxiliary_domain_tes_dist),
            (target_domain_tra_pois, target_domain_tes_pois), (target_domain_tra_dist, target_domain_tes_dist)] = \
    load_data(os.path.join(PATH, FILE_NAME), 'test', -1, dd, dist_num, PROCESS_DATA_FOLDER, percent)


print('user_num \t\t\t %s' % user_num)
print('item_num \t\t\t %s' % item_num)



tree = kd_tree.create(pois_cordis)
m = 4
epsilon = 2

print('辅助域数据加噪')
obfuscated_auxiliary_domain_data = obfuscate_visited_data_list(auxiliary_domain_tra_pois, pois_cordis, tree, epsilon=epsilon)
print('计算置信矩阵')
confidence_matrix = compute_confidence_matrix(tree, obfuscated_auxiliary_domain_data, m, pois_cordis, epsilon)

print('obfuscated_auxiliary_domain_data \t\t\t %s' % obfuscated_auxiliary_domain_data)
print('confidence_matrix \t\t\t %s' % confidence_matrix)

print(compute_distance(obfuscated_auxiliary_domain_data, pois_cordis, dd, dist_num))

save_as_pkl(PROCESS_DATA_FOLDER + 'auxiliary_domain/', 'obfuscated_pois', obfuscated_auxiliary_domain_data)
save_as_pkl(PROCESS_DATA_FOLDER + 'auxiliary_domain/', 'confidence_matrix', confidence_matrix)

