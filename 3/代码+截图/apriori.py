import numpy as np
import pandas as pd


"""
support
confidence
step1:根据最小支持度找出所有频集
step2:根据频繁项目集和最小置信度找出关联规则

k->k+1:连接 + 剪枝
"""


def read_data(data_file):
    data = pd.read_csv(data_file)
    item = data['items']
    data = np.array(item)
    item_list = []
    for items in data:
        items = items.strip('{').strip('}').split(',')
        item = []
        for i in items:
            item.append(i)
        item_list.append(item)
    data = item_list
    return data


def init_c1(data):
    c1 = set()
    for items in data:
        for item in items:
            item_set = frozenset([item])
            # print(item_set)
            c1.add(item_set)
    return c1


def prune_apriori(ck_item, lk_1):
    for item in ck_item:
        sub_set = ck_item - frozenset([item])
        if sub_set not in lk_1:
            return False
    return True


def concat_ck(lk, k):
    ck = set()
    list_k = list(lk)
    for i in range(len(lk)):
        for j in range(i+1, len(lk)):
            l1_prefix = list(list_k[i])[0:k - 2]
            l2_prefix = list(list_k[j])[0:k - 2]
            l1_prefix.sort()
            l2_prefix.sort()

            if l1_prefix == l2_prefix:
                ck_item = list_k[i] | list_k[j]
                if prune_apriori(ck_item, lk):
                    ck.add(ck_item)

    return ck


def calculate_lk(data, ck, min_sup, support_data):
    lk = set()
    x_cnt = {}
    for t in data:
        for x in ck:
            if x.issubset(t):
                if x not in x_cnt:
                    x_cnt[x] = 1
                else:
                    x_cnt[x] += 1
    d = float(len(data))
    for x in x_cnt:
        if (x_cnt[x] / d) >= min_sup:
            lk.add(x)
            support_data[x] = x_cnt[x] / d
    return lk


def get_rule(l, support_data, min_conf):
    """
    :param l: 所有频集
    :param support_data: 支持度
    :param min_conf: 最小置信度
    :return:
    """
    rule_list = []
    sub_set_list = []
    for i in range(len(l)):
        for freq_set in l[i]:
            for sub_set in sub_set_list:
                if sub_set.issubset(freq_set):
                    conf = support_data[freq_set] / support_data[sub_set]
                    rule = (sub_set, freq_set-sub_set, conf)
                    if conf >= min_conf and rule not in rule_list:
                        rule_list.append(rule)
            sub_set_list.append(freq_set)
    return rule_list


if __name__ == "__main__":
    data = read_data("./Groceries.csv")
    min_sup = 0.005
    min_conf = 0.5
    support_data = {}
    L_all = []

    c1 = init_c1(data)
    l1 = calculate_lk(data, c1, min_sup, support_data)
    lk = l1.copy()
    L_all.append(lk)
    for k in range(2, 4):
        ck = concat_ck(lk, k)
        lk = calculate_lk(data, ck, min_sup, support_data)
        lk = lk.copy()
        L_all.append(lk)
    # 关联规则
    rule_list = get_rule(L_all, support_data, min_conf)

    for i, lk in enumerate(L_all):
        with open('L{}.csv'.format(i+1), 'w') as f:
            cnt = 0
            for key in lk:
                cnt += 1
                f.write('{},\t{}\n'.format(key, support_data[key]))
            print("{} 阶频繁项集数量为: {}.\n".format(i+1, cnt))

    with open('rules.csv', 'w') as f:
        cnt = 0
        for item in rule_list:
            cnt += 1
            f.write('{}\t{}\t{}\t: {}\n'.format(item[0], "of", item[1], item[2]))
        print("关联规则的数量为: {}.\n".format(cnt))
    print('--------end--------')

