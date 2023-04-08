import random as rd
import numpy as np
import pandas as pd
import math


def read_movie_data(movie_file):
    movie_data = pd.read_csv(movie_file)
    ids = movie_data['movieId']
    title = movie_data['title']
    genre = movie_data['genres']

    movies_genre = {}
    movie_titles = {}

    for i, gens in enumerate(genre):
        genre_arr = gens.split("|")
        movies_genre[ids[i]] = genre_arr

    for i, t in enumerate(title):
        movie_titles[ids[i]] = t

    return movies_genre, movie_titles


def read_rating_data(rating_file):
    train_file = open(rating_file)
    ratings = train_file.readlines()
    train_file.close()
    ratings.pop(0)

    u_m_rate = []
    for line in ratings:
        rate = line.strip().split(',')
        u_m_rate.append([int(rate[0]), int(rate[1]), float(rate[2])])

    return u_m_rate


def mini_hash(user_movie, user_num, movie_num, hash_num=10):
    movie_user = user_movie.T

    inf = 100000000000
    sig = np.zeros([hash_num, user_num])
    # 模拟随机哈希函数的系数矩阵
    ab_co = np.zeros([hash_num, 2])
    temp = int(math.sqrt(hash_num))

    for i in range(hash_num):
        ab_co[i, 0] = rd.randint(1, temp * 2)
        ab_co[i, 1] = rd.randint(1, temp * 2)
        for j in range(user_num):
            sig[i, j] = inf

    for row in range(movie_num):
        # 计算随机排列行号
        if row % 910 == 0:
            print("---mini hash processing {}%---".format(row/91))
        hashes = []
        for i in range(hash_num):
            hashes.append((ab_co[i, 0] * row + ab_co[i, 1]) % movie_num)

        for col in range(user_num):
            if movie_user[row, col] == 0:
                continue
            else:
                for k in range(hash_num):
                    sig[k, col] = min(sig[k, col], hashes[k])
    print("---end---")
    return sig


def calculate_jaccard(user1, user2):
    inter = 0
    r = len(user1)
    for i in range(r):
        if user1[i] == user2[i]:
            inter += 1
    return float(inter / r)


# 签名矩阵
def calculate_co_matrix(u_m_rate):
    movie_ids = []
    for c in u_m_rate:
        if c[1] not in movie_ids:
            movie_ids.append(c[1])
    #   构建用户-电影效用矩阵
    movie_num = len(movie_ids)
    user_movie = np.zeros([671, movie_num])
    for comment in u_m_rate:
        m_id = movie_ids.index(comment[1])
        user_movie[comment[0] - 1, m_id] = comment[2]

    #   01处理：0.5-2.5->0, 3.0-5.0->1
    for i in range(671):
        for j in range(movie_num):
            if 0.5 <= user_movie[i, j] <= 2.5:
                user_movie[i, j] = 0
            else:
                user_movie[i, j] = 1

    # 随机数映射
    # print(movie_num)
    # arr = random.sample([i for i in range(0, movie_num+1)], movie_num)
    # print(arr)
    sig = mini_hash(user_movie, 671, movie_num, hash_num=20)
    return sig


def get_user_movie_dict(u_m):
    u_rate = {}
    movie_user = {}

    for i in u_m:
        u_m_rank = (i[1], i[2])
        # user dict
        if i[0] in u_rate:
            u_rate[i[0]].append(u_m_rank)
        else:
            u_rate[i[0]] = [u_m_rank]
        # movie-user dict
        if i[1] in movie_user:
            movie_user[i[1]].append(i[0])
        else:
            movie_user[i[1]] = [i[0]]

    return u_rate, movie_user


def get_near_k(user_id, user_rate, movie_user, sig, k):
    neighbors = []
    nei_dist = []
    for m_rate in user_rate[user_id]:
        # 对于每个给该电影评分的用户
        for m_user in movie_user[m_rate[0]]:
            if m_user != user_id and m_user not in neighbors:
                neighbors.append(m_user)
                dist = calculate_jaccard(sig[:, user_id-1], sig[:, m_user-1])
                # dist = user_user[user_id-1, m_user-1]
                nei_dist.append([m_user, dist])
    nei_dist.sort(reverse=True)
    return nei_dist[:k]


def predict_score(user_id, movie_id, user_rate, movie_user, sig, k):
    # 计算邻居的每一部电影与被推荐用户之间的相似度大小
    neighbor_dist = get_near_k(user_id, user_rate, movie_user, sig, k)

    score_sum = 0
    for m_rate in user_rate[user_id]:
        score_sum += m_rate[1]
    acc = score_sum / len(user_rate[user_id])
    pre_sum = 0
    co_sum = 0
    for user in neighbor_dist:
        s1 = 0
        if calculate_jaccard(sig[:, user_id-1], sig[:, user[0]-1]) < 0.5:
            continue
        movies_rate = user_rate[user[0]]
        # 计算每一部电影对用户的推荐程度大小
        s = 0
        for movie in movies_rate:
            s += movie[1]
        nei_acc = s / len(movies_rate)
        for movie in movies_rate:
            if movie[0] == movie_id:
                s1 += user[1]
                pre_sum += user[1] * (movie[1] - nei_acc)
        # 该用户没有对此电影进行评分, 平均值
        if s1 == 0:
            s1 = user[1]
        co_sum += s1
    pre_score = pre_sum / co_sum + acc
    return pre_score


def recommend(user_id, user_rate, movie_user, sig, k):
    neighbor_dist = get_near_k(user_id, user_rate, movie_user, sig, k)

    rec_dict = {}
    rec_movie = {}

    s = 0
    for movie in user_rate[user_id]:
        s += movie[1]
    u_acc = s / len(user_rate[user_id])

    for user in neighbor_dist:
        if calculate_jaccard(sig[:, user_id - 1], sig[:, user[0] - 1]) < 0.5:
            continue
        movies = user_rate[user[0]]
        s = 0
        for movie in movies:
            s += movie[1]
        n_acc = s / len(movies)
        for movie in movies:
            if movie[0] not in rec_dict:
                rec_dict[movie[0]] = user[1]
                rec_movie[movie[0]] = user[1] * (movie[1] - n_acc)
            else:
                rec_dict[movie[0]] += user[1]
                rec_movie[movie[0]] += user[1] * (movie[1] - n_acc)

    recommend_list = []
    for mid in rec_dict:
        rec_dict[mid] = rec_movie[mid] / rec_dict[mid] + u_acc
        recommend_list.append([rec_dict[mid], mid])

    recommend_list.sort(reverse=True)
    return recommend_list, rec_dict


if __name__ == '__main__':
    movies_genre, movie_titles = read_movie_data('movies.csv')
    u_m_rate = read_rating_data('ratings.csv')
    sig = calculate_co_matrix(u_m_rate)
    user_rate, movie_user = get_user_movie_dict(u_m_rate)
    user_id = int(input('被推荐用户ID（0：运行测试集）:'))
    if user_id != 0:
        movie_id = 1
        k = int(input('K:'))
        n = int(input('N:'))
        rec_list, rec_dict = recommend(user_id, user_rate, movie_user, sig, k)
        # print n recommendations
        print("{0:6}\t{1:6}\t{2:50}\t{3}".format('MOVIE ID', 'Pre Score', 'TITLE', 'GENRE'))
        for recommendation in rec_list[:n]:
            movie_id = recommendation[1]
            print("{0:<10}\t{1:.6f}\t{2:<50}\t{3}".format(movie_id, rec_dict[movie_id], movie_titles[movie_id],
                                                         movies_genre[movie_id]))
    else:
        data = pd.read_csv('test_set.csv')
        user_id = data['userId']
        movie_id = data['movieId']
        rating = data['rating']
        m = len(user_id)
        k = 80
        sse = 0
        for i in range(m):
            p_rating = predict_score(user_id[i], movie_id[i], user_rate, movie_user, sig, k)
            print("用户 {0:3} 对电影 {1:2} 的评分预测值为 {2:.6f} ，实际值为 {3:.2f} ".format(user_id[i],movie_id[i],p_rating,rating[i]))
            sse += (p_rating - rating[i]) ** 2
        print("SSE为：", sse)