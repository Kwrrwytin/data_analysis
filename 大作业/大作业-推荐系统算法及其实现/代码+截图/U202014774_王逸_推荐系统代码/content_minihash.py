import numpy as np
import pandas as pd
import random as rd
import math
from sklearn.metrics.pairwise import cosine_similarity


def read_movie_info(movie_file):
    movie_data = pd.read_csv(movie_file)
    ids = movie_data['movieId']
    title = movie_data['title']
    genre = movie_data['genres']

    movies_genre = []
    movie_ids = []
    movie_info = {}

    for gens in genre:
        genre_arr = gens.split("|")
        movies_genre.append(genre_arr)

    for i in range(len(ids)):
        movie = [ids[i], title[i], movies_genre[i]]
        movie_info[i] = movie
        movie_ids.append((int(ids[i])))

    return movie_info, movie_ids


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


def get_user_movie_dict(u_m):
    u_rate = {}
    user_movie = {}

    for i in u_m:
        u_m_rank = (i[1], i[2])
        # user dict
        if i[0] in u_rate:
            u_rate[i[0]].append(u_m_rank)
        else:
            u_rate[i[0]] = [u_m_rank]
        # user-movie dict
        if i[0] in user_movie:
            user_movie[i[0]].append(i[1])
        else:
            user_movie[i[0]] = [i[1]]

    return u_rate, user_movie


def mini_hash(genre_movie, movie_num, genre_num, hash_num=10):

    inf = 100000000000
    sig = np.zeros([hash_num, movie_num])
    ab_co = np.zeros([hash_num, 2])
    temp = int(math.sqrt(hash_num))

    for i in range(hash_num):
        ab_co[i, 0] = rd.randint(1, temp * 2)
        ab_co[i, 1] = rd.randint(1, temp * 2)
        for j in range(movie_num):
            sig[i, j] = inf

    for row in range(genre_num):
        # 计算随机排列行号
        print("---mini hash processing {}---".format(row))
        hashes = []
        for i in range(hash_num):
            hashes.append((ab_co[i, 0] * row + ab_co[i, 1]) % genre_num)

        for col in range(movie_num):
            if genre_movie[row, col] == 0:
                continue
            else:
                for k in range(hash_num):
                    sig[k, col] = min(sig[k, col], hashes[k])

    return sig


def calculate_jaccard(user1, user2):
    inter = 0
    r = len(user1)
    for i in range(r):
        if user1[i] == user2[i]:
            inter += 1
    return float(inter / r)


def calculate_sig(movie_info):
    # 电影类别集合
    genre_list = []
    for i, movie in movie_info.items():
        for genre in movie[2]:
            if genre not in genre_list:
                genre_list.append(genre)

    movie_num = len(movie_info)
    genre_num = len(genre_list)
    print(genre_num)
    genre_movie = np.zeros([genre_num, movie_num])

    # 01 效用矩阵
    for i, (key, movie) in enumerate(movie_info.items()):
        for genre in movie[2]:
            g = genre_list.index(genre)
            genre_movie[g, i] = 1
    sig = mini_hash(genre_movie, movie_num, genre_num)
    return sig


def calculate_sim(sig, movie_num):
    sig = sig.T
    sim = np.zeros([movie_num, movie_num])
    for i in range(movie_num):
        print("-------sim {}-------".format(i))
        for j in range(i, movie_num):
            sim[i, j] = calculate_jaccard(sig[i], sig[j])
            sim[j, i] = sim[i, j]
    return sim


def recommend(user_rate, user_id, movie_id, sig, user_movie):
    sig = sig.T
    movie_num = len(movie_id)
    rated = user_rate[user_id]
    u_rated_num = len(rated)

    rec_list = []
    rec_dict = {}

    for i in range(movie_num):
        if movie_id[i] not in user_movie[user_id]:
            s1 = 0
            s2 = 0
            s3 = 0
            for movie in rated:
                row = movie_id.index(movie[0])
                sim = calculate_jaccard(sig[row], sig[i])
                if sim > 0:
                    # 已评分
                    if movie_id[i] in rated:
                        continue
                    else:
                        # 未评分
                        s1 += sim * movie[1]
                        s2 += sim
                        s3 += movie[1]
                else:
                    continue
            if s2 == 0:
                # 无人评分， 取已打分电影平均值
                pre_score = s3 / u_rated_num
            else:
                pre_score = s1 / s2

            rec_list.append([pre_score, i])
            rec_dict[movie_id[i]] = pre_score

    rec_list.sort(reverse=True)
    return rec_list, rec_dict


def predict_score(user_rate, user_id, movie_id, sig, m_id):
    sig = sig.T
    rated = user_rate[user_id]
    u_rated_num = len(rated)
    col = movie_id.index(m_id)

    s1 = 0
    s2 = 0
    s3 = 0
    for movie in rated:
        row = movie_id.index(movie[0])
        sim = calculate_jaccard(sig[row], sig[col])
        if sim > 0:
            if movie_id[col] in rated:
                continue
            else:
                s1 += sim * movie[1]
                s2 += sim
                s3 += movie[1]
        else:
            continue
    if s2 == 0:
        pre_score = s3 / u_rated_num
    else:
        pre_score = s1 / s2
    return pre_score


if __name__ == '__main__':
    movie_info, movie_ids = read_movie_info('movies.csv')
    u_m_rate = read_rating_data('train_set.csv')
    user_rate, user_movie = get_user_movie_dict(u_m_rate)

    sig = calculate_sig(movie_info)
    # sim = calculate_sim(sig, 9125)

    user_id = int(input('请输入被推荐用户ID(0：运行测试集)：'))
    if user_id != 0:
        n = int(input('推荐电影个数:'))
        recommend_list, recommend_dict = recommend(user_rate, user_id, movie_ids, sig, user_movie)
        print("{0:6}\t{1:6}\t{2:65}\t{3}".format('MOVIE ID', 'Pre Score', 'TITLE', 'GENRE'))
        for i in range(n):
            j = recommend_list[i][1]
            print('{0:<10}\t{1:.6f}\t{2:65}\t{3}'.format(movie_info[j][0], recommend_list[i][0], movie_info[j][1],
                                                       movie_info[j][2]))
    else:
        test_data = pd.read_csv("test_set.csv")
        user_id = test_data['userId']
        movie_id = test_data['movieId']
        rating = test_data['rating']
        m = len(user_id)
        sse = 0
        print('{0:4}\t{1:6}\t{2:6}\t{3:6}'.format('用户ID', '电影ID', '预期评分', '实际评分'))
        for i in range(m):
            pre_score = predict_score(user_rate, user_id[i], movie_ids, sig, movie_id[i])
            print('{0:4}\t{1:<6}\t{2:.6f}\t{3:.6f}'.format(user_id[i], movie_id[i], pre_score, rating[i]))
            sse += (pre_score - rating[i]) ** 2
        print("SSE=", sse)
