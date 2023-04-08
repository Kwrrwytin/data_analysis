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


def pearson(user1, user2):
    s1 = 0.0
    s2 = 0.0
    s_12 = 0.0
    avg1 = 0.0
    avg2 = 0.0

    for m_rate in user1:
        avg1 += m_rate[1]
    avg1 = avg1 / len(user1)
    for m_rate in user2:
        avg2 += m_rate[1]
    avg2 = avg2 / len(user2)

    for m_r1 in user1:
        for m_r2 in user2:
            # 评价相同电影
            if m_r1[0] == m_r2[0]:
                s_12 += (m_r1[1] - avg1) * (m_r2[1] - avg2)
                s2 += (m_r2[1] - avg2) * (m_r2[1] - avg2)
            s1 += (m_r1[1] - avg1) * (m_r1[1] - avg1)

    if s_12 == 0.0:
        return 0
    sx_sy = math.sqrt(s1 * s2)
    return s_12 / sx_sy


# 计算相似度矩阵
def calculate_co_matrix(u_m_rate):
    movie_ids = []
    for c in u_m_rate:
        if c[1] not in movie_ids:
            movie_ids.append(c[1])

    user_movie = np.zeros([671, len(movie_ids)])
    for comment in u_m_rate:
        m_id = movie_ids.index(comment[1])
        user_movie[comment[0] - 1, m_id] = comment[2]
    u_u_matrix = np.corrcoef(user_movie)  # 计算用户相关系数矩阵, pearson
    return u_u_matrix


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


def get_near_k(user_id, user_rate, movie_user, user_user, k):
    neighbors = []
    nei_dist = []
    for m_rate in user_rate[user_id]:
        # 对于每个给该电影评分的用户
        for m_user in movie_user[m_rate[0]]:
            if m_user != user_id and m_user not in neighbors:
                neighbors.append(m_user)
                dist = user_user[user_id-1, m_user-1]
                nei_dist.append([m_user, dist])
    nei_dist.sort(reverse=True)
    return nei_dist[:k]


def predict_score(user_id, movie_id, user_rate, movie_user, user_user, k):
    neighbor_dist = get_near_k(user_id, user_rate, movie_user, user_user, k)

    score_sum = 0
    for m_rate in user_rate[user_id]:
        score_sum += m_rate[1]
    acc = score_sum / len(user_rate[user_id])

    # 补充评分：相似用户对该电影的评分与用户平均评分之差的加权平均
    pre_sum = 0
    co_sum = 0
    for user in neighbor_dist:
        if user_user[user_id-1, user[0]-1] < 0:
            continue
        movies_rate = user_rate[user[0]]
        s = 0
        for movie in movies_rate:
            s += movie[1]
        nei_acc = s / len(movies_rate)
        s1 = 0
        for movie in movies_rate:
            if movie[0] == movie_id:
                s1 += user[1]
                pre_sum += user[1]*(movie[1] - nei_acc)
        # 该用户没有对此电影进行评分, 平均值
        if s1 == 0:
            s1 = user[1]
        co_sum += s1
    pre_score = pre_sum / co_sum + acc
    return pre_score


def recommend(user_id, user_rate, movie_user, user_user, k):
    # 选取最相似的k个用户
    neighbor_dist = get_near_k(user_id, user_rate, movie_user, user_user, k)

    rec_dict = {}  # (id, sum_coef)
    rec_movie = {}  # (id, pre_score)

    s = 0
    for movie in user_rate[user_id]:
        s += movie[1]
    # 对其已看过电影的平均评分u_acc
    u_acc = s / len(user_rate[user_id])
    # 补充评分：相似用户对该电影的评分
    for user in neighbor_dist:
        if user_user[user_id-1, user[0]-1] < 0:
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
    user_user = calculate_co_matrix(u_m_rate)
    user_rate, movie_user = get_user_movie_dict(u_m_rate)

    user_id = int(input('被推荐用户ID（0：运行测试集）:'))
    if user_id != 0:
        movie_id = 1
        k = int(input('K:'))
        n = int(input('N:'))
        rec_list, rec_dict = recommend(user_id, user_rate, movie_user, user_user, k)
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
            p_rating = predict_score(user_id[i], movie_id[i], user_rate, movie_user, user_user, k)
            print("用户 {0:3} 对电影 {1:2} 的评分预测值为 {2:.6f} ，实际值为 {3:.2f} ".format(user_id[i],movie_id[i],p_rating,rating[i]))
            sse += (p_rating - rating[i]) ** 2
        print("SSE为：", sse)
