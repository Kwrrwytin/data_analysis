import threading
import time


def reduce(readfile, writefile):
    input_file = open(readfile)
    output = open(writefile, 'w')
    cnt_dict = {}

    for line in input_file:
        line = line.strip()
        word, cnt = line.split(',', 1)
        try:
            cnt = int(cnt)
        except ValueError:
            continue

        if word in cnt_dict.keys():
            cnt_dict[word] += cnt
        else:
            cnt_dict[word] = cnt

    cnt_dict = sorted(cnt_dict.items(), key=lambda x: x[0])

    for key, value in cnt_dict:
        output.write("{},{}\n".format(key, value))


if __name__ == '__main__':
    th1 = threading.Thread(target=reduce('./data/shuffle_data/shuffle01', './data/reduce_data/reduce01'), args=("th1",))
    th2 = threading.Thread(target=reduce('./data/shuffle_data/shuffle02', './data/reduce_data/reduce02'), args=("th2",))
    th3 = threading.Thread(target=reduce('./data/shuffle_data/shuffle03', './data/reduce_data/reduce03'), args=("th3",))

    print("==========reduce time=========")
    start = time.perf_counter()
    th1.start()
    th2.start()
    th3.start()

    th1.join()
    print("th1: %s s" % (time.perf_counter() - start))
    th2.join()
    print("th2: %s s" % (time.perf_counter() - start))
    th3.join()
    print("th3: %s s" % (time.perf_counter() - start))
    