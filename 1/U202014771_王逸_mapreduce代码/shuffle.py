import threading
import time


def shuffle(readfile):
    input_file = open(readfile)
    output1 = open('./data/shuffle_data/shuffle01', 'a')
    output2 = open('./data/shuffle_data/shuffle02', 'a')
    output3 = open('./data/shuffle_data/shuffle03', 'a')

    list1 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
    list2 = ['j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q']

    for line in input_file:
        line = line.strip()
        word, cnt = line.split(',', 1)
        if word[0] in list1:
            output1.write("{},{}\n".format(word, cnt))
        elif word[0] in list2:
            output2.write("{},{}\n".format(word, cnt))
        else:
            output3.write("{},{}\n".format(word, cnt))


if __name__ == '__main__':
    th1 = threading.Thread(target=shuffle('./data/combine_data/combine01'), args=("th1",))
    th2 = threading.Thread(target=shuffle('./data/combine_data/combine02'), args=("th2",))
    th3 = threading.Thread(target=shuffle('./data/combine_data/combine03'), args=("th3",))
    th4 = threading.Thread(target=shuffle('./data/combine_data/combine04'), args=("th4",))
    th5 = threading.Thread(target=shuffle('./data/combine_data/combine05'), args=("th5",))
    th6 = threading.Thread(target=shuffle('./data/combine_data/combine06'), args=("th6",))
    th7 = threading.Thread(target=shuffle('./data/combine_data/combine07'), args=("th7",))
    th8 = threading.Thread(target=shuffle('./data/combine_data/combine08'), args=("th8",))
    th9 = threading.Thread(target=shuffle('./data/combine_data/combine09'), args=("th9",))

    print("==========shuffle time=========")
    start = time.perf_counter()
    th1.start()
    th2.start()
    th3.start()
    th4.start()
    th5.start()
    th6.start()
    th7.start()
    th8.start()
    th9.start()

    th1.join()
    print("th1: %s s" % (time.perf_counter() - start))
    th2.join()
    print("th2: %s s" % (time.perf_counter() - start))
    th3.join()
    print("th3: %s s" % (time.perf_counter() - start))
    th4.join()
    print("th4: %s s" % (time.perf_counter() - start))
    th5.join()
    print("th5: %s s" % (time.perf_counter() - start))
    th6.join()
    print("th6: %s s" % (time.perf_counter() - start))
    th7.join()
    print("th7: %s s" % (time.perf_counter() - start))
    th8.join()
    print("th8: %s s" % (time.perf_counter() - start))
    th9.join()
    print("th9: %s s" % (time.perf_counter() - start))

