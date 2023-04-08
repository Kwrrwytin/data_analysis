import threading
import time


def combine(readfile, writefile):
    input_data = open(readfile)
    output = open(writefile, 'w')
    cnt_dict = {}
    for line in input_data:
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
    th1 = threading.Thread(target=combine('./data/map_data/mapper01', './data/combine_data/combine01'), args=("th1",))
    th2 = threading.Thread(target=combine('./data/map_data/mapper02', './data/combine_data/combine02'), args=("th2",))
    th3 = threading.Thread(target=combine('./data/map_data/mapper03', './data/combine_data/combine03'), args=("th3",))
    th4 = threading.Thread(target=combine('./data/map_data/mapper04', './data/combine_data/combine04'), args=("th4",))
    th5 = threading.Thread(target=combine('./data/map_data/mapper05', './data/combine_data/combine05'), args=("th5",))
    th6 = threading.Thread(target=combine('./data/map_data/mapper06', './data/combine_data/combine06'), args=("th6",))
    th7 = threading.Thread(target=combine('./data/map_data/mapper07', './data/combine_data/combine07'), args=("th7",))
    th8 = threading.Thread(target=combine('./data/map_data/mapper08', './data/combine_data/combine08'), args=("th8",))
    th9 = threading.Thread(target=combine('./data/map_data/mapper09', './data/combine_data/combine09'), args=("th9",))

    print("==========combine time=========")
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





