# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import threading
import time


def read_line(source_file):
    for line in source_file:
        line = line.strip()
        yield line.split(', ')


def mapper(readfile, writefile):
    word_file = open(readfile)
    output = open(writefile, 'w')
    lines = read_line(word_file)
    for words in lines:
        for word in words:
            output.write("{},{}\n".format(word, 1))


if __name__ == '__main__':
    th1 = threading.Thread(target=mapper('./data/source01', './data/map_data/mapper01'), args=("th1",))
    th2 = threading.Thread(target=mapper('./data/source02', './data/map_data/mapper02'), args=("th2",))
    th3 = threading.Thread(target=mapper('./data/source03', './data/map_data/mapper03'), args=("th3",))
    th4 = threading.Thread(target=mapper('./data/source04', './data/map_data/mapper04'), args=("th4",))
    th5 = threading.Thread(target=mapper('./data/source05', './data/map_data/mapper05'), args=("th5",))
    th6 = threading.Thread(target=mapper('./data/source06', './data/map_data/mapper06'), args=("th6",))
    th7 = threading.Thread(target=mapper('./data/source07', './data/map_data/mapper07'), args=("th7",))
    th8 = threading.Thread(target=mapper('./data/source08', './data/map_data/mapper08'), args=("th8",))
    th9 = threading.Thread(target=mapper('./data/source09', './data/map_data/mapper09'), args=("th9",))

    print("==========map time=========")
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





