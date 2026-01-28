import csv
from collections import defaultdict

import numpy as np
import os


def read_file(file):

    fields = [
        "sentiment",
        "productId",
        "userId",
        "summary",
        "text",
        "helpfulY",
        "helpfulN",
    ]
    data = defaultdict(list)
    d = []
    with open(file, "r", encoding="latin1") as f:
        reader = csv.DictReader(f, delimiter="\t")
        # for key,reader:
        #     #print([x[y] for x,y in zip(row,fields)])
        #     for field in fields:
        #         data[field].append(row[field])
        #     #d.append({x:[row[fields[i]] for i in range(len(fields))] })
        # for row in reader:
        #     d.append([row[fields[i]] for i in range(len(fields))])
        for row in reader:
            for i in range(len(fields)):
                data[fields[i]].append(row[fields[i]])

    # print(data)
    return data


def main():
    file_data = list(list())
    for file in os.listdir(""):
        # print(file,file[-3:])
        if file[-3:] == "tsv":
            print(file)
            data = read_file(file)
            break
    for key, value in data.items():
        print(key, value)


if __name__ == "__main__":
    main()
    # data = np.genfromtxt('4000.txt', dtype=str, encoding=None, delimiter=",")
    # print(data)


#
# #importing numpy library
# import numpy as np
#
# #driver code
# if __name__ == "__main__":
#   data = np.genfromtxt("gfg.txt", dtype=str, encoding = None, delimiter=",")
#   #displaying the data
#   for i in data:
#     print(i,end=" ")
