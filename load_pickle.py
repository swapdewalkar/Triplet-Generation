import functions as F
import sys
# python load_pickle.py folder name
data=F.load_to_file(sys.argv[2],sys.argv[1])
print(len(data))
for k in data:
#    print(str(k)+"\t"+str(data[k]))
     print(k)
