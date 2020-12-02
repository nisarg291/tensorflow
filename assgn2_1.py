import re
#for actual data
file=open('actual.txt')
ans=[]
total=0
for line in file.readlines():
    lst=re.findall('[0-9]+',str(line))
    print(lst)
    number=""
    for i in lst:
        total+=int(i)
        
print(total)
# #for actual data sum of ends with 389
# file1=open('actual.txt')
# sum2=0
# for line in file1.readlines():
#     lst=re.findall('$389',str(line))
#     for i in lst:
#         sum2=sum2+int(i)

# print(sum2)
