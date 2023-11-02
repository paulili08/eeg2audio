a = int(input())
a = set()
b = set()
num = input().split(' ')
for i in num:
    if i not in a:
        a.add(i)
    else:
        b.add(i)
c = a-b
result = c.pop()
print(result)