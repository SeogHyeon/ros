__author__='SeogHyeon'

'''
<manual>
1. 사용할 varialbe 개수와 expression 개수를 입력
2. variable 끼리는 <space>로, expression끼리는 <enter>로 구분하여 입력
3. 약 4초간 기다리기

<input>
1. number of variable
2. number of expression
3. each minterm expression(in alphabet) (ex. a' b' c d)

<output>
minimal expression(각 expression은 <enter>로 구분됨)
'''

import time
import random

num_v=int(input('variable 개수를 입력하세요(3~4):'))    #variable 개수
num_e=int(input('expression 개수를 입력하세요:'))       #expression 개수

def possible(arr_a,arr_b):      ##  합치기 가능 여부 return
    if len(arr_a)==len(arr_b):
        c=set(arr_a[:-1]).intersection(arr_b[:-1])      ##  두개의 교집합
        if len(c)==len(arr_a)-2:   ##  원소 중 하나만 다르면!
            dif_a=list(set(arr_a[:-1])-c)
            dif_b=list(set(arr_b[:-1])-c)       ##  각 집합에서 교집합 뺀 것
            if dif_a[0]+"'"==dif_b[0] or dif_b[0]+"'"==dif_a[0]:        ##  "'"붙은 거랑 같으면(a'와 a 같이 합쳐질 수 있을 때)
                return True
    return False

exp_list=[]     ##  expression 저장 행렬(띄어쓰기 기준으로 나누기)
for i in range(num_e):
    arr=input().split()
    arr.sort()          ##  정렬 - commutative laws
    arr.append(False)   ##  초기 사용여부 False로 저장
    exp_list.append(arr)

start=time.time()
while time.time()-start<2:  ##  2초만 계산!
    for a in exp_list:
        for b in exp_list:
            if possible(a,b):  ##  합치기 가능하다면:
                temp=list(set(a[:-1]).intersection(b[:-1])) ##  temp에 공통부분 저장
                temp.sort()
                temp=temp+[False]
                if temp not in exp_list:    ##  저장이 안돼있으면
                    exp_list.append(temp)   ##  a와b의 공통부분(TrueFalse제외) + False 묶어서 exp_list 맨 밑에 append
                a[-1]=True
                b[-1]=True  ##  사용했으므로 True!

result=[]
for i in exp_list:
    if i[-1]==False:    ##  합쳐지지 못한 것들(마지막 False)만 결과에 append
        result.append(i[:-1])
# print('consensus 전:',result)

######################## consensus theorem part #################################
start=time.time()
while time.time()-start<2:  ##  2초만 계산!
    try:
        exp=random.sample(result,3)             ##  3개 임의로 추출
        if 'a' in exp[0] and "a'" in exp[1]:    ##  첫 번째에 a, 두 번째에 a'가 존재한다면
            temp1=set(exp[0])-set(['a'])        ##  첫 번째에서 a뺀 것
            temp2=set(exp[1])-set(["a'"])       ##  첫 번째에서 a'뺀 것
            temp_sum=list(temp1.union(temp2))   ##  두개의 합집합
            temp_sum.sort()                     ##  정렬
            if temp_sum==exp[2]:                ##  두 개의 합집합이 남은 하나가 같을 경우에는 남은 하나 제거 가능
                try:
                    result.remove(exp[2]) 
                except:
                    pass
        if 'b' in exp[0] and "b'" in exp[1]:    ##  아래 b와 c에 대해서도 똑같이 적용
            temp1=set(exp[0])-set(['b'])
            temp2=set(exp[1])-set(["b'"])
            temp_sum=list(temp1.union(temp2))
            temp_sum.sort()
            if temp_sum==exp[2]:
                try:
                    result.remove(exp[2]) 
                except:
                    pass
        if 'c' in exp[0] and "c'" in exp[1]:
            temp1=set(exp[0])-set(['c'])
            temp2=set(exp[1])-set(["c'"])
            temp_sum=list(temp1.union(temp2))
            temp_sum.sort()
            if temp_sum==exp[2]:
                try:
                    result.remove(exp[2]) 
                except:
                    pass
    except:
        pass

#######################printing part#################################
print('result:')
for i in result:
    for j in i:
        print(j.ljust(3),end='')
    print()