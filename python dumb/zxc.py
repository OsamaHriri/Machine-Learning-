import numpy as np

# summationl = summationr = 0  #variable to store the summation of differences
# n = len(num_right) #finding total number of items in list
# sum_left=sum_right=0
# for i in range(1, n):
#     sum_left += num_left[i] * i
#     sum_right += num_right[i] * i
# print(sum_left)
# print(num_left)
# mean_left = sum_left / sum(num_left)
# mean_right = sum_right / sum(num_right)
# print(mean_left)
#
# for i in range (1,n):  #looping through each element of the list
#
#     differencel = (i- mean_left)*num_left[i]  #finding the difference between observed and predicted value
#     squared_differencel = differencel**2  #taking square of the differene
#     summationl = summationl + squared_differencel  #taking a sum of all the differences
#     differencer = (i- mean_right)*num_right[i]  #finding the difference between observed and predicted value
#     squared_differencer = differencer**2  #taking square of the differene
#     summationr = summationr + squared_differencer  #taking a sum of all the differences
# MSEr = summationr/sum(num_left)  #dividing summation by total values to obtain average
# MSEl = summationl/sum(num_right)
# mse = (MSEl*sum(num_left)+MSEr*sum(num_right))/(sum(num_left) + sum(num_right) )
# print(mse)
#
#
#
#
a= np.array([1,2,3,4,5,6])
print(np.mean(a))
print(np.mean((a-np.mean(a))**2))
