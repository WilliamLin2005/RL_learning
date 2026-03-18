import time

numbers=range(10000000)

list_num=list(numbers)
start_time=time.time()
print(51234221 in list_num)
end_time=time.time()
print(f"list take time: {end_time-start_time}")

set_num=list(numbers)
start_time_1=time.time()
print(51234221 in list_num)
end_time_1=time.time()
print(f"set take time: {end_time_1-start_time_1}")