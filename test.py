import time as tt
import datetime

count =1
total = 60
while True:

    oldtime = datetime.datetime.now().time()

    tt.sleep(1)

    time = datetime.datetime.now().time()

    e1 = int(time.second)
    e2 = int(oldtime.second)
    timeDif = abs(e1-e2)
    totalLeft = abs(count-total)
    estimate = timeDif*totalLeft
    minutes = estimate/60
    if(minutes < 1):
        print(estimate)
    else:
        print(minutes)
    count+=1
