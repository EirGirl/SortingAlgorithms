# by Dr Aksana Chyzheuskaya

#•	Bubble Sort
#•	Quicksort
#•	Bucket sort
#•	Heapsort
#•	Introsort

from random import sample
from timeit import repeat
from random import randint
import time
from time import time
from random import sample
import numpy as np
import random
from sklearn.utils import shuffle
import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.max_rows
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def main():
    algs=["bubble_sort", "quick_sort", "bucket_sort", "heap_sort", "intro_sort"] # an array of row indices for a df
    #algs=["quick_sort", "bucket_sort", "heap_sort", "intro_sort"] # for running and plotting results sans bubble_sort
    nnn = [100, 250, 500, 750, 1000, 1250, 2500, 3750, 5000, 6250, 7500, 8750, 10000] # cols of df
    benchmark(bubble_sort) # call algoriths one after another
    benchmark(quick_sort)
    benchmark(bucket_sort)
    benchmark(heap_sort)
    benchmark(intro_sort)
    df = pd.DataFrame(data=resultsALL, index=algs, columns=nnn)  # results to dataframe
    df2=df.T # transpose dataframe for plotting
    df2.plot(kind='line') # plot results
    plt.xlabel("Array size")
    plt.ylabel("Running Time, milliseconds")
    plt.show()
    print(df) # print dataframe to present results
global resultsALL   
resultsALL=[]

# Adapted from https://gist.github.com/eoconnell/3328430
def benchmark(func):
    
    nnn = [100, 250, 500, 750, 1000, 1250, 2500, 3750, 5000, 6250, 7500, 8750, 10000] # array of N of elements
    results=[]                                   # create an empty array to recprd results to

    for i in nnn:
        unsorted = sample(range(0, i), i)        #generate an array to sort and suffle it
        #unsorted = np.ones((i,), dtype=int)      # best case scenario array
        #unsorted3=np.sort(unsorted)[::-1]        # worst case scenario array
        al=[]                                    # create an empty array to record sorting times to
        print("")
        print("Is al empty? :", al)
        print("I am sorting: ", i)
        for j in range(10):                      # repeat sorting 10 times to take an average
            unsorted2=shuffle(unsorted)          # shuffle an array for sorting every time
 
            s_time = time()                    # Set start time for quick sort testing
            func(unsorted2)
            #func(unsorted3)                    # for worst case, sort through array 3
            f_time = time()                    # Set end time for quick sort testing
            
            print("I am executing: ", (func.__name__, j))
            elapsed = f_time - s_time         # Calculate time taken to run quick sort function
            elaps=round(elapsed*1000, 3)      # convert seconds into milliseconds and round to 3 decimal places
            al.append(elaps)                  # add results to the array
            print("al:  ", al)

        av_time=np.average(al)                    # take an average of 10 runs
        results.append(av_time)                   # add average time to an array
 
    resultsALL.append(results)                    # add a resulting array for a sorting algorythm to an array of results
    #print(resultsALL)
        
    return print(resultsALL)


# Adapted from: https://realpython.com/sorting-algorithms-python/		
def bubble_sort(array):
    n = len(array)

    for i in range(n):
        # Look at each item of the list one by one,compare with the next value.
        # With each iteration, array to sort gets smaleer
        for j in range(n - i - 1):
            # Terminate if sorted
            already_sorted = True

            if array[j] > array[j + 1]:      # If the item is greater than the next -swap
                array[j], array[j + 1] = array[j + 1], array[j]
                
                already_sorted = False # chnage already sorted to False

        if already_sorted: # break when sorted
            break

# Adapted from: https://realpython.com/sorting-algorithms-python/
def quick_sort(array):

    if len(array) < 2: # if the input array contains fewer than two elements,
        return array   # return the array

    low, same, high = [], [], [] # create 3 empty arrays

    pivot = array[randint(0, len(array) - 1)] # randomely select pivot

    for item in array:

        if item < pivot:       # if an element < pivot
            low.append(item)   # append to "low" array
        elif item == pivot:    # if an element = pivot
            same.append(item)  # append to "same" array
        elif item > pivot:     # if an element > pivot
            high.append(item)  # append to "high array

    return quick_sort(low) + same + quick_sort(high) # combine results
    return array
    
    
# Adapted from: https://www.sanfoundry.com/python-program-implement-bucket-sort/
 
def bucket_sort(array):
    largest = max(array) # largest element
    length = len(array)  # length of an array
    size = largest/length
 
    buckets = [[] for _ in range(length)]  # create empty buckets as a list of empty lists
    for i in range(length): # determine what bucket array(i) belongs to
                            # an append value to the bucket
        j = int(array[i]/size)
        if j != length:
            buckets[j].append(array[i])
        else:
            buckets[length - 1].append(array[i])
 
    for i in range(length):
        insertion_sort(buckets[i]) # perfotm insertion sort on each bucket
 
    result = []
    for i in range(length):
        result = result + buckets[i] # concatenate all buckets together
 
    return result
 
def insertion_sort(array):
    for i in range(1, len(array)): # loop over the elements of the list
        temp = array[i]  # creat temp var equal to an element at index i
        j = i - 1        # set a counter 
        while (j >= 0 and temp < array[j]): # if j is not negative and 
                                            # an element at index i < than preceeding element
            array[j + 1] = array[j] # set the element at index j + 1 equal to the element at index j 
            j = j - 1               # increment counter
        array[j + 1] = temp         # set the element at index j + 1 equal to temp
 
 
# Adapted from: https://www.tutorialspoint.com/python-program-for-heap-sort
def heapify(arr, n, i):
   largest = i # largest value
   l = 2 * i + 1 # left
   r = 2 * i + 2 # right  
   if l < n and arr[i] < arr[l]: # if left child exists
      largest = l 
   if r < n and arr[largest] < arr[r]: # if right child exits
      largest = r
   # root
   if largest != i:
      arr[i],arr[largest] = arr[largest],arr[i] # swap
      # root.
      heapify(arr, n, largest)
# sort
def heap_sort(arr):
   n = len(arr)
   # maxheap
   for i in range(n, -1, -1):
      heapify(arr, n, i)
   # element extraction
   for i in range(n-1, 0, -1):
      arr[i], arr[0] = arr[0], arr[i] # swap
      heapify(arr, i, 0)

# Adapted from: https://www.sanfoundry.com/python-program-implement-introsort/
def intro_sort(alist):
    maxdepth = (len(alist).bit_length() - 1)*2 #maxdepth = to 2 times floor of log base 2 of the length of the list.
    introsort_helper(alist, 0, len(alist), maxdepth) # call introsort_helper 
 
def introsort_helper(alist, start, end, maxdepth):
    if end - start <= 1: # if the length of the list to be sorted is not greater than 1
        return           # return
    elif maxdepth == 0:  # if maxdepth is 0
        heapsort(alist, start, end) # perform a heapsort
    else:
        p = partition(alist, start, end) # therwise, call partition 
        introsort_helper(alist, start, p + 1, maxdepth - 1) # call introsort_helper on 2 halfs of the list
        introsort_helper(alist, p + 1, end, maxdepth - 1)
 
def partition(alist, start, end):
    pivot = alist[start]
    i = start - 1
    j = end
 
    while True:
        i = i + 1
        while alist[i] < pivot:
            i = i + 1
        j = j - 1
        while alist[j] > pivot:
            j = j - 1
 
        if i >= j:
            return j
 
        swap(alist, i, j)
 
def swap(alist, i, j):
    alist[i], alist[j] = alist[j], alist[i]
 
def heapsort(alist, start, end):
    build_max_heap(alist, start, end)
    for i in range(end - 1, start, -1):
        swap(alist, start, i)
        max_heapify(alist, index=0, start=start, end=i)
 
def build_max_heap(alist, start, end):
    def parent(i):
        return (i - 1)//2
    length = end - start
    index = parent(length - 1)
    while index >= 0:
        max_heapify(alist, index, start, end)
        index = index - 1
 
def max_heapify(alist, index, start, end):
    def left(i):
        return 2*i + 1
    def right(i):
        return 2*i + 2
 
    size = end - start
    l = left(index)
    r = right(index)
    if (l < size and alist[start + l] > alist[start + index]):
        largest = l
    else:
        largest = index
    if (r < size and alist[start + r] > alist[start + largest]):
        largest = r
    if largest != index:
        swap(alist, start + largest, start + index)
        max_heapify(alist, largest, start, end)
 

 
if __name__ == "__main__":
    main()	




    

  


	