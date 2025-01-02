def quicksort(arr,left,right):
    if left<right:
        pos = partition(arr,left,right)
        quicksort(arr,left,pos-1)
        quicksort(arr,pos+1,right)
        
def partition(arr, left, right):
    i=left
    j=right-1
    pivot=arr[right]
    
    while i<j:
        while i<right and arr[i]<pivot:
            i=i+1
            
        while j>left and arr[j]>=pivot:
            j=j-1
            
        if i<j:
            arr[i], arr[j] = arr[j], arr[i]
            
    if arr[i]>pivot:
        arr[i], arr[right]= arr[right], arr[i]
        
    return i
    
n=int(input())
arr=[]
for i in range(n):
    arr.append(int(input()))
quicksort(arr, 0, n-1)
print(arr)