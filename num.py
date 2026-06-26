import numpy as np

#creating array
array_1d= np.array([1,2,3,4])
array_2d = np.array([[1,2,3],[4,5,6]])#here the first bar bracket []is array and inside that anything can be there that will be a array part
print(f"1d array: {array_1d}")
print(f"2d array: {array_2d}")

#list vs numpy array
py_list=[1,2,3]*2
np_array= np.array([1,2,3])*2
print("multiplied list", py_list)#prints the list 2 times 
print("multiplied array", np_array)#multiply the all the elements in the array by 2

#creating array from scratch
array_0=np.zeros((3,4))#rows, columns
array_1=np.ones((3,4))
full= np.full((3,4),7)#row,column and ur choice number
random=np.random.random((3,4))
sequence= np.arange(0,10,2)#start end(exclusive), step
print("0s array", array_0)
print("1s array", array_1)
print("any number array", full)
print("random array", random)
print("squence array", sequence)

#vector matrice and tensor
vector= np.array([1,2,3])
matrice= np.array([[1,2,3],
                  [1,2,3]])
tensor= np.array([[[1,2,3],[4,5,6]],
                  [[1,2,3],[4,5,6]]
                  ])#grp of matrice
print("vector", vector)
print("matice", matrice)
print("tesnors", tensor)

#array properties
array= np.array([[1,2,3],[1,2,3]])
shape=array.shape
dim= array.ndim
type=array.dtype
size= array.size
print("shape", shape)
print("dimesnions", dim)
print("data type", type)
print("no of elements", size)

#array reshaping
arr= np.arange(0,12)
arr_reshaped=arr.reshape((3,4))
print("reshaped array", arr_reshaped)
flatten= arr_reshaped.flatten()
print("falttened reshaped array", flatten)

#ravel(retuns the original insted of the copy array)
raveled= arr_reshaped.ravel()
print("ravelled array", raveled)

#transpose(column, row convertion)
transpose= arr_reshaped.T
print("tranposed array", transpose)

#array operations
print("basic slicing", arr[2:7])#from postion index to 7
print("with step slicing",  arr[2:7:2])
print("with ngative indexing",arr[-7:-2])

arr_2d=arr.reshape(3,4)
print("for 2d array", arr_2d[1,2])#1,2 is the cord say index 1 row 2nd index element
print("entire row in 2d", arr_2d[1,:])
print("entire column in 2d", arr_2d[:,1])

#sorting
unsorted= np.array([3,4,2,6,5,1,9,5])
print("sorted", np.sort(unsorted))
arr_2d=([[3,2],[7,11],[5,4]])
print("sorted", np.sort(arr_2d, axis=1))#1 for row vise and 0 for column

#filtering
arr= np.arange(1,10)
even_number=arr[arr%2==0]
print("even number",even_number)

#filter with mask
mask=arr >5 
print("numbers greater than 5", arr[mask])

#fancy indexing vs np.where()
indices=[0,2,4]#0 idex, 2 index, 4 index
print("on using indecies",arr[indices])
where_result= np.where(arr>5)
print("on using where",  arr[where_result])

#conditional array
conditional_arr=np.where(arr>5)
print("for condition >5",conditional_arr)
conditional_arr=np.where(arr>5, arr*3, arr)#condtion, task, if its less than 5 then print the elemnts as it is in arr
print("for conditon if greater than 5 multiply by 3",conditional_arr)

#adding / removing data
arr1=np.array([1,2,3])
arr2=np.array([4,5,6])
print("combined array", np.concatenate((arr1, arr2)))
a=np.array([[1,2,3],[4,5,6]])
b=np.array([[7,8,9]])
c= np.vstack((a,b))
print("from vstack", c)
b=np.array([[8],[7]])
c= np.hstack((a,b))
print("from hstack", c)

#array compatibility
a=np.array([1,2,3])
b=np.array([4,5,6])
c=np.array([7,8,9])
print("checking shape similarity", a.shape==b.shape)

#delete operation
arr=np.array([1,2,3,4,5])
delete= np.delete(arr, 3)#index is 3
print("after deletion array", delete)