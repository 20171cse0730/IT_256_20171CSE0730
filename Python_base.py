# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 12:38:25 2017

@author: virat
"""
############################## Spyder 3 Installation ############################

###Python 3.6

#I. Installing on Windows (Vista/7/8/10)

#Spyder is already included in the Python Scientific Distributions:

#1. Anaconda    (https://www.continuum.io/downloads)
#2. WinPython   (https://winpython.github.io/)
#3. Python(x,y) (https://winpython.github.io/)

#You can start using it immediately after installing one of them (only need to install one!).

#Alternatively, if you want to install Spyder directly (the hard way), you need to:

#1. Install the essential requirements:
#The Python Programming Language
#PyQt5(recommended) or PyQt4

#2. Install Spyder and its dependencies by running the command:
#       pip install spyder 

#Updating Spyder

#You can update Spyder by:
#Updating Anaconda, WinPython or Python(x,y)
#Or using the command:
#    pip install –upgrade spyder

#II. Installing in MacOS X
    #Easiest way is to install using Anaconda Python distribution.
    #Alternatively, spyder can downloaded and installed using DMG installers
    
#III. Installing on Linux 
    #Easiest method would be to install it using the official package manager, 
     #although this can be slightly outdated:
        # sudo apt-get install spyder
    
    # Install using the pip manager:
        # sudo pip install spyder
        # sudo pip install -U spyder


########################### Checking the Python version #######################
    
    #On MAC/Linux
        
        #Open the terminal and type:
        #    python -V
    
    #On Windows 
    
        #In the Command Prompt, type:
        #    python


################################### DATA TYPES ################################
#Python Data types: 
    #1.Numbers (integers,floats,long)
    #2.Boolean
    #3.Strings & Characters
    #4.Dates & Time

## Integer & Floating point numbers

#int (signed integers): They are often called just integers or ints,
                      # are positive or negative whole numbers with 
                      # no decimal point.

#long (long integers ): Also called longs, they are integers of unlimited size,
                      # written like integers and followed by an uppercase or lowercase L.

#float (floating point real values) : Also called floats, they represent real
                                    # numbers and are written with a decimal 
                                    # point dividing the integer and fractional
                                    # parts. Floats may also be in scientific
                                    # notation, with E or e indicating the
                                    # power of 10 (2.5e2 = 2.5 x 102 = 250).

#complex (complex numbers) : are of the form a + bJ, where a and b are floats
                           # and J (or j) represents the square root of -1
                           # (which is an imaginary number). The real part of
                           # the number is a, and the imaginary part is b.
                           # Complex numbers are not used much in Python programming

#Creating a integer
int1 = 25
type(int1)

#Creating a floating point number(float)
flt1 = 5.65
type(flt1)

#Converting int to float
int(flt1)

#Converting float to int
float(int1)

#Addition
a1 = 1+1

#Substraction
a2 = a1-2

#Multiplication
a3 = a1*2

#Division
a4 = a2/a1

#Exponentiation
a5 = 2 ** 4 #yields 2^4=16

#Remainder
a6 = 5 % 2 #yields the remainder of 5/2 which is 1

#To get information about round() function use help(round)
#Likewise, help() can be used to get information about other functions
help(round)

#round(x,n) returns x rounded to n digits
round(12.33)
round(12.75)

#Round off to 3 decimal points
round(3.897645342,3)

#Ceiling, floor & truncate
#Importing Math library to perform ceil, floor and truncate operation
import math as m

#floor(x) returns the floor of x as a float.
#This is the largest integral value <= x.
m.floor(4.55)
m.floor(-4.55)

#trunc() behaves as a ceiling function for negative number and
#floor function for positive number
m.trunc(4.55)
m.trunc(-4.55)

#ceil() Return the ceiling of x as a float.
#This is the smallest integral value >= x
m.ceil(4.55)
m.ceil(-4.55)

#Addition 
type(int1 + 30)
type(flt1 + 10)


## Logical Data Types

#Creating a boolean (logical) type
l1 = True
l2 = False
type(l1)
type(l2)

#Coverting boolean to integer & floating point number
int(l1)
float(l2)

##Built-in methods in numbers

#abs(x): The absolute value of x: the (positive) distance between x and zero.
#cmp(x, y): -1 if x < y, 0 if x == y, or 1 if x > y
#exp(x):	The exponential of x: ex
#log(x):	The natural logarithm of x, for x> 0
#log10(x): The base-10 logarithm of x for x> 0 .
#max(x1, x2,...): The largest of its arguments: the value closest to positive infinity
#min(x1, x2,...): The smallest of its arguments: the value closest to negative infinity
                # sign as x. The integer part is returned as a float.
#pow(x, y): The value of x**y.
#sqrt(x): The square root of x for x > 0


##Sets
s1 = {1,2,3,1,1,1,2,2,3,3,2}

set(s1) #Returns a list with only unique elements i.e. 
        #repeatations are removed.

## Strings

#String is basically a array of characters.
#Python does not support a character type, these are treated as strings of
#length one, thus also considered a substring

str1 = 'Hello World!'

type(str1) #Prints the data type of str1

print(str1)          # Prints complete string
print(str1[0])       # Prints first character of the string
print(str1[2:5])     # Prints characters starting from 3rd to 5th
print(str1[2:])      # Prints string starting from 3rd character
print(str1 * 2)      # Prints string two times
print(str1 + "TEST") # Prints concatenated string

#string formatting
print("My name is %s and I weigh about %d kg" % ("Virat", 56))
print("My name is {} and I weigh about {} kg".format("Virat", 56))

##String built-in methods

#capitalize(): Capitalizes first letter of string
#count(str, beg= 0,end=len(string)): Counts how many times str occurs in string
                                   # or in a substring of string if starting
                                   # index beg and ending index end are given.
#find(str, beg=0 end=len(string)): Determine if str occurs in string or in a
                                 # substring of string if starting index beg
                                 # and ending index end are given returns index
                                 # if found and -1 otherwise.                     
#index(str, beg=0, end=len(string)): Same as find(), but raises an exception
                                   # if str not found.
#isalnum(): Returns true if string has at least 1 character and all characters
          # are alphanumeric and false otherwise.
#isalpha(): Returns true if string has at least 1 character and all characters 
          # are alphabetic and false otherwise.
#isdigit(): Returns true if string contains only digits and false otherwise.
#islower(): Returns true if string has at least 1 cased character and all cased characters 
          # are in lowercase and false otherwise.
#isnumeric(): Returns true if a unicode string contains only numeric characters
            # and false otherwise.
#join(seq): Merges (concatenates) the string representations of elements in
          # sequence seq into a string, with separator string.
#len(string):Returns the length of the string
#max(str): Returns the max alphabetical character from the string str.
#min(str): Returns the min alphabetical character from the string str.
#replace(old, new [, max]): Replaces all occurrences of old in string with new
                          # or at most max occurrences if max given.
#split(str="", num=string.count(str)): Splits string according to delimiter str
                                     # (space if not provided) and returns
                                     # list of substrings; split into at most
                                     # num substrings if given.


##Dates and Time

#"time" module available in Python which provides functions for working with
#times, and for converting between representations

#Need to import time module to deal with date & time
import time as t
import datetime as dt

#Getting current time
localtime = t.localtime(t.time())
print("Local current time :", localtime)

#Getting formatted time
localtime = t.asctime( t.localtime(t.time()) )
print("Local current time :", localtime)

#Returns the time(in seconds) since epoch, i.e. since 1/01/1970
print("Time in seconds since the epoch: %s" %t.time())

#Returns current date & time
print("Current date and time: " , dt.datetime.now())

#Returns current data & time in a more readable format
print("Current date & time, in a readable format: ",dt.datetime.now().strftime("%y-%m-%d-%H-%M"))

#Returns the current year
print("Current year: ", dt.date.today().strftime("%Y"))

#Returns the current month 
print("Month of year: ", dt.date.today().strftime("%B"))

#Returns the current week number
print("Week number of the year: ", dt.date.today().strftime("%W"))

#Returns the day of the week(in number, as in 1=>Monday,2=>Tuesday...etc...)
print("Weekday of the week: ", dt.date.today().strftime("%w"))

#Returns the day of the week
print("Day of week: ", dt.date.today().strftime("%A"))

#Retuns the day of the year 
print("Day of year: ", dt.date.today().strftime("%j"))

#Returns the day of the month 
print("Day of the month : ", dt.date.today().strftime("%d"))

#Getting calender for a month
import calendar as c

cal = c.month(2008, 1)
print("Here is the calendar:")
print(cal)


#time.clock( ): Returns the current CPU time as a floating-point number of
              # seconds. To measure computational costs of different approaches,
              # the value of time.clock is more useful than that of time.time().

#time.localtime([secs]): Accepts an instant expressed in seconds since the epoch
                       # and returns a time-tuple t with the local time
                       # (t.tm_isdst is 0 or 1, depending on whether DST applies
                       # to instant secs by local rules).

#time.strftime(fmt[,tupletime]): Accepts an instant expressed as a time-tuple 
                               # in local time and returns a string representing
                               # the instant as specified by string fmt.

#time.strptime(str,fmt='%a %b %d %H:%M:%S %Y'): Parses str according to format
                                              # string fmt and returns the 
                                              # instant in time-tuple format

############################### DATA STRUCTURES ###############################
#Python Data structures:
    #1.Lists
    #2.Tuples
    #3.Dictionaries
    #4.Numpy Arrays
    #5.Panda DataFrames
    #6.Files 

##Lists

#The most basic data structure in Python is the sequence.
#Each element of a sequence is assigned a number - its position or index. 
#The first index is zero, the second index is one, and so on.

#Python has six built-in types of sequences(String, lists, tuples, buffer, 
#xrange, unicode), but the most common ones are lists and tuples.

##List Notation
#a[start:end] # items start through end-1
#a[start:]    # items start through the rest of the array
#a[:end]      # items from the beginning through end-1
#a[:]         # a copy of the whole array
#a[start:end:step] # start through not past end, by step

list1 = [ 'abcd', 786 , 2.23, 'john', 70.2 ]
tinylist = [123, 'john']
type(list1)
type(tinylist)

print(list1)            # Prints complete list
print(list1[0])         # Prints first element of the list
print(list1[1:3])       # Prints elements starting from 2nd till 3rd 
print(list1[2:])        # Prints elements starting from 3rd element
print(tinylist * 2)     # Prints list two times
print(list1 + tinylist) # Prints concatenated lists 

nest = [1,2,3,[4,5,['target']]] #List with nested lists

nest[3]          #Prints the nested list inside of nest
nest[3][2]       #Prints the list inside nested list i.e. ['Target']
nest[3][2][0]    #Prints the 0th element of the list inside of nested list

##List methods

#cmp(list1, list2): Compares elements of both lists.
#len(list): Gives the total length of the list.
#max(list): Returns item from the list with max value.
#min(list): Returns item from the list with min value.
#list(seq): Converts a tuple into list.

#list.append(obj): Appends object obj to list
#list.count(obj): Returns count of how many times obj occurs in list
#list.index(obj): Returns the lowest index in list that obj appears
#list.insert(index, obj): Inserts object obj into list at offset index
#list.pop(obj=list[-1]): Removes and returns last object or obj from list
#list.remove(obj): Removes object obj from list
#list.reverse(): Reverses objects of list in place
#list.sort([func]): Sorts objects of list, use compare func if given
    
    
## Tuples

#A tuple is a sequence of immutable Python objects. Tuples are sequences, 
#just like lists. The differences between tuples and lists are, 
#the tuples cannot be changed unlike lists and tuples use parentheses, 
#whereas lists use square brackets.

tuple1 = ( 'abcd', 786 , 2.23, 'john', 70.2  )
tinytuple = (123, 'john')
type(tuple1)
type(tinytuple)

print(tuple1)             # Prints complete list
print(tuple1[0])          # Prints first element of the list
print(tuple1[1:3])        # Prints elements starting from 2nd till 3rd 
print(tuple1[2:])         # Prints elements starting from 3rd element
print(tinytuple * 2)      # Prints list two times
print(tuple1 + tinytuple) # Prints concatenated lists

##Tuple methods

#cmp(tuple1, tuple2): Compares elements of both tuples.
#len(tuple): Gives the total length of the tuple.
#max(tuple): Returns item from the tuple with max value.
#min(tuple): Returns item from the tuple with min value.
#tuple(seq): Converts a list into tuple.


## Dictionaries

#Dictionaries are basically a data structure that consists of key-value pairs.

#Keys are unique within a dictionary while values may not be. The values of a 
#dictionary can be of any type,but the keys must be of an immutable data type
#such as strings, numbers, or tuples.

dict = {}
dict['one'] = "This is one"
dict[2]     = "This is two"
type(dict)

tinydict = {'name': 'john','code':6734, 'dept': 'sales'}

print(dict['one'])       # Prints value for key 'one'
print(dict[2])           # Prints value for 2 key
print(tinydict)          # Prints complete dictionary
print(tinydict.keys())   # Prints all the keys
print(tinydict.values()) # Prints all the values

##Dictionary methods

#cmp(dict1, dict2): Compares elements of both dict.
#len(dict): Gives the total length of the dictionary. This would be equal to 
          # the number of items in the dictionary.
#str(dict): Produces a printable string representation of a dictionary

#dict.clear(): Removes all elements of dictionary dict	
#dict.copy(): Returns a shallow copy of dictionary dict
#dict.fromkeys(): Create a new dictionary with keys from seq and values set to value.
#dict.get(key, default=None): For key key, returns value or default if key not
                            # in dictionary
#dict.has_key(key): Returns true if key in dictionary dict, false otherwise
#dict.items(): Returns a list of dict's (key, value) tuple pairs
#dict.keys(): Returns list of dictionary dict's keys
#dict.setdefault(key, default=None): Similar to get(), but will set dict[key]=default
                                   # if key is not already in dict
#dict.update(dict2): Adds dictionary dict2's key-values pairs to dict
#dict.values(): Returns list of dictionary dict's values


##Numpy arrays

#Import the numpy module
import numpy as np

np.arange(0,10)   #Returns evenly spaced values within a given interval.
np.arange(0,10,2) #Returns evenly spaced values within a given interval,
                  #in steps of 2.

np.zeros(3)   #Returns a 1-D array of zeros
np.zeros(5,5) #Returns a 5x5 array of zeros

np.ones(3)   #Returns a 1-D array of ones
np.ones(5,5) #Returns a 5x5 array of ones

#linspace returns evenly spaced numbers over a specified interval.
np.linspace(0,10,3)  #Returns 3 numbers between 0 & 10
np.linspace(0,10,50) #Returns 50 numbers between 0 & 10

np.eye(4) #Returns an identity matrix

#rand() Create an array of the given shape and populate it with
#random samples from a uniform distribution over [0, 1).
np.random.rand(2)   #Creates a 1-d array of random values between 0,1
np.random.rand(5,5) #Creates a 5x5 array of random values between 0,1

#randn() Return a sample (or samples) from the "standard normal" distribution
#Unlike rand which is uniform    
np.random.randn(2)
np.random.randn(5,5)

#randint() Returns random integers from low (inclusive) to high (exclusive).
np.random.randint(1,100)    #Generates only 1 integer in the given range
np.random.randint(1,100,10) #Generates 10 integers in the given range

#Array attributes
arr = np.arange(25)
arr.dtype #Returns the data type of the object in the array.

arr.shape  #Returns the shape of the array
           #NOTE- Shape is an attribute that arrays have(not a method)
           
arr.reshape(5,5) #Returns the array containing the same data with a new shape(5x5)

#Creating sample array
arr1 = np.arange(0,11)

#Show the array
arr1

#Get a value at an index
arr1[8]

#Get a value in range
arr1[0:5]

#Numpy arrays differ from a normal Python list because of their ability to broadcast:
#Setting a value with index range (Broadcasting)
arr1[0:5]=100

#Show
arr1

# Reset array, we'll see why I had to reset in  a moment
arr1 = np.arange(0,11)

#Show
arr1

#Important notes on Slices
slice_of_arr1 = arr[0:6]

#Show slice
slice_of_arr1
    
#Change Slice
slice_of_arr1[:]=99

#Show Slice again
slice_of_arr1

#Now note the changes also occur in our original array!
arr1

#Data is not copied, it's a view of the original array! This avoids memory problems!
#To get a copy, need to be explicit
arr1_copy = arr.copy()

arr1_copy

#For indexing 2D array, The general format is arr_2d[row][col] or arr_2d[row,col]
arr_2d = np.array(([5,10,15],[20,25,30],[35,40,45]))

#Show the 2D array
arr_2d

#Indexing row
arr_2d[1]

# Format is arr_2d[row][col] or arr_2d[row,col]
# Getting individual element value
arr_2d[1][0]

# Getting individual element value
arr_2d[1,0]

# 2D array slicing
#Shape (2,2) from top right corner
arr_2d[:2,1:]

#Shape bottom row
arr_2d[2]
arr_2d[2,:]

#Set up matrix
arr2d = np.zeros((10,10))

#Length of array
arr_length = arr2d.shape[1]

#Set up array
for i in range(arr_length):
    arr2d[i] = i
    
arr2d

#Fancy indexing allows the following:
arr2d[[2,4,6,8]]

#Allows in any order
arr2d[[6,4,2,7]]

#Selection 
arr2 = np.arange(1,11)

arr2 > 4

bool_arr = arr2 > 4
bool_arr

arr2[bool_arr]

arr2[arr2>4]
    
arr3 = np.arange(0,10)

#Array Arithmatic
arr3 + arr3
arr3 * arr3
arr3 - arr3

# Warning on division by zero, but not an error!
# Just replaced with nan
arr3/arr3

#Taking Square Roots
np.sqrt(arr3)

#Calcualting exponential (e^)
np.exp(arr3)

np.max(arr3) #same as arr.max()


## Pandas DataTypes

#importing pandas
import pandas as pd

##Series

#A Series is very similar to a NumPy array (in fact it is built on top of the 
#NumPy array object). What differentiates the NumPy array from a Series, is that
#a Series can have axis labels, meaning it can be indexed by a label, instead of
#just a number location.It also doesn't need to hold numeric data, it can hold 
#any arbitrary Python Object

#We can convert a list,numpy array, or dictionary to a Series

labels = ['a','b','c']
my_list = [10,20,30]
arr = np.array([10,20,30])
d = {'a':10,'b':20,'c':30}

#Using Lists
pd.Series(data=my_list)
pd.Series(data=my_list,index=labels)
pd.Series(my_list,labels)

#Numpy Arrays
pd.Series(arr)
pd.Series(arr, labels)

#Dictionary
pd.Series(d)

#Panda Series can hold a variety of object types
pd.Series(data=labels)

#Even functions (although unlikely that you will use this)
pd.Series([sum,print,len])

#The key to using a Series is understanding its index. Pandas makes use of these 
#index names or numbers by allowing for fast look ups of information (works 
#like a hash table or dictionary).

ser1 = pd.Series([1,2,3,4],index = ['USA', 'Germany','USSR', 'Japan']) 
ser1

ser2 = pd.Series([1,2,5,4],index = ['USA', 'Germany','Italy', 'Japan'])
ser2                                                                       
                                                                      
ser1['USA']

#Operations are then also done based off of index
ser1 + ser2

#Defining a pandas DataFrame
df = pd.DataFrame(np.random.randn(5,4),index='A B C D E'.split(),
                  columns='W X Y Z'.split())
df #Display the DataFrame

df['W']       #Returns the 'W' Column
df['W','Z']   #Returns the 'W','Z' column
type(df['W']) #Returns the datatype

df['new']=df['W']+df['Y'] #Creates a new column named 'New' 
df.drop('new',axis=1) #Drops column 'new'. Axis=0 refers to Rows and Axis=1 refers
                      #to Columns
df #Displays the DataFrame

df.loc['B','Y'] #Returns Bth row and Yth column
df.loc[['A','B'],['W','Y']]  #Returns Ath,Bth row and Wth,Yth column

print(df[df>0]) #Applying conditional selection on df

#More on condtional selection
print(df[df['W']>0])
print(df[df['W']>0]['Y'])
print(df[df['W']>0][['Y','X']])
print(df[(df['W']>0) & (df['Y'] > 1)])
              
#Display the dataframe
df

# Reset to default 0,1...n index
df.reset_index()

newind = 'CA NY WY OR CO'.split()
df['States'] = newind

#Display the df
df

#Adding new column "States"
df.set_index('States')

#Display the dataframe
df

##Multi Index & Index hierarchy

# Index Levels
outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]

#
hier_index = list(zip(outside,inside))

#Converting list of tuples to multi-index
hier_index = pd.MultiIndex.from_tuples(hier_index)

hier_index

df = pd.DataFrame(np.random.randn(6,2), index= hier_index, columns=['A','B'])
df

#For index hierarchy we use df.loc[], if this was on the columns axis, 
#you would just use normal bracket notation df[]. Calling one level of the index
#returns the sub-dataframe
df.loc['G1']

df.loc['G1'].loc[1]

#Check out the index names
df.index.names 

#Rename the indexes as "Group" & "Num"
df.index.names = ['Group','Num']

#Display the updated DataFrame
df

#Display the contents of Group G1
df.xs('G1')

#Display contents of 1st row of G1
df.xs(['G1',1])

#Display contents of 1st column of Group G1,G2
df.xs(1, level='Num')

##Filling missing data
df1 = pd.DataFrame({'A':[1,2,np.nan],
                  'B':[5,np.nan,np.nan],
                  'C':[1,2,3]})
#Display df1    
df1

#Displays the data without na
df1.dropna()    

df1.dropna(axis=1)

df1.dropna(thresh=2)

df1.fillna(value='FILL VALUE')

df['A'].fillna(value=df['A'].mean())

##Groupby

#The groupby method allows you to group rows of data together and call aggregate
#functions

data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
       'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
       'Sales':[200,120,340,124,243,350]}

df2 = pd.DataFrame(data)
df2

#Now you can use the .groupby() method to group rows together based off of a column
#name. For instance let's group based off of Company. This will create a
#DataFrameGroupBy object

df2.groupby['Company']

#We can save this object as a new variable
by_comp = df.groupby("Company")

#call aggregate methods off the object
by_comp.mean()

df.groupby('Company').mean()

#More examples on aggregate method
by_comp.std()
by_comp.min()
by_comp.max()
by_comp.count()
by_comp.describe()
by_comp.describe().transpose()
by_comp.describe().transpose()['GOOG']

##Panda Operations
df3 = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],
                    'col3':['abc','def','ghi','xyz']})
df3.head()

#Information on Unique values
df3['col2'].unique()
df3['col2'].nunique()
df3['col2'].value_counts()

#Select from DataFrame using criteria from multiple columns
newdf = df[(df['col1']>2) & (df['col2']==444)]
newdf

def times2(x):
    return x*2

df3['col1'].apply(times2)

df3['col3'].apply(len)

df3['col1'].sum()

#removing a column
del df3['col1']
df3

#Get columns & index names
df3.columns
df3.index

df3

#Sorting & Ordering DataFrames
df3.sort_values(by='col2') #inplace=False by default

#Find Null Values or Check for Null Values
df3.isnull()

# Drop rows with NaN Values
df3.dropna()

#Filling in NaN values with something else
df4 =  pd.DataFrame({'col1':[1,2,3,np.nan],
                   'col2':[np.nan,555,666,444],
                   'col3':['abc','def','ghi','xyz']})
df4.head()

df4.fillna('FILL')

data = {'A':['foo','foo','foo','bar','bar','bar'],
     'B':['one','one','two','two','one','one'],
       'C':['x','y','x','y','x','y'],
       'D':[1,3,2,5,4,1]}

df4 = pd.DataFrame(data)
df4

df4.pivot_table(values='D',index=['A', 'B'],columns=['C'])

##Merging, Joining, and Concatenating

#There are 3 main ways of combining DataFrames together: Merging, Joining and 
#Concatenating.

df5 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']},
                        index=[0, 1, 2, 3])

df6 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                        'B': ['B4', 'B5', 'B6', 'B7'],
                        'C': ['C4', 'C5', 'C6', 'C7'],
                        'D': ['D4', 'D5', 'D6', 'D7']},
                         index=[4, 5, 6, 7]) 

df7 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                        'B': ['B8', 'B9', 'B10', 'B11'],
                        'C': ['C8', 'C9', 'C10', 'C11'],
                        'D': ['D8', 'D9', 'D10', 'D11']},
                        index=[8, 9, 10, 11])

df5
df6
df7

#Concatenation basically glues together DataFrames. Keep in mind that dimensions
#should match along the axis you are concatenating on. You can use pd.concat and 
#pass in a list of DataFrames to concatenate together

pd.concat([df5,df6,df7])

pd.concat([df5,df6,df7],axis=1)

left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
   
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                          'C': ['C0', 'C1', 'C2', 'C3'],
                          'D': ['D0', 'D1', 'D2', 'D3']}) 
left
right

#The merge function allows you to merge DataFrames together using a similar
#logic as merging SQL Tables together.

pd.merge(left,right,how='inner',on='key')

#A more complicated example

left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                        'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3']})
    
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                               'key2': ['K0', 'K0', 'K0', 'K0'],
                                  'C': ['C0', 'C1', 'C2', 'C3'],
                                  'D': ['D0', 'D1', 'D2', 'D3']})

pd.merge(left, right, on=['key1', 'key2'])

pd.merge(left, right, how='outer', on=['key1', 'key2'])

pd.merge(left, right, how='right', on=['key1', 'key2'])

pd.merge(left, right, how='left', on=['key1', 'key2'])

#Joining is a convenient method for combining the columns of two potentially 
#differently-indexed DataFrames into a single result DataFrame

left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                      index=['K0', 'K1', 'K2']) 

right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                    'D': ['D0', 'D2', 'D3']},
                      index=['K0', 'K2', 'K3'])

left.join(right)

left.join(right, how='outer')

##Data Input & Output

##CSV 
df8 = pd.read_csv('myFile')
df8

df8.to_csv('myFile', index=False)

##Excel

pd.read_excel('myExcelFile.xlsx',sheetname='Sheet1')

df.to_excel('myExcelFile.xlsx',sheet_name='Sheet1')

##HTML

#You may need to install htmllib5,lxml, and BeautifulSoup4.

#Pandas read_html function will read tables off of a webpage and return a list 
#of DataFrame objects:

df = pd.read_html('http://www.fdic.gov/bank/individual/failed/banklist.html')
df[0]

##SQL

#The pandas.io.sql module provides a collection of query wrappers to both facilitate
#data retrieval and to reduce dependency on DB-specific API. Database abstraction
#is provided by SQLAlchemy if installed. In addition you will need a driver library
#for your database. Examples of such drivers are psycopg2 for PostgreSQL or pymysql
#for MySQL. For SQLite this is included in Python’s standard library by default.
#We can find an overview of supported drivers for each SQL dialect in the SQLAlchemy
#docs.

##The key functions are:
    
#read_sql_table(table_name, con[, schema, ...])
#Read SQL database table into a DataFrame.
#read_sql_query(sql, con[, index_col, ...])
#Read SQL query into a DataFrame.
#read_sql(sql, con[, index_col, ...])
#Read SQL query or database table into a DataFrame.
#DataFrame.to_sql(name, con[, flavor, ...])
#Write records stored in a DataFrame to a SQL database.



##Files Input/Output

#Taking input from the user and displaying it
#NOTE- raw_input() has been renamed to input()       
str = input("Enter your input: ")
print("Received input is : ", str)

#Creating a text file and writing to it
file = open('/home/bestintown/textfile.txt','w')

file.write("Hello World!!") 
file.write("This is our new text file") 
file.write("and this is another line.") 
file.write("Why? Because we can.") 

file.close()

#We open in the file in read mode, specified by 'r'
f1 = open('/home/bestintown/textfile.txt','r')
f1.read() #Reads the contents of the file f1

#Reading excel or csv files
import csv as csv

with open('/home/bestintown/Records.csv','r') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)





########################## Loops & Control structures #########################

##if-else statements
if 1 < 2:
    print('first')
else:
    print('last')

#if-else-elif    
if 1 == 2:
    print('first')
elif 3 == 3:
    print('middle')
else:
    print('Last')

s=[1,2,3,4,5]

##for-loop
for item in s:
    print(item)    

##while loops
i = 1
while i < 5:
    print('i is: {}'.format(i))
    i = i+1
    
#range() function is widely used in defining loops
range(5)     

for i in range(5):
    print(i)

list(range(5))


################################# Functions ###################################

#Defining a function "my_func"
def my_func(param1='default'):
    """
    Docstring goes here.
    """
    print(param1)

my_func
my_func()
my_func('new param')
my_func(param1='new param')

#Defining a function, that returns the square of the given value
def square(x):
    return x**2

out = square(2) 
print(out)

##Lambda Expressions
#This gives us the liberty of defining a function, 

#This is how we normally define a function
def times2(var):
    return var*2

times2(2)

#To write a simplified expression, this is how we define a function, dropping
#Keywords
lambda v:v*2

##Map & filters
s=[1,2,3,4,5]

#Map basically maps the given function (times2) with the specified arguments(s).
#Filter basically filters out all the values based on the specified condition.
list(map(times2,s)) #

list(map(lambda var: var*2,s))

list(filter(lambda item: item%2 == 0,s))


##String methods

st = 'hello my name is Sam'

st.lower() #Returns the string in lowercase format
st.upper() #Returns the string in uppercase format
st.split() #Split method splits the string based on the arguments given,
           #if nothing is mentioned then, the given string is split into words,
           #returning a list of words.

tweet = 'Go Sports! #Sports'
tweet.split('#')
tweet.split('#')[1]       

##Dictionary methods

d = {'key1':'item1','key2':'item2'}
d #Displays the dictionary

d.keys() #Displays all the keys present in the dictionary
d.items()#Displays the key-value pairs

##List methods

l1 = [1,2,3]

l1.pop() #Removes the last element of the list
l1       #Displays l1


############################ Exceptional handling #############################
'''
try:
   You do your operations here;
   ......................
except ExceptionI:
   If there is ExceptionI, then execute this block.
except ExceptionII:
   If there is ExceptionII, then execute this block.
   ......................
else:
   If there is no exception then execute this block.'''
   
'''
try:
   You do your operations here;
   ......................
except(Exception1[, Exception2[,...ExceptionN]]]):
   If there is any exception from the given exception list, 
   then execute this block.
   ......................
else:
   If there is no exception then execute this block. '''   


############################ Importing modules ################################ 

arr = np.zeros(3) #array of zeros

from fib import fibonacci #does not import the entire module fib into the 
                          #current namespace; it just introduces the item fibonacci
                          #from the module fib into the global symbol table
                          #of the importing module

from modname import * #Provides an easy way to import all the items
                      #from the module into the current namespace


########################### Data Visualisation ################################

##Matplotlib

#Matplotlib allows you to create reproducible figures programmatically.
#Do explore the official Matplotlib web page: http://matplotlib.org/

#You'll need to install matplotlib first with either:
#conda install matplotlib or pip install matplotlib

import matplotlib.pyplot as plt

x = np.linspace(0, 5, 11)
y = x ** 2

x
y

#Basic Matplotlib commands to create simple line plots
plt.plot(x, y, 'r') # 'r' is the color red

plt.xlabel('X Axis Title Here')
plt.ylabel('Y Axis Title Here')
plt.title('String Title Here')
plt.show()

#Creating multiplots on same canvas

# plt.subplot(nrows, ncols, plot_number)
plt.subplot(1,2,1)
plt.plot(x, y, 'r--') # creates a -- line in red color
plt.subplot(1,2,2)
plt.plot(y, x, 'g*-') # creates a g*- line in green color

#The main idea in using the more formal Object Oriented method is to create figure
#objects and then just call methods or attributes off of that object.
#This approach is nicer when dealing with a canvas that has multiple plots on it.

# Create Figure (empty canvas)
fig = plt.figure()

# Add set of axes to figure
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)

# Plot on that set of axes
axes.plot(x, y, 'b') #'b' is the color blue
axes.set_xlabel('Set X Label') # Notice the use of set_ to begin methods
axes.set_ylabel('Set y Label')
axes.set_title('Set Title')

# Creates blank canvas
fig = plt.figure()

axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3]) # inset axes

# Larger Figure Axes 1
axes1.plot(x, y, 'b')
axes1.set_xlabel('X_label_axes2')
axes1.set_ylabel('Y_label_axes2')
axes1.set_title('Axes 2 Title')

# Insert Figure Axes 2
axes2.plot(y, x, 'r')
axes2.set_xlabel('X_label_axes2')
axes2.set_ylabel('Y_label_axes2')
axes2.set_title('Axes 2 Title')

#plt.subplots() object will act as a more automatic axis manager
# Use similar to plt.figure() except use tuple unpacking to grab fig and axes
fig, axes = plt.subplots()

# Now use the axes object to add stuff to plot
axes.plot(x, y, 'r')
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('title')

# Empty canvas of 1 by 2 subplots
fig, axes = plt.subplots(nrows=1, ncols=2)

# Axes is an array of axes to plot on
axes

#We can iterate this array
for ax in axes:
    ax.plot(x, y, 'b')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('title')

# Display the figure object    
fig

#A common issue with matplolib is overlapping subplots or figures. We ca use fig.
#tight_layout() or plt.tight_layout() method, which automatically adjusts 
#the positions of the axes on the figure canvas so that there is no overlapping
#content

fig, axes = plt.subplots(nrows=1, ncols=2)

for ax in axes:
    ax.plot(x, y, 'g')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('title')

fig    
plt.tight_layout()

#Matplotlib allows the aspect ratio, DPI and figure size to be specified when
#the Figure object is created. You can use the figsize and dpi keyword arguments.

fig = plt.figure(figsize=(8,4), dpi=100)

#The same arguments can be passed to layout managers, such as the subplots function
fig, axes = plt.subplots(figsize=(12,3))

axes.plot(x, y, 'r')
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('title')

#Matplotlib can generate high-quality output in a number formats, including PNG,
#JPG, EPS, SVG, PGF and PDF. To save a figure to a file we can use the savefig
#method in the Figure class

fig.savefig("filename.png")

fig.savefig("filename.png", dpi=200)

#Figure titles
#A title can be added to each axis instance in a figure. To set the title, use 
#the set_title method in the axes instance
ax.set_title("title")

#Axis labels
#Similarly, with the methods set_xlabel and set_ylabel, we can set the labels
#of the X and Y axes

ax.set_xlabel("x")
ax.set_ylabel("y")

#Legends
#You can use the label="label text" keyword argument when plots or other objects
#are added to the figure,and then using the legend method without arguments to
#add the legend to the figure

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.plot(x, x**2, label="x**2")
ax.plot(x, x**3, label="x**3")
ax.legend()

#The legend function takes an optional keyword argument loc that can be used to 
#specify where in the figure the legend is to be drawn. The allowed values of loc 
#are numerical codes for the various places the legend can be drawn.
#See the documentation page for details. Some of the most common loc values are:

ax.legend(loc=1) # upper right corner
ax.legend(loc=2) # upper left corner
ax.legend(loc=3) # lower left corner
ax.legend(loc=4) # lower right corner

# .. many more options are available

# Most common to choose
ax.legend(loc=0) # let matplotlib decide the optimal location
fig

#With matplotlib, we can define the colors of lines and other graphical elements
#in a number of ways

# MATLAB style line color and style 
fig, ax = plt.subplots()
ax.plot(x, x**2, 'b.-') # blue line with dots
ax.plot(x, x**3, 'g--') # green dashed line

#We can also define colors by their names or RGB hex codes and optionally provide
#an alpha value using the color and alpha keyword arguments. Alpha indicates
#opacity

fig, ax = plt.subplots()

ax.plot(x, x+1, color="blue", alpha=0.5) # half-transparant
ax.plot(x, x+2, color="#8B008B")         # RGB hex code
ax.plot(x, x+3, color="#FF8C00")         # RGB hex code 

#To change the line width, we can use the linewidth or lw keyword argument. 
#The line style can be selected using the linestyle or ls keyword arguments

fig, ax = plt.subplots(figsize=(12,6))

ax.plot(x, x+1, color="red", linewidth=0.25)
ax.plot(x, x+2, color="red", linewidth=0.50)
ax.plot(x, x+3, color="red", linewidth=1.00)
ax.plot(x, x+4, color="red", linewidth=2.00)

# possible linestype options ‘-‘, ‘–’, ‘-.’, ‘:’, ‘steps’
ax.plot(x, x+5, color="green", lw=3, linestyle='-')
ax.plot(x, x+6, color="green", lw=3, ls='-.')
ax.plot(x, x+7, color="green", lw=3, ls=':')

# custom dash
line, = ax.plot(x, x+8, color="black", lw=1.50)
line.set_dashes([5, 10, 15, 10]) # format: line length, space length, ...

# possible marker symbols: marker = '+', 'o', '*', 's', ',', '.', '1', '2', '3', '4', ...
ax.plot(x, x+ 9, color="blue", lw=3, ls='-', marker='+')
ax.plot(x, x+10, color="blue", lw=3, ls='--', marker='o')
ax.plot(x, x+11, color="blue", lw=3, ls='-', marker='s')
ax.plot(x, x+12, color="blue", lw=3, ls='--', marker='1')

# marker size and color
ax.plot(x, x+13, color="purple", lw=1, ls='-', marker='o', markersize=2)
ax.plot(x, x+14, color="purple", lw=1, ls='-', marker='o', markersize=4)
ax.plot(x, x+15, color="purple", lw=1, ls='-', marker='o', markersize=8, markerfacecolor="red")
ax.plot(x, x+16, color="purple", lw=1, ls='-', marker='s', markersize=8, 
        markerfacecolor="yellow", markeredgewidth=3, markeredgecolor="green")

#We can configure the ranges of the axes using the set_ylim and set_xlim methods
#in the axis object, or axis('tight') for automatically getting "tightly fitted" 
#axes ranges

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].plot(x, x**2, x, x**3)
axes[0].set_title("default axes ranges")

axes[1].plot(x, x**2, x, x**3)
axes[1].axis('tight')
axes[1].set_title("tight axes")

axes[2].plot(x, x**2, x, x**3)
axes[2].set_ylim([0, 60])
axes[2].set_xlim([2, 5])
axes[2].set_title("custom axes range")

#There are many specialized plots we can create, such as barplots, histograms,
#scatter plots, and much more.

#Scatter plot
plt.scatter(x,y)

from random import sample
data = sample(range(1, 1000), 100)

#Histogram
plt.hist(data)

data = [np.random.normal(0, std, 100) for std in range(1, 4)]

# rectangular box plot
plt.boxplot(data,vert=True,patch_artist=True)

##Scientific notation
#With large numbers on axes, it is often better use scientific notation

fig, ax = plt.subplots(1, 1)
      
ax.plot(x, x**2, x, np.exp(x))
ax.set_title("scientific notation")

ax.set_yticks([0, 50, 100, 150])

from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 
ax.yaxis.set_major_formatter(formatter)

#Axis number & axis label space
# distance between x and y axis and the numbers on the axes
plt.rcParams['xtick.major.pad'] = 5
plt.rcParams['ytick.major.pad'] = 5

fig, ax = plt.subplots(1, 1)
      
ax.plot(x, x**2, x, np.exp(x))
ax.set_yticks([0, 50, 100, 150])

ax.set_title("label and axis spacing")

# padding between axis label and axis numbers
ax.xaxis.labelpad = 5
ax.yaxis.labelpad = 5

ax.set_xlabel("x")
ax.set_ylabel("y")

# restore defaults
plt.rcParams['xtick.major.pad'] = 3
plt.rcParams['ytick.major.pad'] = 3

#Unfortunately, when saving figures the labels are sometimes clipped, and it can
#be necessary to adjust the positions of axes a little bit. This can be done
#using subplots_adjust
fig, ax = plt.subplots(1, 1)
      
ax.plot(x, x**2, x, np.exp(x))
ax.set_yticks([0, 50, 100, 150])

ax.set_title("title")
ax.set_xlabel("x")
ax.set_ylabel("y")

fig.subplots_adjust(left=0.15, right=.9, bottom=0.1, top=0.9)

#With the grid method in the axis object, we can turn on and off grid lines. 
#We can also customize the appearance of the grid lines using the same keyword 
#arguments as the plot function

fig, axes = plt.subplots(1, 2, figsize=(10,3))

# default grid appearance
axes[0].plot(x, x**2, x, x**3, lw=2)
axes[0].grid(True)

# custom grid appearance
axes[1].plot(x, x**2, x, x**3, lw=2)
axes[1].grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)

#We can also change the axis spines
fig, ax = plt.subplots(figsize=(6,2))

ax.spines['bottom'].set_color('blue')
ax.spines['top'].set_color('blue')

ax.spines['left'].set_color('red')
ax.spines['left'].set_linewidth(2)

# turn off axis spine to the right
ax.spines['right'].set_color("none")
ax.yaxis.tick_left() # only ticks on the left side

#Sometimes it is useful to have dual x or y axes in a figure; for example, 
#when plotting curves with different units together.
#Matplotlib supports this with the twinx and twiny functions
fig, ax1 = plt.subplots()

ax1.plot(x, x**2, lw=2, color="blue")
ax1.set_ylabel(r"area $(m^2)$", fontsize=18, color="blue")
for label in ax1.get_yticklabels():
    label.set_color("blue")
    
ax2 = ax1.twinx()
ax2.plot(x, x**3, lw=2, color="red")
ax2.set_ylabel(r"volume $(m^3)$", fontsize=18, color="red")
for label in ax2.get_yticklabels():
    label.set_color("red")

#Axes where x & y is zero
fig, ax = plt.subplots()

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0)) # set position of x spine to x=0

ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))   # set position of y spine to y=0

xx = np.linspace(-0.75, 1., 100)
ax.plot(xx, xx**3)

##Other 2D plots
n = np.array([0,1,2,3,4,5])
fig, axes = plt.subplots(1, 4, figsize=(12,3))

axes[0].scatter(xx, xx + 0.25*np.random.randn(len(xx)))
axes[0].set_title("scatter")

axes[1].step(n, n**2, lw=2)
axes[1].set_title("step")

axes[2].bar(n, n**2, align="center", width=0.5, alpha=0.5)
axes[2].set_title("bar")

axes[3].fill_between(x, x**2, x**3, color="green", alpha=0.5);
axes[3].set_title("fill_between")

#Text annotation
#Annotating text in matplotlib figures can be done using the text function
fig, ax = plt.subplots()

ax.plot(xx, xx**2, xx, xx**3)

ax.text(0.15, 0.2, r"$y=x^2$", fontsize=20, color="blue")
ax.text(0.65, 0.1, r"$y=x^3$", fontsize=20, color="green")

#Figures with multiple subplots and insets
#Axes can be added to a matplotlib Figure canvas manually using fig.add_axes or
#using a sub-figure layout manager such as subplots, subplot2grid, or gridspec

#subplots
fig, ax = plt.subplots(2, 3)
fig.tight_layout()

#subplot2grid
fig = plt.figure()
ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)
ax2 = plt.subplot2grid((3,3), (1,0), colspan=2)
ax3 = plt.subplot2grid((3,3), (1,2), rowspan=2)
ax4 = plt.subplot2grid((3,3), (2,0))
ax5 = plt.subplot2grid((3,3), (2,1))
fig.tight_layout()

#Gridspec
import matplotlib.gridspec as gridspec

fig = plt.figure()

gs = gridspec.GridSpec(2, 3, height_ratios=[2,1], width_ratios=[1,2,1])
for g in gs:
    ax = fig.add_subplot(g)
    
fig.tight_layout()

#add_axes
#Manually adding axes with add_axes is useful for adding insets to figures
fig, ax = plt.subplots()

ax.plot(xx, xx**2, xx, xx**3)
fig.tight_layout()

# inset
inset_ax = fig.add_axes([0.2, 0.55, 0.35, 0.35]) # X, Y, width, height

inset_ax.plot(xx, xx**2, xx, xx**3)
inset_ax.set_title('zoom near origin')

# set axis range
inset_ax.set_xlim(-.2, .2)
inset_ax.set_ylim(-.005, .01)

# set axis tick locations
inset_ax.set_yticks([0, 0.005, 0.01])
inset_ax.set_xticks([-0.1,0,.1])

##Colormap & Contour maps
#Colormaps and contour figures are useful for plotting functions of two variables. 
#In most of these functions we will use a colormap to encode one dimension of 
#the data. There are a number of predefined colormaps. It is relatively
#straightforward to define custom colormaps

alpha = 0.7
phi_ext = 2 * np.pi * 0.5

def flux_qubit_potential(phi_m, phi_p):
    return 2 + alpha - 2 * np.cos(phi_p) * np.cos(phi_m) - alpha * np.cos(phi_ext - 2*phi_p)

phi_m = np.linspace(0, 2*np.pi, 100)
phi_p = np.linspace(0, 2*np.pi, 100)
X,Y = np.meshgrid(phi_p, phi_m)
Z = flux_qubit_potential(X, Y).T

#pcolor
fig, ax = plt.subplots()

p = ax.pcolor(X/(2*np.pi), Y/(2*np.pi), Z, cmap=plt.cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max())
cb = fig.colorbar(p, ax=ax)

#imshow
fig, ax = plt.subplots()

im = ax.imshow(Z, cmap=plt.cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max(), extent=[0, 1, 0, 1])
im.set_interpolation('bilinear')

cb = fig.colorbar(im, ax=ax)

#contour
fig, ax = plt.subplots()

cnt = ax.contour(Z, cmap=plt.cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max(), extent=[0, 1, 0, 1])

##3D Figures

#To use 3D graphics in matplotlib, we first need to create an instance of the Axes3D 
#class. 3D axes can be added to a matplotlib figure canvas in exactly the same 
#way as 2D axes; or, more conveniently,by passing a projection='3d' keyword argument
#to the add_axes or add_subplot methods

from mpl_toolkits.mplot3d.axes3d import Axes3D

fig = plt.figure(figsize=(14,6))

# `ax` is a 3D-aware axis instance because of the projection='3d' keyword argument to add_subplot
ax = fig.add_subplot(1, 2, 1, projection='3d')

p = ax.plot_surface(X, Y, Z, rstride=4, cstride=4, linewidth=0)

# surface_plot with color grading and color bar
ax = fig.add_subplot(1, 2, 2, projection='3d')
p = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
cb = fig.colorbar(p, shrink=0.5)

#Wireframe plot
fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(1, 1, 1, projection='3d')

p = ax.plot_wireframe(X, Y, Z, rstride=4, cstride=4)

#Contour plots with projections
fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(1,1,1, projection='3d')

ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.25)
cset = ax.contour(X, Y, Z, zdir='z', offset=-np.pi, cmap=plt.cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='x', offset=-np.pi, cmap=plt.cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='y', offset=3*np.pi, cmap=plt.cm.coolwarm)

ax.set_xlim3d(-np.pi, 2*np.pi)
ax.set_ylim3d(0, 3*np.pi);
ax.set_zlim3d(-np.pi, 2*np.pi)


###Seaborn

#Seaborn is a Python visualization library based on matplotlib. It provides a 
#high-level interface for drawing attractive statistical graphics.

import seaborn as sns

tips = sns.load_dataset('tips')
tips.head()

#Barplot & countplot

#These very similar plots allow you to get aggregate data off a categorical feature
#in your data. Barplot is a general plot that allows you to aggregate the categorical 
#data based off some function, by default the mean

sns.barplot(x='sex',y='total_bill',data=tips)

#countplot

#This is essentially the same as barplot except the estimator is explicitly 
#counting the number of occurrences. Which is why we only pass the x value
sns.countplot(x='sex',data=tips)

#boxplot and violinplot

#boxplots and violinplots are used to shown the distribution of categorical data.

#A box plot (or box-and-whisker plot) shows the distribution of quantitative data
#in a way that facilitates comparisons between variables or across levels of a 
#categorical variable. The box shows the quartiles of the dataset while the whiskers
#extend to show the rest of the distribution, except for points that are determined 
#to be “outliers” using a method that is a function of the inter-quartile range

sns.boxplot(x="day", y="total_bill", data=tips,palette='rainbow')

# Can do entire dataframe with orient='h'
sns.boxplot(data=tips,palette='rainbow',orient='h')

sns.boxplot(x="day", y="total_bill", hue="smoker",data=tips, palette="coolwarm")

#violinplot

#A violin plot plays a similar role as a box and whisker plot. It shows the 
#distribution of quantitative data across several levels of one (or more) categorical
#variables such that those distributions can be compared. Unlike a box plot, in 
#which all of the plot components correspond to actual datapoints, the violin 
#plot features a kernel density estimation of the underlying distribution.

sns.violinplot(x="day", y="total_bill", data=tips,palette='rainbow')

sns.violinplot(x="day", y="total_bill", data=tips,hue='sex',palette='Set1')

sns.violinplot(x="day", y="total_bill", data=tips,hue='sex',split=True,palette='Set1')

#stripplot and swarmplot

#The stripplot will draw a scatterplot where one variable is categorical. A strip
#plot can be drawn on its own, but it is also a good complement to a box or violin
#plot in cases where you want to show all observations along with some representation 
#of the underlying distribution.

#The swarmplot is similar to stripplot(), but the points are adjusted (only along 
#the categorical axis) so that they don’t overlap. This gives a better representation 
#of the distribution of values, although it does not scale as well to large numbers 
#of observations (both in terms of the ability to show all the points and in terms
#of the computation needed to arrange them)

sns.stripplot(x="day", y="total_bill", data=tips)

sns.stripplot(x="day", y="total_bill", data=tips,jitter=True)

sns.stripplot(x="day", y="total_bill", data=tips,jitter=True,hue='sex',palette='Set1')

sns.stripplot(x="day", y="total_bill", data=tips,jitter=True,hue='sex',palette='Set1',split=True)

sns.swarmplot(x="day", y="total_bill", data=tips)

sns.swarmplot(x="day", y="total_bill",hue='sex',data=tips, palette="Set1", split=True)

##Distribution plots

#distplot
#The distplot shows the distribution of a univariate set of observations.

sns.distplot(tips['total_bill'])
# Safe to ignore warnings

#To remove the kde layer and just have the histogram use
sns.distplot(tips['total_bill'],kde=False,bins=30)

#jointplot
#jointplot() allows you to basically match up two distplots for bivariate data

sns.jointplot(x='total_bill',y='tip',data=tips,kind='scatter')

sns.jointplot(x='total_bill',y='tip',data=tips,kind='hex')

sns.jointplot(x='total_bill',y='tip',data=tips,kind='reg')

#pairplot
#pairplot will plot pairwise relationships across an entire dataframe (for the 
#numerical columns) and supports a color hue argument (for categorical columns)

sns.pairplot(tips)

sns.pairplot(tips,hue='sex',palette='coolwarm')

#rugplot

#rugplots are actually a very simple concept, they just draw a dash mark for 
#every point on a univariate distribution. They are the building block of a KDE plot

sns.rugplot(tips['total_bill'])

#kdeplot
#kdeplots are Kernel Density Estimation plots. These KDE plots replace every 
#single observation with a Gaussian (Normal) distribution centered around that value. For example:

from scipy import stats    
   
#Create dataset
dataset = np.random.randn(25)

# Create another rugplot
sns.rugplot(dataset);

# Set up the x-axis for the plot
x_min = dataset.min() - 2
x_max = dataset.max() + 2

# 100 equally spaced points from x_min to x_max
x_axis = np.linspace(x_min,x_max,100)

# Set up the bandwidth, for info on this:
url = 'http://en.wikipedia.org/wiki/Kernel_density_estimation#Practical_estimation_of_the_bandwidth'

bandwidth = ((4*dataset.std()**5)/(3*len(dataset)))**.2


# Create an empty kernel list
kernel_list = []

# Plot each basis function
for data_point in dataset:
    
    # Create a kernel for each point and append to list
    kernel = stats.norm(data_point,bandwidth).pdf(x_axis)
    kernel_list.append(kernel)
    
    #Scale for plotting
    kernel = kernel / kernel.max()
    kernel = kernel * .4
    plt.plot(x_axis,kernel,color = 'grey',alpha=0.5)

plt.ylim(0,1)

# To get the kde plot we can sum these basis functions.

# Plot the sum of the basis function
sum_of_kde = np.sum(kernel_list,axis=0)

# Plot figure
fig = plt.plot(x_axis,sum_of_kde,color='indianred')

# Add the initial rugplot
sns.rugplot(dataset,c = 'indianred')

# Get rid of y-tick marks
plt.yticks([])

# Set title
plt.suptitle("Sum of the Basis Functions")

#With the tips dataset
sns.kdeplot(tips['total_bill'])
sns.rugplot(tips['total_bill'])

sns.kdeplot(tips['tip'])
sns.rugplot(tips['tip'])

#Regression plots

#Seaborn has many built-in capabilities for regression plots

#lmplot allows you to display linear models, but it also conveniently allows you
#to split up those plots based off of features, as well as coloring the hue based 
#off of features.

#implot()

sns.lmplot(x='total_bill',y='tip',data=tips)

sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex')

#Working with markers

# http://matplotlib.org/api/markers_api.html
sns.lmplot(x='total_bill',y='tip',data=tips,hue='sex',palette='coolwarm',
           markers=['o','v'],scatter_kws={'s':100})

#Using a Grid

#We can add more variable separation through columns and rows with the use of a
#grid. Just indicate this with the col or row arguments

sns.lmplot(x='total_bill',y='tip',data=tips,col='sex')

sns.lmplot(x="total_bill", y="tip", row="sex", col="time",data=tips)

sns.lmplot(x='total_bill',y='tip',data=tips,col='day',hue='sex',palette='coolwarm')

#Aspect and Size

#Seaborn figures can have their size and aspect ratio adjusted with the size and
#aspect parameters
sns.lmplot(x='total_bill',y='tip',data=tips,col='day',hue='sex',palette='coolwarm',
          aspect=0.6,size=8)


##Matrix Plots

#Matrix plots allow you to plot data as color-encoded matrices and can also be used
#to indicate clusters within the data

flights = sns.load_dataset('flights')
flights.head()

#Heatmap

#In order for a heatmap to work properly, your data should already be in a matrix 
#form, the sns.heatmap function basically just colors it in for you

# Matrix form for correlation data
tips.corr()

sns.heatmap(tips.corr())

#A more fancy heatmap
sns.heatmap(tips.corr(),cmap='coolwarm',annot=True)

#For flights data
flights.pivot_table(values='passengers',index='month',columns='year')

pvflights = flights.pivot_table(values='passengers',index='month',columns='year')

#Heatmaps for private flights
sns.heatmap(pvflights)

#A more fancy heat-map for flights
sns.heatmap(pvflights,cmap='magma',linecolor='white',linewidths=1)

#clustermap

#The clustermap uses hierarchal clustering to produce a clustered version of the
#heatmap
sns.clustermap(pvflights)

# More options to get the information a little clearer like normalization
sns.clustermap(pvflights,cmap='coolwarm',standard_scale=1)


##Grids

#Grids are general types of plots that allow you to map plot types to rows and 
#columns of a grid, this helps you create similar plots separated by features

iris = sns.load_dataset('iris')
iris.head()

#PairGrid
#Pairgrid is a subplot grid for plotting pairwise relationships in a dataset

#just the grid
sns.PairGrid(iris)

# Then you map to the grid
g = sns.PairGrid(iris)
g.map(plt.scatter)

# Map to upper,lower, and diagonal
g = sns.PairGrid(iris)
g.map_diag(plt.hist)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)


#pairplot
#pairplot is a simpler version of PairGrid

sns.pairplot(iris)

sns.pairplot(iris,hue='species',palette='rainbow')


#Facet Grid
#FacetGrid is the general way to create grids of plots based off of a feature

# Just the Grid
g = sns.FacetGrid(tips, col="time", row="smoker")

g = sns.FacetGrid(tips, col="time",  row="smoker")
g = g.map(plt.hist, "total_bill")

g = sns.FacetGrid(tips, col="time",  row="smoker",hue='sex')
# Notice how the arguments come after plt.scatter call
g = g.map(plt.scatter, "total_bill", "tip").add_legend()

#JointGrid
#JointGrid is the general version for jointplot() type grids
g = sns.JointGrid(x="total_bill", y="tip", data=tips)
g = g.plot(sns.regplot, sns.distplot)

##Seaborn styles & Colors

#Styles
#You can set particular styles

sns.countplot(x='sex',data=tips)

sns.set_style('white')
sns.countplot(x='sex',data=tips) #Notice the difference

sns.set_style('ticks')
sns.countplot(x='sex',data=tips,palette='deep')

#Spine removal
sns.countplot(x='sex',data=tips)
sns.despine()

sns.countplot(x='sex',data=tips)
sns.despine(left=True)

#Size & aspect

#You can use matplotlib's plt.figure(figsize=(width,height) to change the size 
#of most seaborn plots. You can control the size and aspect ratio of most seaborn
#grid plots by passing in parameters: size, and aspect.

# Non Grid Plot
plt.figure(figsize=(12,3))
sns.countplot(x='sex',data=tips)

# Grid Type Plot
sns.lmplot(x='total_bill',y='tip',size=2,aspect=4,data=tips)

#Scale & Context

#set_context() allows you to override default parameters
sns.set_context('poster',font_scale=4)
sns.countplot(x='sex',data=tips,palette='coolwarm')

##Bokeh

#Bokeh is a Python library for interactive visualization that targets web 
#browsers for representation

#Bokeh has multiple language bindings (Python, R, lua and Julia). These bindings
#produce a JSON file, which works as an input for BokehJS (a Javascript library),
#which in turn presents data to the modern web browsers.

#Bokeh can produce elegant and interactive visualization like D3.js with high
#performance interactivity over very large or streaming datasets. Bokeh can help
#anyone who would like to quickly and easily create interactive plots, dashboards,
#and data applications.

##Installing Bokeh

#Install Bokeh using either:
    #conda install bokeh 
    #pip install bokeh
    
    
''' NOTE: Bokeh is under continuous development. Some methods that are used
          currently, might be depreciated in the new versions.
          Refer http://bokeh.pydata.org/en/latest/docs/user_guide/quickstart.html
          for reference.  '''
    
from bokeh.plotting import figure, output_file, show

# prepare some data
x = [1, 2, 3, 4, 5]
y = [6, 7, 2, 4, 5]

# output to static HTML file, which opens in your default browser
output_file("lines.html", title="line plot example")

# create a new plot with a title and axis labels
p = figure(title="simple line example", x_axis_label='x', y_axis_label='y')

# add a line renderer with legend and line thickness
p.line(x, y, legend="Temp.", line_width=2)

# show the results
show(p)

##Plotting a different plot
# prepare some data
x = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
y0 = [i**2 for i in x]
y1 = [10**i for i in x]
y2 = [10**(i**2) for i in x]

# output to static HTML file, which opens in your default browser
output_file("log_lines.html")

# create a new plot
p = figure(
   tools="pan,box_zoom,reset,save",
   y_axis_type="log", y_range=[0.001, 10**11], title="log axis example",
   x_axis_label='sections', y_axis_label='particles'
)

# add some renderers
p.line(x, x, legend="y=x")
p.circle(x, x, legend="y=x", fill_color="white", size=8)
p.line(x, y0, legend="y=x^2", line_width=3)
p.line(x, y1, legend="y=10^x", line_color="red")
p.circle(x, y1, legend="y=10^x", fill_color="red", line_color="red", size=6)
p.line(x, y2, legend="y=10^x^2", line_color="orange", line_dash="4 4")

# show the results
show(p)    

##Creating Bar charts

#Import Library
from bokeh.charts import Bar

# prepare data (dummy data)
data = {"y": [1, 2, 3, 4, 5]}

# Output to Line.HTML
output_file("lines.html", title="Bar plot example") 

# create a new line chat with a title and axis labels
p = Bar(data, title="Line Chart Example", xlabel='x', ylabel='values', width=400, height=400)   

# show the results
show(p)

#Vectorized colors and sizes

# prepare some data
N = 4000
x = np.random.random(size=N) * 100
y = np.random.random(size=N) * 100
radii = np.random.random(size=N) * 1.5
colors = ["#%02x%02x%02x" % (r, g, 150) for r, g in zip(np.floor(50+2*x), np.floor(30+2*y))]

# output to static HTML file (with CDN resources)
output_file("color_scatter.html", title="color_scatter.py example", mode="cdn")

TOOLS="resize,crosshair,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select"

# create a new plot with the tools above, and explicit ranges
p = figure(tools=TOOLS, x_range=(0,100), y_range=(0,100))

# add a circle renderer with vecorized colors and sizes
p.circle(x,y, radius=radii, fill_color=colors, fill_alpha=0.6, line_color=None)

# show the results
show(p)

##Linked panning & brushing

# linked panning where changing the range of one plot causes others to update
from bokeh.plotting import *

# prepare some data
N = 100
x = np.linspace(0, 4*np.pi, N)
y0 = np.sin(x)
y1 = np.cos(x)
y2 = np.sin(x) + np.cos(x)

# create a new plot
s1 = figure(width=250, plot_height=250, title=None)
s1.circle(x, y0, size=10, color="navy", alpha=0.5)

# NEW: create a new plot and share both ranges
s2 = figure(width=250, height=250, x_range=s1.x_range, y_range=s1.y_range, title=None)
s2.triangle(x, y1, size=10, color="firebrick", alpha=0.5)

# NEW: create a new plot and share only one range
s3 = figure(width=250, height=250, x_range=s1.x_range, title=None)
s3.square(x, y2, size=10, color="olive", alpha=0.5)

# NEW: put the subplots in a gridplot
p = gridplot([[s1, s2, s3]], toolbar_location=None)

# show the results
show(p)

#linked brushing (where a selection on one plot causes a selection to update on 
#other plots

from bokeh.models import ColumnDataSource

# prepare some date
N = 300
x = np.linspace(0, 4*np.pi, N)
y0 = np.sin(x)
y1 = np.cos(x)

# output to static HTML file
output_file("linked_brushing.html")

# NEW: create a column data source for the plots to share
source = ColumnDataSource(data=dict(x=x, y0=y0, y1=y1))

TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select"

# create a new plot and add a renderer
left = figure(tools=TOOLS, width=350, height=350, title=None)
left.circle('x', 'y0', source=source)

# create another new plot and add a renderer
right = figure(tools=TOOLS, width=350, height=350, title=None)
right.circle('x', 'y1', source=source)

# put the subplots in a gridplot
p = gridplot([[left, right]])

# show the results
show(p)

#Create a scatter square mark on XY frame
p = figure(plot_width=400, plot_height=400)
# add square with a size, color, and alpha
p.square([2, 5, 6, 4], [2, 3, 2, 1, 2], size=20, color="navy")
# show the results
show(p)

#Combine two visual elements in a plot
p = figure(plot_width=400, plot_height=400)
# add square with a size, color, and alpha
p.square([2, 5, 6, 4], [2, 3, 2, 1, 2], size=20, color="navy")
p.line([1, 2, 3, 4, 5], [1, 2, 2, 4, 5], line_width=2) #added a line plot to existing figure
# show the results
show(p)

#Add Hover tools & axis label to the above plot
from bokeh.models import HoverTool, BoxSelectTool #For enabling tools

#Add tools
TOOLS = [BoxSelectTool(), HoverTool()]

p = figure(plot_width=400, plot_height=400, tools=TOOLS)

# add a square with a size, color, and alpha
p.square([2, 5, 6, 4], [2, 3, 2, 1, 2], size=20, color="navy", alpha=0.5)

#Visual Elements
p.xaxis.axis_label = "X-axis"
p.yaxis.axis_label = "Y-axis"

# show the results
show(p)



##Scipy

#SciPy is a collection of mathematical algorithms and convenience functions built
#on the Numpy extension of Python

#The additional benefit of basing SciPy on Python is that this also makes a powerful
#programming language available for use in developing sophisticated programs and
#specialized applications. Scientific applications using SciPy benefit from the
#development of additional modules in numerous niches of the software landscape 
#by developers across the world.

#Everything from parallel programming to web and data-base subroutines and classes 
#have been made available to the Python programmer. All of this power is available
#in addition to the mathematical libraries in SciPy.


#Linear Algebra

#import linalg
from scipy import linalg

A = np.array([[1,2,3],[4,5,6],[7,8,8]])

# Compute the determinant of a matrix
linalg.det(A)

#Compute pivoted LU decomposition of a matrix.
#The decomposition is::
                       # A = P L U
#where P is a permutation matrix, L lower triangular with unit diagonal elements, 
#and U upper triangular.

P, L, U = linalg.lu(A)

P

L

U

np.dot(L,U)

#We can find out the eigenvalues and eigenvectors of this matrix
EW, EV = linalg.eig(A)

EW

EV

#Solving systems of linear equations can also be done
v = np.array([[2],[3],[5]])
v

s = linalg.solve(A,v)
s

#Sparse Linear Algebra
#SciPy has some routines for computing with sparse and potentially very large 
#matrices. The necessary tools are in the submodule scipy.sparse.

from scipy import sparse

# Row-based linked list sparse matrix
A = sparse.lil_matrix((1000, 1000))
A

A[0,:100] = np.random.randn(100)

A[1,100:200] = A[0,:100]

A.setdiag(np.random.rand(1000))
A

#Linear Algebra for sparse matrices

# Convert this matrix to Compressed Sparse Row format.
A.tocsr()

A = A.tocsr()

b = np.random.rand(1000)

sparse.linalg.spsolve(A,b)


##Stats in Scipy

#Import stats from scipy
from scipy import stats
from scipy.stats import norm 

#main public methods for continuous RVs are:
'''
    rvs: Random Variates
    pdf: Probability Density Function
    cdf: Cumulative Distribution Function
    sf: Survival Function (1-CDF)
    ppf: Percent Point Function (Inverse of CDF)
    isf: Inverse Survival Function (Inverse of SF)
    stats: Return mean, variance, (Fisher’s) skew, or (Fisher’s) kurtosis
    moment: non-central moments of the distribution

'''

#Normal RV 
print(norm.cdf(0))

#Generally useful methods are supported too
#Eg: norm.mean(), norm.std(), norm.var()

#To find the median of a distribution we can use the percent point function ppf, 
#which is the inverse of the cdf:
print(norm.ppf(0.5))

#To generate a sequence of random variates, use the size keyword argument:
print(norm.rvs(size=3))

#All continuous distributions take loc and scale as keyword parameters to adjust
#the location and scale of the distribution, e.g. for the standard normal
#distribution the location is the mean and the scale is the standard deviation.
print(norm.stats(loc = 3, scale = 4, moments = "mv"))

#uniform distribution:
from scipy.stats import uniform
print(uniform.cdf([0, 1, 2, 3, 4, 5], loc = 1, scale = 4))

#KDE
x1 = np.array([-7, -5, 1, 4, 5], dtype=np.float)
kde1 = stats.gaussian_kde(x1)
kde2 = stats.gaussian_kde(x1, bw_method='silverman')

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(x1, np.zeros(x1.shape), 'b+', ms=20)  # rug plot
x_eval = np.linspace(-10, 10, num=200)
ax.plot(x_eval, kde1(x_eval), 'k-', label="Scott's Rule")
ax.plot(x_eval, kde2(x_eval), 'r-', label="Silverman's Rule")

plt.show()

# We can define our own bandwidth function to get a less smoothed out result.
def my_kde_bandwidth(obj, fac=1./5):
     """We use Scott's Rule, multiplied by a constant factor."""
     return np.power(obj.n, -1./(obj.d+4)) * fac

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(x1, np.zeros(x1.shape), 'b+', ms=20)  # rug plot
kde3 = stats.gaussian_kde(x1, bw_method=my_kde_bandwidth)
ax.plot(x_eval, kde3(x_eval), 'g-', label="With smaller BW")

plt.show()

#We now take a more realistic example, and look at the difference between the two available bandwidth selection rules. Those rules are known to work well for (close to) normal distributions, but even for unimodal distributions that are quite strongly non-normal they work reasonably well. As a non-normal distribution we take a Student’s T distribution with 5 degrees of freedom.
np.random.seed(12456)
x1 = np.random.normal(size=200)  # random data, normal distribution
xs = np.linspace(x1.min()-1, x1.max()+1, 200)

kde1 = stats.gaussian_kde(x1)
kde2 = stats.gaussian_kde(x1, bw_method='silverman')

fig = plt.figure(figsize=(8, 6))

ax1 = fig.add_subplot(211)
ax1.plot(x1, np.zeros(x1.shape), 'b+', ms=12)  # rug plot
ax1.plot(xs, kde1(xs), 'k-', label="Scott's Rule")
ax1.plot(xs, kde2(xs), 'b-', label="Silverman's Rule")
ax1.plot(xs, stats.norm.pdf(xs), 'r--', label="True PDF")

ax1.set_xlabel('x')
ax1.set_ylabel('Density')
ax1.set_title("Normal (top) and Student's T$_{df=5}$ (bottom) distributions")
ax1.legend(loc=1)

x2 = stats.t.rvs(5, size=200)  # random data, T distribution
xs = np.linspace(x2.min() - 1, x2.max() + 1, 200)

kde3 = stats.gaussian_kde(x2)
kde4 = stats.gaussian_kde(x2, bw_method='silverman')

ax2 = fig.add_subplot(212)
ax2.plot(x2, np.zeros(x2.shape), 'b+', ms=12)  # rug plot
ax2.plot(xs, kde3(xs), 'k-', label="Scott's Rule")
ax2.plot(xs, kde4(xs), 'b-', label="Silverman's Rule")
ax2.plot(xs, stats.t.pdf(xs, 5), 'r--', label="True PDF")

ax2.set_xlabel('x')
ax2.set_ylabel('Density')

plt.show()

#We now take a look at a bimodal distribution with one wider and one narrower
#Gaussian feature. We expect that this will be a more difficult density to
#approximate, due to the different bandwidths required to accurately resolve each 
#feature.

from functools import partial

loc1, scale1, size1 = (-2, 1, 175)
loc2, scale2, size2 = (2, 0.2, 50)
x2 = np.concatenate([np.random.normal(loc=loc1, scale=scale1, size=size1),
                      np.random.normal(loc=loc2, scale=scale2, size=size2)])

x_eval = np.linspace(x2.min() - 1, x2.max() + 1, 500)

kde = stats.gaussian_kde(x2)
kde2 = stats.gaussian_kde(x2, bw_method='silverman')
kde3 = stats.gaussian_kde(x2, bw_method=partial(my_kde_bandwidth, fac=0.2))
kde4 = stats.gaussian_kde(x2, bw_method=partial(my_kde_bandwidth, fac=0.5))

pdf = stats.norm.pdf
bimodal_pdf = pdf(x_eval, loc=loc1, scale=scale1) * float(size1) / x2.size + \
               pdf(x_eval, loc=loc2, scale=scale2) * float(size2) / x2.size

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)

ax.plot(x2, np.zeros(x2.shape), 'b+', ms=12)
ax.plot(x_eval, kde(x_eval), 'k-', label="Scott's Rule")
ax.plot(x_eval, kde2(x_eval), 'b-', label="Silverman's Rule")
ax.plot(x_eval, kde3(x_eval), 'g-', label="Scott * 0.2")
ax.plot(x_eval, kde4(x_eval), 'c-', label="Scott * 0.5")
ax.plot(x_eval, bimodal_pdf, 'r--', label="Actual PDF")

ax.set_xlim([x_eval.min(), x_eval.max()])
ax.legend(loc=2)
ax.set_xlabel('x')
ax.set_ylabel('Density')
plt.show()

#With gaussian_kde we can perform multivariate as well as univariate estimation.
#We demonstrate the bivariate case. First we generate some random data with a 
#model in which the two variates are correlated.

def measure(n):
     """Measurement model, return two coupled measurements."""
     m1 = np.random.normal(size=n)
     m2 = np.random.normal(scale=0.5, size=n)
     return m1+m2, m1-m2

m1, m2 = measure(2000)
xmin = m1.min()
xmax = m1.max()
ymin = m2.min()
ymax = m2.max()

#Then we apply the KDE to the data:
X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([m1, m2])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel.evaluate(positions).T, X.shape)

#Finally we plot the estimated bivariate distribution as a colormap, and plot the individual data points on top.
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)

ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
           extent=[xmin, xmax, ymin, ymax])
ax.plot(m1, m2, 'k.', markersize=2)

ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])

plt.show()

##Fitting curves







































         