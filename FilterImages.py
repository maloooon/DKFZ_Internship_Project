import os, os.path


""" Filter images which are less than 78 kb (78000 byte) in size (since these are 
mostly just blank white images which are not interesting for us)"""

i = 0
file_path = f'0to128/{22}/imgs/0'



for root, _, files in os.walk(file_path):
    for f in files:
        fullpath = os.path.join(root, f)
        try:
            if os.path.getsize(fullpath) < 78 * 1000:   #set file size in kb
               # print(fullpath)
                os.remove(fullpath)
        except ValueError:
            print("Error" + fullpath)