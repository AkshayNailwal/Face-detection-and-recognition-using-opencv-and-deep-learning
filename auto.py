#import necessary modules
import os
import sys
import argparse

#initialize argument parser
ap = argparse.ArgumentParser()

#Add arguments to the parser
ap.add_argument("-f", "--file", required=True, help= "Provide the file with info aboutrequired depencies")

#parse arguments and print the file name
args = ap.parse_args()
print("Given file ---> "+str(args.file))

#opening the input file
INPUT_FILE = args.file
try:
    data = os.open(str(INPUT_FILE),os.O_RDONLY)
except Exception as err:
    #print exception to stdout
    sys.stdout.write("Can't open the given file")

#initiate remained dependencies to be installed
REMAINED = []

#install dependencies
for dependency in data:
    temp = os.system("pip install {}".format(dependency))
    if temp==0:
        #if status code is '0' means success
        sys.stdout.write("Succesfully installed {}".format(dependency))
    else:
        #if status code is '1' means fail
        REMAINED.append(dependency)
        sys.stdout.write("Can't Install {}".format(dependency))

#print all failed installations
out = "   ".join(REMAINED)
sys.stdout.write("Couldn't install following dependencies -->{}".format(out))