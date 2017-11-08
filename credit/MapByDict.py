import sys
import re
import os
import datetime

def createDateDict(start, end, fileout, DateDict):
    start_date = datetime.date(*start)
    end_date = datetime.date(*end)


    result = []
    curr_date = start_date
    while curr_date != end_date:
        result.append("%04d%02d%02d" % (curr_date.year, curr_date.month, curr_date.day))
        curr_date += datetime.timedelta(1)
    result.append("%04d%02d%02d" % (curr_date.year, curr_date.month, curr_date.day))

    i = 0
    with open(fileout,'w') as FILEOUT:
        for date in result:
            print>>FILEOUT, date,"\t",i
            DateDict[date] = i
            i=i+1

    FILEOUT.close()


#a['gender1'].cat.categories=['male','female']

if __name__ == '__main__':
        DateDict = {}
        createDateDict((2015, 10, 1), (2017, 3, 1),"DateDicts.txt",DateDict)

 
        
