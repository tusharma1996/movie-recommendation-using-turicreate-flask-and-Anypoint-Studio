
import pyodbc
import csv
import pandas as pd
con = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER=172.16.100.7;DATABASE=JV3_JDI_PIM;UID=uatuser;PWD=uat@123')


cur = con.cursor()
querystring = "select * from testing"
cur.execute(querystring)

row = cur.fetchall()


with open('abcd.user', 'w', newline= '') as f:
    a = csv.writer(f, delimiter=',')
    a.writerows(row)  ## closing paren added

u_cols = ['name', 'age', 'class']
users = pd.read_csv('test1.csv', sep=',', names=u_cols,encoding='latin-1')
print (users)