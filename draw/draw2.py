import matplotlib.pyplot as plt
from xlrd import open_workbook
import xlsxwriter

book = open_workbook('../results/result_v0.6_QL.xls')
book2 = open_workbook('../results/result_v1.0_QL.xls')

file_name = '../result_draw/result_v0.1.xlsx'
workbook = xlsxwriter.Workbook(file_name)
worksheet = workbook.add_worksheet()

sheet = book.sheet_by_index(0)
sheet2 = book2.sheet_by_index(0)
interval = 10

# read header values into the list
# DDQNkeys = [sheet.cell(0, col_index).value for col_index in xrange(sheet.ncols)]
X, Y, Y2, Y_average, Y2_average = [], [], [], [], []
for row_index in xrange(1, sheet2.nrows):
    x = sheet2.cell_value(row_index, 0)
    if row_index < sheet.nrows:
        y = sheet.cell_value(row_index, 1)
    else:
        y = 0
    y2 = sheet2.cell_value(row_index, 1)

    Y.append(float(y))
    Y2.append(float(y2))

    # X.append(int(x))

for ave_index in range(0, len(Y)/interval-1):
    Y_ave = sum(Y[ave_index*interval:((ave_index+1)*interval)]) / interval
    Y_average.append(Y_ave)
    Y2_ave = sum(Y2[ave_index * interval:((ave_index + 1) * interval)]) / interval
    Y2_average.append(Y2_ave)
    X.append(ave_index)
    worksheet.write(ave_index + 1, 0, str(ave_index))
    worksheet.write(ave_index + 1, 1, str(Y_ave))
    worksheet.write(ave_index + 1, 2, str(Y2_ave))

workbook.close()
print(X)
print(Y)
print(Y2)
plt.xlabel('Number of episodes')
plt.ylabel('Total reward')
plt.plot(X, Y_average, 'b', label="DQN", zorder=10)
plt.plot(X, Y2_average, 'r', label="QL", zorder=10)

plt.legend()
plt.show()