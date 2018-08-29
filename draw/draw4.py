import matplotlib.pyplot as plt
from xlrd import open_workbook
import xlsxwriter

interval = 10
book = open_workbook('../results/result_v0.6.xls')
book2 = open_workbook('../results/result_v0.6_QL.xls')
book3 = open_workbook('../results/result_v0.6_DDQN.xls')
book4 = open_workbook('../results/result_v0.6_QL.xls')
sheet = book.sheet_by_index(0)
sheet2 = book2.sheet_by_index(0)
sheet3 = book3.sheet_by_index(0)
sheet4 = book4.sheet_by_index(0)

file_name = '../result_draw/result_v1.2.1.xlsx'
workbook = xlsxwriter.Workbook(file_name)
worksheet = workbook.add_worksheet()


X, Y1, Y2, Y3, Y4, Y1_average, Y2_average, Y3_average, Y4_average = [], [], [], [], [], [], [], [], []
for row_index in xrange(1, sheet2.nrows):
    y1 = sheet.cell_value(row_index, 1)
    y2 = sheet2.cell_value(row_index, 1)
    y3 = sheet3.cell_value(row_index, 1)
    y4 = sheet4.cell_value(row_index, 1)
    Y1.append(float(y1))
    Y2.append(float(y2))
    Y3.append(float(y3))
    Y4.append(float(y4))

for ave_index in range(0, len(Y1)/interval-1):
    Y1_ave = sum(Y1[ave_index*interval:((ave_index+1)*interval)]) / interval
    Y1_average.append(Y1_ave)
    Y2_ave = sum(Y2[ave_index * interval:((ave_index + 1) * interval)]) / interval
    Y2_average.append(Y2_ave)
    Y3_ave = sum(Y3[ave_index * interval:((ave_index + 1) * interval)]) / interval
    Y3_average.append(Y3_ave)
    Y4_ave = sum(Y4[ave_index * interval:((ave_index + 1) * interval)]) / interval
    Y4_average.append(Y4_ave)
    X.append(ave_index)
    worksheet.write(ave_index + 1, 0, str(ave_index))
    worksheet.write(ave_index + 1, 1, str(Y1_ave))
    worksheet.write(ave_index + 1, 2, str(Y2_ave))
    worksheet.write(ave_index + 1, 3, str(Y3_ave))
    worksheet.write(ave_index + 1, 4, str(Y4_ave))

workbook.close()
# print(X)
# print(Y)
# print(Y2)
# print(Y3)
plt.xlabel('x' + str(interval) + ' Number of episodes')
plt.ylabel('Total reward')
plt.plot(X, Y1_average, 'r', label="DQN", zorder=10)
plt.plot(X, Y2_average, 'b', label="QL", zorder=10)
plt.plot(X, Y3_average, 'y', label="DDQN", zorder=10)
plt.plot(X, Y4_average, 'g', label="D3QN", zorder=10)
plt.legend()
plt.show()