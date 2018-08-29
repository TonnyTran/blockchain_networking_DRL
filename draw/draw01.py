import matplotlib.pyplot as plt
from xlrd import open_workbook

# book = open_workbook('../results/result_v0.2_DDQN.xls')
book = open_workbook('../results/result_v0.6_10tr_QL.xls')

sheet = book.sheet_by_index(0)
interval = 10
# read header values into the list
# DDQNkeys = [sheet.cell(0, col_index).value for col_index in xrange(sheet.ncols)]
X, R, R_average = [], [], []
for row_index in xrange(1, sheet.nrows):
    # x = sheet.cell_value(row_index, 0)
    y = sheet.cell_value(row_index, 1)
    R.append(float(y))

for ave_index in range(0, len(R)/interval-1):
    R_ave = sum(R[ave_index*interval:((ave_index+1)*interval)])/interval
    R_average.append(R_ave)
    X.append(ave_index)

# print(X)
# print(R)
plt.xlabel('Number of episodes')
plt.ylabel('Total cost')
plt.plot(X, R_average, 'r', label="DQN", zorder=10)
plt.legend()
plt.show()