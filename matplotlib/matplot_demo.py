import matplotlib.pyplot as plt

year = [1950, 1970, 1990, 2010]
pop = [2.519, 3.692, 5.263, 6.972]
# print(year.pop())
# print(pop.pop())
# 散点图
# plt.scatter(year, pop)
# 折线图
# plt.plot(year, pop)
# plt.show()

# values = [0, 0.6,1.4,1.6,2.2,2.5,2.6,3.2,3.5,3.9,4.2,6]
# plt.hist(values, bins=3)
# plt.show()
# plt.clf()

year = [1950, 1951, 1952, 2100]
pop = [2.538, 2.57, 2.62, 10.85]

# Add more data
year = [1800, 1850, 1900] + year
pop = [1.0, 1.262, 1.650] + pop

plt.plot(year, pop)
plt.grid(True)
plt.title('World Population Projections')
plt.xlabel('Year')
plt.ylabel('Population')
plt.yticks([0, 2, 4, 6, 8, 10], ['0', '2B', '4B', '6B', '8B', '10B'])
plt.show()
