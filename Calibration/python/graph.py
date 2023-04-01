import matplotlib.pyplot as plt
a = sorted([27, 31, 43, 58, 69, 86, 102, 111, 122, 137, 18, 176])
b = sorted([71, 64, 52, 41, 33, 23, 17, 12, 2, 0, 87, -5], reverse=True)

a_outliners = sorted([122, 137])
b_outliners = sorted([2, 0], reverse=True)

c = [pow(10, -0.0107841 * x + 2.18732) for x in b]

plt.plot(b, a, linestyle='None', marker='.')
plt.plot(b_outliners, a_outliners, linestyle='None', marker='*')
plt.plot(b, c, marker='None')

plt.savefig('graphic.png')
