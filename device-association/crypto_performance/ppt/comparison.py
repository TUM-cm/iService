import numpy
import scipy.spatial

a = numpy.array([1,342,376,185,55,9,470,472,29,70,198,442,95,347,111,478,51,88,44,318])
b = numpy.array([15,477,373,40,118,158,129,297,454,220,487,437,14,289,37,111,172,279,124,98])

diff = a-b
sum_diff_ab = sum(diff*diff)

sum_ab = sum(a*b)
sum_aa = sum(a*a)
sum_bb = sum(b*b)
print(sum_ab)
print(sum_aa)
print(sum_bb)
print(sum_diff_ab)

print("with modulo")
modulo = 44000069
print(sum_aa % modulo)
print(sum_bb % modulo)
print(sum_diff_ab % modulo)

print("-----")
rest = sum_ab % modulo
print(rest)
print(rest % modulo)

print("Euclidean: ", scipy.spatial.distance.euclidean(a, b))
print("Cosine: ", scipy.spatial.distance.cosine(a, b))

#./helib v1='[397 323 145]' v2='[450 232 289]'
