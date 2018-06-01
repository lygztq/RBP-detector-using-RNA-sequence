from data_utils.pyrnashapes import rnashapes
b = 'cgattgcatgtcgatgtcgatgctgatgcagttgcatgcgtatgcatgcgta'
a = rnashapes(b)
print(type(a))
print(len(a), len(b))
print(a)

