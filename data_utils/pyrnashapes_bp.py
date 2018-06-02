import subprocess
# from data_utils.RNA_process import shape2matrix
from RNA_process import shape2matrix

def rnashapes(rna):
    result = subprocess.run(['./data_utils/rnashapes', '-O', 'D{$%s}', rna], stdout = subprocess.PIPE)
    result.check_returncode()
    result = result.stdout.decode('utf8')
    result = result[result.index('$') + 1:]
    # result = shape2matrix(result)
    return result

def _test():
    
    a = 'GACAGAGTGAGACCCTATCTCAAAAAACAAACAAAAAAGAGTTCTGTTTGGGCATGAAAGTGTTTAATGTTTATTGGACATTTGGGTGGAGATGTCGAGTAGGCAGTTGGAAATTCAAGTCTGTAGCTTAGGGATGAGAGCAGGTGGTCTCTCAGCAGTCATTGGCAATACATGATTCGTGGGACTGCAAGAGCTCATCACGGGAGAGAATATAAAGAAGCCCTTTCCCCTCATTCTGTAGATGAGGGAACTATGCTCAGAATAGACGTATCTGTTTTCACAGTGAATCAGCAATAGAAT'
    a = a.lower()
    b = rnashapes(a[:100])
    print(b)
    print(len(b))
    print(a)

if __name__ == '__main__':
    _test()
