import subprocess

def rnashapes(rna):
    result = subprocess.run(['./data_utils/rnashapes', '-O', 'D{$%s}', rna], stdout = subprocess.PIPE)
    result.check_returncode()
    result = result.stdout.decode('utf8')
    result = result[result.index('$') + 1:]
    return result

def _test():
    print(rnashapes('cgauugcaugucgaugucgaugcugaugcaguugcaugcguaugcaugcgua'))

if __name__ == '__main__':
    _test()
