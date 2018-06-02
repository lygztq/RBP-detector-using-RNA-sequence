import os
import subprocess

ENTRY_DELIMITER = '$'
FIELD_DELIMITER = '#'


class RNAFolding:
	def __init__(self, **kw):
		if 'raw' in kw.keys():
			self.raw = kw['raw']
			fields = self.raw.split(FIELD_DELIMITER)
			self.folding = fields[0]
			self.energy, self.structure_probability = map(float, fields[1:])
		else:
			self.folding = kw['folding']
			self.energy = kw['energy']
			self.structure_probability = kw['structure_prob']

	def __str__(self):
		return '{}  Energy = {:.2f}  Structure probability = {:.7f}'.format(
			self.folding, self.energy, self.structure_probability)

	def __repr__(self):
		return self.__str__()


def rnashapes(rna):
	result = subprocess.run(
		[
			'./data_utils/rnashapes',
			'-r',
			'-O', 'D{{{e}%s}}E{{{f}%.7f}}R{{{f}%.7f}}'.format(e = ENTRY_DELIMITER, f = FIELD_DELIMITER),
			rna],
		stdout = subprocess.PIPE,
		cwd = os.path.dirname(__file__))
	result.check_returncode()
	result = [RNAFolding(raw = x) for x in result.stdout.decode('utf8').split(ENTRY_DELIMITER)[1:]]
	return result[0].folding


def _test():
	a = 'GACAGAGTGAGACCCTATCTCAAAAAACAAACAAAAAAGAGTTCTGTTTGGGCATGAAAGTGTTTAATGTTTATTGGACATTTGGGTGGAGATGTCGAGTAGGCAGTTGGAAATTCAAGTCTGTAGCTTAGGGATGAGAGCAGGTGGTCTCTCAGCAGTCATTGGCAATACATGATTCGTGGGACTGCAAGAGCTCATCACGGGAGAGAATATAAAGAAGCCCTTTCCCCTCATTCTGTAGATGAGGGAACTAT'
	b = rnashapes(a[:250])
	print(len(a), len(b))
	print(len(b[0].folding))
	print(b[0])

	sum = 0.0
	max_prob = 0.0
	for i in b:
		print(i.structure_probability)


if __name__ == '__main__':
	_test()
