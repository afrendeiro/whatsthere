#!/usr/bin/env python


def parse_gmt(gmt_file):
	"""
	:gmt_file: str
	:returns: dict
	"""
	try:
		with open(gmt_file, "r") as handle:
			lines = handle.readlines()
	except IOError("Could not open %s" % gmt_file) as e:
		raise e

	# parse into dict
	gene_sets = dict()
	for line in lines:
		line = line.strip().split("\t")
		line.pop(1)  # remove url
		set_id = line.pop(0)

		gene_sets[set_id] = set(line)

	return gene_sets
