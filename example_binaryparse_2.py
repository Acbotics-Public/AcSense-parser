import sys

# sys.path.append("../src/")
import os
import AcSense_parser_utils

###################
# example scripts to operate parser on data from different AcSense units
# run:
# python3 example.py
#
##################
# Model options:
# 0: AcSense-PLUS
# 1: AcSense-8CH or AcSense-16CH
# 2: AcSense-Mini
# 3: AcSense-Mini-48hr


def run_parse_AcSense(indir, unit_model=0, outdir=None, plotting=False, export=True):
    if outdir is None:
        outdir = os.path.join(indir, "parsed")
    if unit_model == 0 or unit_model == 1:
        AcSense_parser_utils.load_process_all_files(
            indir, outdir, hydrophone_ADC="EXT", plotting=plotting, export=export
        )
    else:
        AcSense_parser_utils.load_process_all_files(
            indir, outdir, hydrophone_ADC="INT", plotting=plotting, export=export
        )


mypath = "/home/efischell/Desktop/SDcard/D7/"
run_parse_AcSense(mypath, unit_model=3, plotting=True)
