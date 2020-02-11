# Go to website https://data.galaxyzoo.org and download full catalog
# Go to http://skyserver.sdss.org/dr7/en/tools/chart/list.asp to download images
# Use aspx, e.g. http://skyservice.pha.jhu.edu/DR7/ImgCutout/getjpeg.aspx?ra=0.0017083333333333331&dec=-10.373805555555556&scale=0.2&width=240&height=120

# for backwards compatibility
from __future__ import print_function

# for downloading images
import requests
import shutil

# for parsing the csv file
import csv

# for checking whether the csv file exists
import os

# for giving the number of images to download as an argument
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='GalaxyZoo/')
parser.add_argument('--max', type=int, default=5000)
args = parser.parse_args()

PATH_TO_CSV = args.dir
if PATH_TO_CSV[-1] != '/':
    PATH_TO_CSV += '/'
if not os.path.exists(PATH_TO_CSV + 'GalaxyZoo1_DR_table2.csv'):
    print("File", PATH_TO_CSV + 'GalaxyZoo1_DR_table2.csv', "does not exist. Please specify a correct path.")
    exit(0)
MAX_NUMBER = args.max

# convert coordinates from csv in hh:mm:ss.ms format to degrees
def time_to_deg(t):
    (h, m, s) = t.split(':')
    mult = 1
    if h[0] == "-": mult = -1
    return int(h) + mult * int(m) / 60. + mult * float(s) / 3600.


# convert right ascension from hh:mm:ss.ms to degrees
def get_ra(t):
    return time_to_deg(t) * 15


def download_image(name, ra, dec):
    image_url = "http://skyservice.pha.jhu.edu/DR7/ImgCutout/getjpeg.aspx?ra=" + ra + "&dec=" + dec + "&scale=0.2&width=120&height=120"
    # print ra, dec, image_url
    resp = requests.get(image_url, stream=True)
    local_file = open(PATH_TO_CSV + name + '.jpg', 'wb')
    resp.raw.decode_content = True
    shutil.copyfileobj(resp.raw, local_file)
    del resp


# read in csv
hnd = open(PATH_TO_CSV + "training_data.txt", "w+")
hnd.write("[\n")
with open(PATH_TO_CSV + 'GalaxyZoo1_DR_table2.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    entry = 0
    for row in csv_reader:
        if entry % 100 == 0:
            print('Reading entry {:4d}'.format(entry))
        if entry == 0:  # skip header
            entry += 1
            continue
        elif entry > MAX_NUMBER:
            break
        else:
            ra, dec = str(get_ra(row[1])), str(time_to_deg(row[2]))
            download_image(row[0], ra, dec)
            training_row = [row[0], int(row[13]), int(row[14]), int(row[15])]
            hnd.write(str(training_row))
            if entry <= MAX_NUMBER: hnd.write(",\n")
            entry += 1

hnd.write("]\n")
hnd.close()
print("done.")
