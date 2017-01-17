#!/usr/bin/env python
"""
Downloads ppts from CSE 548 course to given dir
"""
from __future__ import print_function

import os
import sys
import urllib2

def main():

    out_loc  = sys.argv[1]
    lectures = 13

    BASE_URL = "http://www3.cs.stonybrook.edu/~rezaul/Fall-2016/CSE548/CSE548-lecture-%s.pdf"

    for i in xrange(1, lectures+1):
        URL = BASE_URL % i
        LOCAL = os.path.join(out_loc, "CSE548-Lecture-%s.pdf" % i)
        web_file = urllib2.urlopen(URL)
        local_file = open(LOCAL, "wb")
        local_file.write(web_file.read())
        local_file.close()
        print("Completed", i)

if __name__ == '__main__':
    main()