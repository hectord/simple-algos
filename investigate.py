# -*- coding: utf-8 -*-

from save import read
from display import display

if __name__ == '__main__':

    # check the second layer, it doens't look like anything
    images = []
    for i in [100, 1100, 2100, 3100, 4100, 5100, 9100]:
        ret = read(f'output/{i}/W2.txt')
        minv, maxv = ret.min(), ret.max()
        image = (ret - minv) / (maxv - minv)

        images.append(image)


    display(images)
