# based on https://github.com/turkeyes/codecharts
# adapted by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
from PIL import Image, ImageDraw, ImageColor, ImageFont
import string
import random
import json
import os
import math

# DEFINE PARAMATERS
# letters to not use in code charts because can be confused with digits
forbidden_letters = set(["I", "O"])
px_pt_ratio = 20/29  # according to our image dimensions, 29 point = 20 px
text_color = ImageColor.getrgb("gray")
font_type = "Arial.ttf"
tojitter = True  # add jitter from a regular grid
# buffer number of pixels to leave from the edges of the image so# codecharts
# are not tangent to image edges
ebuf = 5
# if want to make sure to sample triplets to the very edge of the image
# (downside: triplets may be more crowded)
go_to_image_edges = False


def point_to_pixel(num):
    return int(num*px_pt_ratio)


def pixel_to_point(num):
    return int(num*(1/px_pt_ratio))


def generate_rand_letter():
    letter = random.choice(string.ascii_uppercase)
    while letter in forbidden_letters:
        letter = random.choice(string.ascii_uppercase)
    return letter


def generate_rand_triplet():
    code = ""
    code += generate_rand_letter()
    # the following code prevents the two digits from being identical or
    # equal to 0
    for i in range(2):
        if i == 0:
            forbidden_num = 0
        else:
            forbidden_num = int(code[i])
        r = list(range(1, forbidden_num)) + list(range(forbidden_num + 1, 10))
        code += str(random.choice(r))
    return code


def create_codechart(filename, image_width, image_height):
    font_size = int(image_height*0.0185)

    # all these parameters depend on font size
    # in pixel - max triplet width; used 'W88' as widest triplet code
    # (width~60, height=20)
    max_triplet_width = font_size * 3
    max_triplet_height = font_size  # the tallest a triplet can be
    # vertical distance to maintain b/w triplets in the grid
    d_v = 4 * max_triplet_height
    # horizontal distance to maintain b/w triplets (from start of one triplet
    # to start of another)
    d_h = 2 * max_triplet_width

    # make sure that not too much empty space is left over by spacing out
    # triplets
    # number of triplets that will be tiled horizontally
    N_h = int(math.floor((image_width - max_triplet_width - 2 * ebuf) /
              float(d_h)))
    # recompute the horizontal dist between triplets to eliminate extra space
    d_h = int(math.floor((image_width - max_triplet_width - 2 * ebuf) /
              float(N_h)))
    N_v = int(math.floor((image_height - max_triplet_height-2 * ebuf) /
              float(d_v)))
    d_v = int(math.floor((image_height - max_triplet_height - 2 * ebuf) /
              float(N_v)))
    # -------------
    # small buffer to cover edge case of triplets immediately adjacent to one
    # another (for legibility)
    post_jitter_buffer = 6
    # max vertical jitter for one side of a triplet
    j_v = int(0.25 * (d_v) - post_jitter_buffer / 2)
    # max horizontal jitter for on side of a triplet
    j_h = int(0.25 * (d_h) - post_jitter_buffer)

    # set up image canvas and font size/style
    img = Image.new('RGB', (image_width, image_height))
    d = ImageDraw.Draw(img)
    try:
        # takes in point value
        font = ImageFont.truetype(font_type, pixel_to_point(font_size))
    except OSError:
        print("WARNING: using a different font bc could not find %s on your \
               computer" % font_type)
        font_str = "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
        font = ImageFont.truetype(font_str, pixel_to_point(font_size))

    valid_codes = set()
    coordinates = {}

    # initialize starting locations for triplets on the image grid
    x_init = ebuf
    y_init = ebuf

    # -------- improvement made after 01/2020 (after TurkEyes paper) --------
    # original problem was grid-like artifacts in the collected data because
    # the grid spacing between consecutive triplets
    # was always similar (despite a small bit of jitter added when using the
    # triplet)
    xoffset = random.choice(list(range(int(d_h/2.0))))
    yoffset = random.choice(list(range(int(d_v/2.0))))
    x_init = x_init+xoffset
    y_init = y_init+yoffset
    # -----------------------------------------------------------------------

    x = x_init
    while x < image_width-max_triplet_width-ebuf:
        y = y_init
        while y < image_height-max_triplet_height-ebuf:
            triplet_code = generate_rand_triplet()

            # check for if triplet has already been used in image since all
            # codes should be unique
            while triplet_code in valid_codes:
                triplet_code = generate_rand_triplet()
            valid_codes.add(triplet_code)

            if tojitter:
                # implement jitter to x and y coordinates
                # (note: can turn either of them off)
                min_x = max(ebuf, x-j_h)
                max_x = min(x+j_h+1, image_width-max_triplet_width-ebuf)
                min_y = max(ebuf, y-j_v)
                # a little bit of extra buffer in vertical dimension
                max_y = min(y+j_v+1, image_height-max_triplet_height-ebuf-2)
                x_range = list(range(min_x, max_x))
                y_range = list(range(min_y, max_y))
                j_x = random.choice(x_range)
                j_y = random.choice(y_range)
            else:
                j_x = x
                j_y = y

            # writes triplet to image
            d.text((j_x, j_y), triplet_code, text_color, font)
            coordinates[triplet_code] = (j_x, j_y)

            y_prev = y
            y = y+d_v  # regularly sample the image vertically

            # triplets are not guaranteed to go to edge of image,
            # and gap could be large
            # see if can still squeeze in a triplet without overlapping
            # previous ones (could still be quite close)
            if go_to_image_edges and y >= image_height-max_triplet_height-ebuf:
                y = y_prev + max_triplet_height+j_v+1 + post_jitter_buffer*2

        x_prev = x
        x = x+d_h  # regularly sample the image horizontally

        if go_to_image_edges and x >= image_width-max_triplet_width-ebuf:
            x = x_prev + max_triplet_width+j_h+1 + post_jitter_buffer*2

    img.save(filename)
    return (list(valid_codes), coordinates)


if __name__ == "__main__":
    # create some code charts to test this code
    rootdir = '../public/img/codeboard'
    num_codecharts = 100  # generate this many codecharts

    # 720p
    image_width = 1280  # in pixel
    image_height = 720  # in pixel

    # set up directories
    if not os.path.exists(rootdir):
        os.makedirs(rootdir)

    data = {}
    for i in range(num_codecharts):
        filename = os.path.join(rootdir, 'cb_%d.jpg' % (i))
        valid_codes, coordinates = create_codechart(filename,
                                                    image_width,
                                                    image_height)
        data[filename] = (valid_codes, coordinates)

    with open(os.path.join(rootdir, 'data.json'), 'w') as outfile:
        json.dump(data, outfile)
