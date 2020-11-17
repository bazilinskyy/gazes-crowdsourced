from PIL import Image,  ImageDraw,  ImageColor,  ImageFont
import matplotlib.pyplot as plt
import numpy as np
import string
import random
import json
import generate_codecharts as gc
import os

# base images:
# 0. https://www.pexels.com/photo/women-s-white-and-black-button-up-collared-shirt-774909/  # noqa: E501
# 1. https://www.pexels.com/photo/woman-near-house-2804282/
# 2. https://www.pexels.com/photo/woman-standing-near-yellow-petaled-flower-2050994/  # noqa: E501
# 3. https://www.pexels.com/photo/person-holding-hammer-while-smiling-3267784/
# 4. https://www.pexels.com/photo/photography-of-laughing-guy-1408196/
# 5. https://www.pxfuel.com/en/free-photo-jmdxk
# 6. https://www.pexels.com/photo/man-in-blue-top-2830332/
# 7. https://www.pexels.com/photo/photo-of-man-wearing-denim-jacket-1599980/
# 8. https://www.pxfuel.com/en/free-photo-eibzf
# 9. https://www.pxfuel.com/en/free-photo-jrjqb

# parameters
text_color = ImageColor.getrgb("white")
font_type = "Arial.ttf"
px_pt_ratio = 20/29  # according to our image dimensions,  29 point = 20 px
valid_target_types = ["red_dot",  "fix_cross",  "img"]


def make_sentinel(codechart_filename,
                  sentinel_filename,
                  image_width,
                  image_height,
                  border_padding,
                  target_type="red_dot",
                  target_im_dir=""):
    # border_padding used to guarantee that chosen sentinel location is not too
    # close to border to be hard to spot

    font_size = int(image_height*0.0278)
    correct_codes = []

    if target_type not in valid_target_types:
        raise RuntimeError("target_type must be one of %s" %
                           valid_target_types.__str__())
    valid_codes,  coordinates = gc.create_codechart(codechart_filename,
                                                    image_width,
                                                    image_height)
    # pick random code
    r = list(range(0,  len(valid_codes)))
    index = random.choice(r)
    triplet = valid_codes[index]
    triplet_coordinate = coordinates[triplet]
    # to make sure that the cross is visible
    while (triplet_coordinate[0] <= border_padding or triplet_coordinate[0] >= image_width - border_padding) or \
            (triplet_coordinate[1] <= border_padding or triplet_coordinate[1] >= image_height - border_padding):
        index = random.choice(r)
        triplet = valid_codes[index]
        triplet_coordinate = coordinates[triplet]
    # check bg color
    if target_type == "fix_cross":
        bg_color = 126
    else:
        bg_color = 255
    # create and save cross sentinel image
    img = Image.new('RGB',
                    (image_width, image_height),
                    (bg_color, bg_color, bg_color))
    d = ImageDraw.Draw(img)
    try:
        # takes in point value
        font = ImageFont.truetype(font_type,  gc.pixel_to_point(font_size))
    except OSError:
        print("WARNING:using different font bc could not find %d" % font_type)
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf",  # noqa: E501
                                  gc.pixel_to_point(font_size))

    if target_type == "fix_cross":
        # offset cross location to the center of the triplet
        plot_coord = (triplet_coordinate[0]+font_size,  triplet_coordinate[1])
        d.text(plot_coord,  '+',  text_color,  font)
    elif target_type == "red_dot":
        d.ellipse((triplet_coordinate[0],
                   triplet_coordinate[1],
                   triplet_coordinate[0]+font_size*2,
                   triplet_coordinate[1]+font_size*2),
                  fill='red',
                  outline='red')
    elif target_type == "img":
        if not target_im_dir:
            raise RuntimeError("No im dir provided for sentinel targets")
        # Get a list of images in the target im dir
        images = os.listdir(target_im_dir)
        if '.DS_Store' in images:
            images.remove('.DS_Store')
        target = Image.open(os.path.join(target_im_dir,
                                         random.choice(images)))
        # resize the target
        width = 200
        height = int(target.height*width/target.width)
        target = target.resize((width,  height))
        plot_coord = (triplet_coordinate[0]-int(width/2),
                      triplet_coordinate[1]-int(height/2))
        img.paste(target,  plot_coord)

        # correct_codes lie within the sentinel width
        for ii in range(len(valid_codes)):
            dist = np.linalg.norm(np.array(coordinates[valid_codes[ii]]) -
                                  np.array(triplet_coordinate))
            if dist <= width/2.0 + 10:  # added padding for correct codes
                correct_codes.append(valid_codes[ii])

        pass
    else:
        raise RuntimeError("target_type %s does not exist" % target_type)
    img.save(sentinel_filename)
    D = {'correct_code': triplet,
         'coordinate': triplet_coordinate,
         'correct_codes': correct_codes}
    D_full = {'correct_code': triplet,
              'coordinate': triplet_coordinate,
              'valid_codes': valid_codes,
              'coordinates': coordinates,
              'codechart_file': codechart_filename,
              'correct_codes': correct_codes}
    return D, D_full


def generate_sentinels(sentinel_image_dir,
                       sentinel_CC_dir,
                       sentinel_images,
                       image_width,
                       image_height,
                       border_padding,
                       target_type,
                       target_im_dir=""):
    # Set up directories
    if not os.path.exists(sentinel_image_dir):
        os.makedirs(sentinel_image_dir)
    if not os.path.exists(sentinel_CC_dir):
        os.makedirs(sentinel_CC_dir)

    # Start generating sentinels
    # start at a new index id

    if not os.path.exists(sentinel_image_dir):
        os.makedirs(sentinel_image_dir)

    if not os.path.exists(sentinel_CC_dir):
        os.makedirs(sentinel_CC_dir)

    # save to a json the filename,  the coordinate of the + cross,
    # and the triplet at that coordinate
    data = {}
    # also save a list of other valid triplets and coordinates
    # (for analysis)
    data_with_coords = {}

    print('Populating %s with %d sentinel images' %
          (sentinel_image_dir, sentinel_images))
    print('Populating %s with %d corresponding codecharts' %
          (sentinel_CC_dir, sentinel_images))
    for i in range(sentinel_images):
        # generate random code chart
        codechart_filename = os.path.join(sentinel_CC_dir,
                                          'sentinel_cb_%d.jpg' %
                                          i)
        sentinel_filename = os.path.join(sentinel_image_dir,
                                         'sentinel_%d.jpg' %
                                         i)
        D, D_full = make_sentinel(codechart_filename,
                                  sentinel_filename,
                                  image_width,
                                  image_height,
                                  border_padding,
                                  target_type,
                                  target_im_dir)
        data[sentinel_filename] = D
        data_with_coords[sentinel_filename] = D_full

    with open(os.path.join(sentinel_image_dir,
                           'sentinel_codes.json'),
              'w') as outfile:
        json.dump(data,  outfile)
    print('Writing out %s' % (os.path.join(sentinel_image_dir,
                                           'sentinel_codes.json')))

    with open(os.path.join(sentinel_image_dir, 'sentinel_codes_full.json'),
              'w') as outfile:
        json.dump(data_with_coords,  outfile)
    print('Writing out %s' % (os.path.join(sentinel_image_dir,
                                           'sentinel_codes_full.json')))


if __name__ == "__main__":

    # Set these parameters
    sentinel_images = 400

    image_width = 1280
    image_height = 720
    # don't put fixation cross in this region of the image
    border_padding = 100
    rootdir = '../public/img/'

    target_type = "img"
    target_im_dir = "sentinel_target_images/black_white"

    sentinel_image_dir = os.path.join(rootdir, 'sentinel')
    sentinel_CC_dir = os.path.join(rootdir, 'codeboard_sentinel')

    generate_sentinels(sentinel_image_dir,
                       sentinel_CC_dir,
                       sentinel_images,
                       image_width,
                       image_height,
                       border_padding,
                       target_type,
                       target_im_dir)
