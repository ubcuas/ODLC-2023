from imgaug import augmenters as iaa
import imgaug as ia
import math
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage


def imageAugmentation(image, bounding_boxes, image_height, image_width, num_aug_imgs=3, display=False):
    """
    Auments the image given the bounding boxes of the shapes.
    bounding_boxes[i] format: [(top_left_x, top_left_y), (bottom_right_x, bottom_right_y)].
    Output will be a list of num_aug_img augmented images with their corresponding bounding boxes.
    All augmented images will be displayed if display == True.
    """

    # define the augmentations
    # augmentation methods: 1. left right flip, 2. up down flip, 3. affine (scaling + rotate + up down left right tranformation + sheering)
    # 4. noise, 5. croping, 6. change color, 7. (50% of times) guassian blur, 8. change contrast, 9. make brighter or darker
    seq = iaa.Sequential([
        # iaa.Fliplr(0.5),
        # iaa.Flipud(0.5),
        iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            rotate=(-180, 180),
            # translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            shear=(-4, 4),
            mode="constant", # this will fill in newly-created pixels using the mode specified; more info at https://imgaug.readthedocs.io/en/latest/source/api_augmenters_geometric.html
            cval=(128, 128)
        ),
        # iaa.AdditiveGaussianNoise(scale=(10, 50)),
        iaa.Crop(percent=(0, 0.05)),
        iaa.AddToHueAndSaturation((-10, 10)),
        iaa.Sometimes(
            1.0,
            iaa.GaussianBlur(sigma=(0, 1.0))
        ),
        iaa.LinearContrast((0.75, 1.5)),
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
    ], random_order=True)

    # convert boudning box into keypoints
    key_pts_ls = []
    key_pts_on_img = []
    for i in range(len(bounding_boxes)):
        highest = bounding_boxes[i][0][1]
        lowest = bounding_boxes[i][1][1]
        leftmost = bounding_boxes[i][0][0]
        rightmost = bounding_boxes[i][1][0]
        
        # # code for the case len(bounding_boxes) == 1
        # key_pts = [
        #     Keypoint(leftmost, highest),
        #     Keypoint(rightmost, highest),
        #     Keypoint(leftmost, lowest),
        #     Keypoint(rightmost, lowest)
        # ]
        # key_pts_on_img = KeypointsOnImage(key_pts, shape=(image_height, image_width))
        
        for pt in [
            Keypoint(leftmost, highest),
            Keypoint(rightmost, highest),
            Keypoint(leftmost, lowest),
            Keypoint(rightmost, lowest)
        ]:
            key_pts_ls.append(pt)
    key_pts_on_img = KeypointsOnImage(key_pts_ls, shape=(image_height, image_width))

    # # old code to convert bounding boxes into keypoints
    # # get key points
    # key_pts_on_imgs = []
    # for i in range(no_of_imgs):
    #     # get the labels
    #     label_file = open(str(i) + ".txt")
    #     label = label_file.read()
    #     label_splitted = label.split()
    #     label_file.close()
    #     x_center = float(label_splitted[1])
    #     y_center = float(label_splitted[2])
    #     width = float(label_splitted[3])
    #     height = float(label_splitted[4])

    #     # convert labels into absolute coordinates of the corners
    #     highest = 600*(y_center - height/2)
    #     lowest = 600*(y_center + height/2)
    #     leftmost = 800*(x_center - width/2)
    #     rightmost = 800*(x_center + width/2)
    #     key_pts = [
    #         Keypoint(leftmost, highest),
    #         Keypoint(rightmost, highest),
    #         Keypoint(leftmost, lowest),
    #         Keypoint(rightmost, lowest)
    #     ]
    #     key_pts_on_img = KeypointsOnImage(key_pts, shape=(600, 800))
    #     key_pts_on_imgs.append(key_pts_on_img)


    # augment
    images_aug = [image]
    key_pts_on_imgs_aug = [key_pts_on_img]
    for _ in range(num_aug_imgs - 1):
        image_aug, key_pts_on_img_aug = seq(image=image, keypoints=key_pts_on_img)
        images_aug.append(image_aug)
        key_pts_on_imgs_aug.append(key_pts_on_img_aug)
    
    # # this code is for the case len(bounding_boxes) == 1
    # # find the highest, lowest, leftmost and rightmost points of key_pts_on_imgs_aug
    # # then append to new_labels
    # new_labels = []
    # for i in range(num_aug_imgs):
    #     # initialize values
    #     highest = image_height
    #     lowest = 0
    #     leftmost = image_width
    #     rightmost = 0
    #
    #     # determine highest, lowest, leftmost and rightmost
    #     for j in range(4):
    #         x_coord = key_pts_on_imgs_aug[i][j].x
    #         y_coord = key_pts_on_imgs_aug[i][j].y
    #         if x_coord > rightmost:
    #             rightmost = x_coord
    #         if x_coord < leftmost:
    #             leftmost = x_coord
    #         if y_coord < highest:
    #             highest = y_coord
    #         if y_coord > lowest:
    #             lowest = y_coord
    #
    #     # append values to new_label, then to new_labels
    #     new_label = [(leftmost, highest), (rightmost, lowest)]
    #     new_labels.append(new_label)
    
    # convert keypoints back to bounding boxes
    # structure of new_bounding_boxes:
    # [
    #   image 1
    #   [
    #       shape 1
    #           [point 1
    #           point 2]
    #       shape 2
    #           [point 1
    #           point 2]
    #        .
    #        .
    #        .
    #       shape k
    #           [point 1
    #           point 2]
    #   ]
    #    .
    #    .
    #    .
    #   image n
    #   [
    #       shape 1
    #           [point 1
    #           point 2]
    #       shape 2
    #           [point 1
    #           point 2]
    #        .
    #        .
    #        .
    #       shape k
    #           [point 1
    #           point 2]
    #   ]
    # ]
    new_bounding_boxes_ls = []
    num_of_shapes = len(bounding_boxes)
    for img_idx in range(num_aug_imgs):
        new_bounding_boxes = []
        for shape_idx in range(num_of_shapes):
            # initialize values
            highest = image_height
            lowest = 0
            leftmost = image_width
            rightmost = 0
            
            # determine highest, lowest, leftmost and rightmost
            for i in range(4):
                x_coord = key_pts_on_imgs_aug[img_idx][4*shape_idx + i].x
                y_coord = key_pts_on_imgs_aug[img_idx][4*shape_idx + i].y
                if x_coord > rightmost:
                    rightmost = x_coord
                if x_coord < leftmost:
                    leftmost = x_coord
                if y_coord < highest:
                    highest = y_coord
                if y_coord > lowest:
                    lowest = y_coord
            
            # append bounding box
            new_bounding_boxes.append([(leftmost, highest), (rightmost, lowest)])
        
        # append bounding_boxes
        new_bounding_boxes_ls.append(new_bounding_boxes)
            
    
    # display things
    if display == True:
        # images per row and number of rows for displaying
        images_per_row = 5
        rows = math.ceil(10/images_per_row)

        # display
        ia.imshow(
        ia.draw_grid(
            [key_pts_on_imgs_aug[i].draw_on_image(images_aug[i], size=10, color=(255, 69, 0)) for i in range(num_aug_imgs)],
            cols=images_per_row, rows=rows
            )
        )
        
    return images_aug, new_bounding_boxes_ls