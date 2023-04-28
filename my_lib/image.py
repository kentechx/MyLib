import cv2
import numpy as np
from PIL import Image, ImageOps
from typing import List, Dict, Any, Union


def exists(val):
    return val is not None


def img_to_u1(img):
    if img.max() <= 1.:
        img = (img * 255).astype('u1')
    else:
        img = img.astype('u1')
    return img


def bbox_to_xyxy(bboxes, h, w):
    if bboxes.max() <= 1.:
        bboxes = bboxes * np.array([w, h, w, h])
        bboxes = bboxes.astype('i4')
    else:
        bboxes = bboxes.astype('i4')
    return bboxes


def label_to_str(labels):
    labels = np.array(labels, dtype='str')
    return labels


class ImageHelper:
    def __init__(self):
        pass

    @staticmethod
    def get_edge(mask, color=(255, 255, 255), thickness=None):
        """
        :param mask: (h, w)
        :return: contour: (h, w, 3)
        """
        _, thresh_im = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_contour = np.zeros_like(mask)
        cv2.drawContours(img_contour, contours, -1, color, thickness)  # -1 indicates all contours are drawn
        return img_contour

    @staticmethod
    def crop_with_mask(img: np.ndarray, mask: np.ndarray, margin=(50, 50)):
        """
        :param img:
        :param mask: with shape (h, w, 3)
        :param margin: (w, h), where `h` is the top and bottom margin, and `w` is the left and right margin.
        :return:
        """
        if len(mask.shape) == 3:
            x, y, w, h = cv2.boundingRect(mask.max(-1))
        else:
            x, y, w, h = cv2.boundingRect(mask)
        wm, hm = margin
        left, top, right, bottom = max(x - wm, 0), max(y - hm, 0), min(x + w + wm, img.shape[1]), min(y + h + hm,
                                                                                                      img.shape[0])
        return img[top:bottom, left:right], mask[top:bottom, left:right]

    @staticmethod
    def resize_and_pad(img: np.ndarray, mask: np.ndarray, target_res=512):
        # this will keep the aspect ratio of the content
        # resize
        ratio = target_res / max(img.shape)
        new_size = (int(ratio * img.shape[1]), int(ratio * img.shape[0]))  # (w, h)!
        img = cv2.resize(img, new_size)
        mask = cv2.resize(mask, new_size)

        # pad
        delta_w = target_res - new_size[0]
        delta_h = target_res - new_size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
        img, mask = Image.fromarray(img), Image.fromarray(mask)
        img, mask = ImageOps.expand(img, padding), ImageOps.expand(mask, padding)
        return np.array(img), np.array(mask)

    @staticmethod
    def draw_bbox(img: np.ndarray, bboxes: np.ndarray, thickness=2, is_opaque=False, alpha=0.5):
        import bbox_visualizer as bbv
        import seaborn as sns
        colors = sns.color_palette('hls', len(bboxes))
        colors = [[int(c * 255) for c in color] for color in colors]
        img = img_to_u1(img)
        for box, color in zip(bboxes, colors):
            img = bbv.draw_rectangle(img, bbox_to_xyxy(box, *img.shape[:2]),
                               bbox_color=color, thickness=thickness, is_opaque=is_opaque, alpha=alpha)
        return img

    @staticmethod
    def draw_bbox_labels(img: np.ndarray, bboxes: np.ndarray, labels: List[str] = None,
                         bbox_color: Union[List[List[int]], List[int], None] = None,
                         thickness=2):
        import bbox_visualizer as bbv
        img = ImageHelper.draw_bbox(img, bboxes, thickness)
        h, w = img.shape[:2]

        if exists(labels):
            # auto color
            img = bbv.add_multiple_labels(img, label_to_str(labels), bbox_to_xyxy(bboxes, h, w))

        return img

    @staticmethod
    def remove_small_components(image: np.ndarray, thresh=10):
        """
        :param image:  (h, w), in range (0, c), where `c` is the max class label
        :param thresh:
        :return:
        """
        ret_image = image.copy()
        n, labels_im = cv2.connectedComponents(ret_image)

        for i in range(n):
            idx = np.where(labels_im == i)
            if idx[0].shape[0] < thresh:
                ret_image[idx] = 0
        return ret_image

    @staticmethod
    def visualize_switch(img1, img2):
        import cv2
        flag = 1
        img = img1
        while 1:
            cv2.imshow('', img)
            if cv2.waitKey(0) & 0xFF == ord(' '):
                if flag == 1:
                    flag = 2
                    img = img2
                else:
                    flag = 1
                    img = img1
            else:
                break

    @staticmethod
    def visualize_images(imgs: List[np.ndarray], cols=1):
        # stack images
        _imgs = []
        for i in range(0, len(imgs), cols):
            _imgs.append(np.hstack(imgs[i:i + cols]))
        _imgs = np.concatenate(_imgs, axis=0)
        cv2.imshow('', _imgs)
        cv2.waitKey()
        cv2.destroyAllWindows()

    @staticmethod
    def visualize(img):
        cv2.imshow('', img)
        cv2.waitKey()

    @staticmethod
    def visualize_with_bbox(img, bbox, labels=None):
        img = ImageHelper.draw_bbox_labels(img, bbox, labels)
        ImageHelper.visualize(img)
