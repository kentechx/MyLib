import cv2
import numpy as np
from PIL import Image, ImageOps
from typing import List, Dict, Any, Union


def exists(val):
    return val is not None


def generate_colors(n):
    import seaborn as sns
    colors = sns.color_palette('hls', n)
    colors = [[int(c * 255) for c in color] for color in colors]
    return colors


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


def box_label(im, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    # Add one xyxy box to image with label
    lw = 3
    if isinstance(box, np.ndarray):
        box = box.tolist()
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(im, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=0.5, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im,
                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    0.5,
                    txt_color,
                    thickness=0,
                    lineType=cv2.LINE_AA)
    return im


class ImageHelper:
    def __init__(self):
        pass

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
    def draw_bbox_labels(img: np.ndarray, bboxes: np.ndarray, labels: List[str] = None):
        # img = ImageHelper.draw_bbox(img, bboxes, thickness)
        h, w = img.shape[:2]
        img = img_to_u1(img)
        if exists(labels):
            labels = label_to_str(labels)
        else:
            labels = [None for _ in range(len(bboxes))]

        for bbox, label, color in zip(bbox_to_xyxy(bboxes, h, w), labels, generate_colors(len(bboxes))):
            img = box_label(img, bbox, label, color)

        return img

    @staticmethod
    def draw_mask_labels(img: np.ndarray, mask: np.ndarray, ignore_labels=[0]):
        # mask: (h, w), uint8
        out = img.copy()
        ls = np.unique(mask)
        colors = generate_colors(len(ls))
        for l, color in zip(ls, colors):
            if l in ignore_labels:
                continue
            # get contours
            mask_l = np.zeros_like(mask, dtype=np.uint8)
            mask_l[mask == l] = 1
            contours, _ = cv2.findContours(mask_l, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # draw contours
            cv2.drawContours(out, contours, -1, color, 1)
            # draw label
            for contour in contours:
                x, y = contour[:, 0].mean(0).astype(int)
                cv2.putText(out, str(l), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return out

    @staticmethod
    def visualize_points(img, points):
        # cv2 plot points
        img = img.copy()
        for pt in points:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), -1)
        cv2.imshow('img', img)
        cv2.waitKey()
        cv2.destroyAllWindows()

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
    def visualize_switch(imgs: List):
        import cv2
        i = 0
        img = imgs[i]
        while 1:
            cv2.imshow('', img)
            if cv2.waitKey(0) & 0xFF == ord(' '):
                i = (i + 1) % len(imgs)
                img = imgs[i]
            else:
                cv2.destroyAllWindows()
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
        cv2.destroyAllWindows()

    @staticmethod
    def visualize_with_bbox(img, bbox, labels=None):
        img = ImageHelper.draw_bbox_labels(img, bbox, labels)
        ImageHelper.visualize(img)
