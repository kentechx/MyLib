import os

import numpy as np
import cv2
import json
from typing import List, Dict, Any
from .utils import _test_time


class MeshLabelerHelper:
    _shape_keys = ["shape_type", "label", "metadata", "points", "face_ids"]
    _shape_type_keys = ["face", "landmark", "vertex"]

    @staticmethod
    def read_json_data(fp) -> Dict[str, Any]:
        return json.load(open(fp, 'r', encoding='utf-8'))

    @staticmethod
    def write_json_data(data: Dict[str, Any], fp):
        open(fp, 'w', encoding='utf-8').write(json.dumps(data))

    @staticmethod
    def save_face_labels(face_labels: np.ndarray, fp: str):
        data = MeshLabelerHelper.get_json_data_from_face_labels(face_labels)
        MeshLabelerHelper.write_json_data(data, fp)

    @staticmethod
    def get_face_labels(data: Dict[str, Any]) -> np.ndarray:
        if 'num_faces' not in data:
            face_shapes = [sh for sh in data['shapes'] if sh['shape_type'] == 'face']
            n_faces = 1 if len(face_shapes) == 0 else max(max(sh['face_ids']) for sh in face_shapes) + 1
        else:
            n_faces = data['num_faces']

        labels = np.zeros(n_faces)
        for sh in data['shapes']:
            if sh['shape_type'] == 'face':
                labels[sh['face_ids']] = int(sh['label'])
        return labels

    @staticmethod
    def get_json_data_from_face_labels(face_labels: np.ndarray) -> Dict[str, Any]:
        label = {
            "metadata": [],
            "num_faces": len(face_labels),
        }
        shapes = []
        for l in np.unique(face_labels):
            if l == 0:
                continue

            s = {'label': str(l), 'metadata': {}, "shape_type": "face", "face_ids": np.where(face_labels == l)[0].tolist()}
            shapes.append(s)

        label['shapes'] = shapes

        return label



class LabelmeHelper:
    _shape_keys = ['label', 'points', 'group_id', 'shape_type', 'flags']
    _shape_type_keys = ['rectangle', 'polygon']

    @staticmethod
    def read_json_data(fp) -> Dict[str, Any]:
        with open(fp, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    @staticmethod
    def write_json_data(data, fp) -> None:
        suffix = data['imagePath'].rsplit('.')[-1]
        fn = os.path.split(fp)[-1].rsplit('.', maxsplit=1)[0] + '.' + suffix
        data['imagePath'] = fn
        with open(fp, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def update_file_name(json_fps:List[str], log:bool=False)->None:
        """
        Update and overwrite the filename variable in json files.
        :param json_fps: a list of json file paths
        :param log:
        """
        for json_fp in json_fps:
            fn = os.path.split(json_fp)[-1].rsplit('.', 1)[0]
            try:
                with open(json_fp, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                img_format = data['imagePath'].rsplit('.')[-1]
                data['imagePath'] = fn + '.' + img_format

                with open(json_fp, 'w', encoding='utf-8') as f:
                    json.dump(data, f)

            except:
                print('Error', json_fp)

    @staticmethod
    def convert_to_rectangle(shape:Dict)->Dict:
        """
        :param shape: a shape has
        :return:
        """
        new_shape = shape.copy()
        new_shape['shape_type'] = 'rectangle'

        xs = [p[0] for p in shape['points']]
        ys = [p[1] for p in shape['points']]
        new_shape['points'] = [[min(xs), min(ys)], [max(xs), max(ys)]]
        return new_shape

    ## read only
    @staticmethod
    def get_statistics(json_fps, log=True)->Dict:
        stats = {}

        for fp in json_fps:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
                shapes = data['shapes']

                for s in shapes:
                    label = s['label']
                    shape_type = s['shape_type']

                    if label not in stats:
                        stats[label] = {}

                    if shape_type not in stats[label]:
                        stats[label][shape_type] = 1
                    else:
                        stats[label][shape_type] += 1

        if log:
            #
            print('#files', len(json_fps))

            for label, shapes in stats.items():
                n = sum([i for i in shapes.values()])
                print(label, n)

            print()
            for label, shapes in stats.items():
                print(label, shapes)

        return stats

    @staticmethod
    def show_labels(image:np.ndarray, shapes:List)->None:
        for s in shapes:
            pts = np.array(s['points']).astype('i4')
            isClose = True
            color = (0, 255, 0)
            thickness = 2
            img = cv2.polylines(image, [pts], isClose, color=color, thickness=thickness)
        cv2.imshow('', img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    @staticmethod
    def get_shapes(data_shapes: List[Any], labels: List[str]) -> List[Any]:
        ret_shapes = []
        for sh in data_shapes:
            if sh['label'] in labels:
                ret_shapes.append(sh)
        return ret_shapes

    @staticmethod
    def remove_shapes(data_shapes: List[Any], labels: List[str]) -> List[Any]:
        ret_shapes = []
        for sh in data_shapes:
            if sh['label'] not in labels:
                ret_shapes.append(sh)
        return ret_shapes