# @author: Nikolaus Hofer
# @author: Alexander Maletzky
# Based on: https://github.com/kaanakan/object_detection_confusion_matrix
# @date:    2020-10-01
# Confusion matrix calculation and kex performance indicators for object detection

import copy
import numpy as np
import pandas as pd


def box_iou_calc(boxes1, boxes2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (xtl, ytl, xbr, ybr) format.
    Arguments:
        boxes1 (Array[N, 4])
        boxes2 (Array[M, 4])
    Returns:
        iou (Array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    This implementation is taken from the above link and changed so that it only uses numpy.
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = np.prod(np.clip(rb - lt, a_min=0, a_max=None), 2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


class RecognitionAnnotationMatcher:

    def __init__(self, CONF_THRESHOLD: float = 0.25, IOU_THRESHOLD: float = 0.5):
        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.IOU_THRESHOLD = IOU_THRESHOLD

    def match_image(self, recognitions: pd.DataFrame, annotations: pd.DataFrame, conf: str = 'conf') -> list:
        """
        Match recognitions and ground truth annotations in a single image.
        Arguments:
            recognitions:   DataFrame with recognition results. See function `recognitions_to_numpy()` for details.
            annotations:    DataFrame with ground truth annotations. See function `annotations_to_numpy()` for details.
            conf:           Optional, name of the column in `recognitions` containing the confidence scores.
                            See function `recognitions_to_numpy()` for details.
        Returns:
            List with `K` elements, where `K` is the number of matched recognition-annotation pairs. Each element is a
            triple (recognition_id, label_id, and IoU).
        """

        if isinstance(recognitions, pd.DataFrame):
            recognitions = recognitions_to_numpy(recognitions, conf=conf)
        if isinstance(annotations, pd.DataFrame):
            annotations = annotations_to_numpy(annotations)
        
        matches = []
        if len(annotations) == 0 or len(recognitions) == 0:
            return matches

        recognitions = recognitions[recognitions[:, 4] >= self.CONF_THRESHOLD]

        all_ious = box_iou_calc(annotations[:, :4], recognitions[:, :4])
        want_idx = np.where(all_ious >= self.IOU_THRESHOLD)

        all_matches = np.empty((len(want_idx[0]), 3), dtype=all_ious.dtype)
        all_matches[:, 0] = want_idx[0]
        all_matches[:, 1] = want_idx[1]
        all_matches[:, 2] = all_ious[want_idx]

        if len(all_matches) > 0:  # if there is match
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]
            all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]
            all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]

        for i, label in enumerate(annotations):
            mask = all_matches[:, 0] == i
            if mask.sum() == 1:
                recognition_index = int(all_matches[mask, 1][0])
                matches.append((int(recognitions[recognition_index][-1]), int(annotations[i][-1]),
                                all_matches[mask, 2][0]))

        return matches

    def match_images(self, recognitions: pd.DataFrame, annotations: pd.DataFrame,
                     image_ids: list, image_id: str = 'image_id', **kwargs) -> pd.DataFrame:
        results = []
        for iid in image_ids:
            image_annotations = annotations[annotations[image_id] == iid]
            image_detections = recognitions[recognitions[image_id] == iid]
            results += self.match_image(image_detections, image_annotations, **kwargs)

        return pd.DataFrame(results, columns=['recognition_id', 'annotation_id', 'iou'])


def bbox_to_numpy(df: pd.DataFrame) -> np.ndarray:
    if all(c in df.columns for c in ('xtl', 'xbr', 'ytl', 'ybr')):
        return df[['xtl', 'ytl', 'xbr', 'ybr']].values
    else:
        bbox = np.array([eval(v) for v in df['bbox']])
        bbox[:, 0] -= bbox[:, 2] * 0.5
        bbox[:, 1] -= bbox[:, 3] * 0.5
        bbox[:, 2] += bbox[:, 0]
        bbox[:, 3] += bbox[:, 1]
        return bbox


def annotations_to_numpy(annotations: pd.DataFrame):
    """
    Convert a DataFrame with ground truth labels (annotations) into a Numpy array.
    Arguments:
        annotations: DataFrame with ground truth labels, of length M. Must have columns `labels_id` (unless None),
                    and either "bbox" or all of "xtl", "ytl", "xbr", "ybr". "bbox" is assumed to contain bounding boxes
                    in the `(x_center, y_center, width, height)` format native to YOLO.
    Returns:
        Float array of shape `(N, 6)`, where each row consists of bounding-box coordinates (xtl, ytl, xbr, ybr),
        confidence score, and annotation ID.
    """
    out = np.empty((len(annotations), 5), dtype=np.float64)
    out[:, :4] = bbox_to_numpy(annotations)
    out[:, 4] = annotations.index
    return out


def recognitions_to_numpy(recognitions: pd.DataFrame, conf: str) -> np.ndarray:
    """
    Convert a DataFrame with recognition results into a Numpy array.
    Arguments:
        recognitions: DataFrame with recognitions, of length N.
                    Must have columns `conf` (unless None), `detection_id` (unless None), and either "bbox" or all of
                    "xtl", "ytl", "xbr", "ybr". "bbox" is assumed to contain bounding boxes in the
                    `(x_center, y_center, width, height)` format native to YOLO.
        conf:       Optional, name of the column in `recognitions` containing the confidence scores.
                    If None, all confidences are set to 1.
    Returns:
        Float array of shape `(N, 6)`, where each row consists of bounding-box coordinates (xtl, ytl, xbr, ybr),
        confidence score, and recognition ID.
    """
    out = np.empty((len(recognitions), 6), dtype=np.float64)
    out[:, :4] = bbox_to_numpy(recognitions)
    out[:, 4] = 1. if conf is None else recognitions[conf].values
    out[out[:, 4] > 1, 4] /= 100
    out[:, 5] = recognitions.index
    return out


class Statistics:
    
    def __init__(self, num_classes: int, CONF_THRESHOLD: float = 0.25, ap_method: str = 'interp'):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes
        # Custom _tp, _fp, _fn
        self._tp = []
        for i in range(num_classes):
            self._tp.append([])
        self._fp = copy.deepcopy(self._tp)
        self._fn = copy.deepcopy(self._tp)
        self._conf = copy.deepcopy(self._tp)
        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.ap_method = ap_method

    def process_data(self, recognitions: pd.DataFrame, annotations: pd.DataFrame, matches: pd.DataFrame,
                     conf: str = 'conf', cls: str = 'cat_id'):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            recognitions (Array[N, 6]), x1, y1, x2, y2, _conf, class, id
            annotations (Array[M, 5]), class, x1, y1, x2, y2, id
        Returns:
            None, updates confusion matrix accordingly
        """

        # False Negatives
        for i, row in annotations[~annotations.index.isin(matches['annotation_id'])].iterrows():
            class_id = int(row[cls])
            if class_id < self.num_classes:
                self.matrix[class_id, -1] += 1
                self._fn[class_id] += [1]

        # False Positives
        for i, row in recognitions[(~recognitions.index.isin(matches['recognition_id'])) & (recognitions[conf] >= self.CONF_THRESHOLD)].iterrows():
            class_id = int(row[cls])
            if class_id < self.num_classes:
                self.matrix[self.num_classes, class_id] += 1
                self._fp[class_id] += [1]
                self._tp[class_id] += [0]
                self._conf[class_id] += [row[conf]]

        for i, row in matches.iterrows():
            det_row = recognitions.loc[row['recognition_id']]
            ann_row = annotations.loc[row['annotation_id']]
            ann_class = int(ann_row[cls])
            det_class = int(det_row[cls])
            if ann_class < self.num_classes and det_class < self.num_classes:
                self.matrix[ann_class, det_class] += 1
                if ann_class == det_class:
                    # Custom _tp,_fn,_fp
                    self._tp[ann_class] += [1]
                    self._fp[ann_class] += [0]
                    self._conf[ann_class] += [det_row[conf]]
                else:
                    # Custom _tp,_fn,_fp
                    self._fp[det_class] += [1]
                    self._tp[det_class] += [0]
                    self._conf[det_class] += [det_row[conf]]
                    self._fn[ann_class] += [1]

    def get_confusion_matrix(self):
        return self.matrix

    def get_confusion_matrix_normalized(self):
        mat_temp = self.matrix.copy()
        mat_norm = mat_temp.astype('float') / mat_temp.sum(axis=1)[:, np.newaxis]
        return mat_norm

    def get_precision_per_class(self):
        return self.get_precision_recall_per_class()[0]

    def get_recall_per_class(self):
        return self.get_precision_recall_per_class()[1]

    def get_precision_recall_per_class(self):
        precision = []
        recall = []
        for class_id in range(self.num_classes):
            # groundtruth: [x, :]
            tp = sum(self._tp[class_id])
            fn = sum(self._fn[class_id])
            fp = sum(self._fp[class_id])
            with np.errstate(divide='ignore'):
                precision.append((tp / (tp + fp) if (tp + fp) > 0 else 0))
                recall.append((tp / (tp + fn) if (tp + fn) > 0 else 0))
        precision = np.where(np.isnan(precision), 0, precision)
        recall = np.where(np.isnan(recall), 0, recall)
        return precision, recall

    def get_average_precision_by_class(self):
        ap = []
        for class_id in range(self.num_classes):
            gt_count = sum(self._tp[class_id]) + sum(self._fn[class_id])
            tp, fp, conf = self._sort_tp_fp_by_conf(class_id)
            fpc = (1 - tp).cumsum(0)
            tpc = tp.cumsum(0)
            rec = tpc / float(gt_count + 1e-16)
            prec = tpc / np.maximum(tpc + fpc, np.finfo(np.float64).eps)
            ap.append(self._compute_ap(rec, prec)[0])

        return np.array(ap)

    def get_mean_average_precision(self):
        return np.mean(self.get_average_precision_by_class())

    def get_tp(self):
        return [sum(array) for array in self._tp]

    def get_fp(self):
        return [sum(array) for array in self._fp]

    def get_fn(self):
        return [sum(array) for array in self._fn]

    def _sort_tp_fp_by_conf(self, clas):
        indx = np.argsort(-np.array(self._conf[clas]))
        tp = np.array(self._tp[clas])[indx]
        fp = np.array(self._fp[clas])[indx]
        conf = np.array(self._conf[clas])[indx]
        return tp, fp, conf

    def _compute_ap(self, recall, precision):
        """ Compute the average precision, given the recall and precision curves
        Arguments
            recall:    The recall curve (list)
            precision: The precision curve (list)
        Returns
            Average precision, precision curve, recall curve
        """

        # Append sentinel values to beginning and end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        # Compute the precision envelope
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # Integrate area under curve
        if self.ap_method == 'interp':
            x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
        else:  # 'continuous'
            i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

        return ap, mpre, mrec

    def get_performance_metrics(self, class_names=None):
        import pandas as pd
        if class_names is None:
            class_names = [str(i) for i in range(self.num_classes)]
        result_df = pd.DataFrame(
            np.array([
                class_names,
                self.get_tp(),
                self.get_fp(),
                self.get_fn(),
                np.round_(self.get_precision_per_class() * 100, decimals=2),
                np.round_(self.get_recall_per_class() * 100, decimals=2),
                np.round_(self.get_average_precision_by_class() * 100, decimals=2)
            ]).T, columns=["Class", "TP", "FP", "FN", "Precision", "Recall", "AP"])
        result_df.set_index("Class", inplace=True)
        return result_df


_CLASSES = ['01_01', '01_02', '01_03', '01_04', '01_05', '01_06', '01_07', '01_08', '01_09', '01_10',
            '01_11', '01_12', '01_13', '01_14', '01_15', '01_16', '01_17', '01_18', '01_19', '01_20',
            '01_21', '01_22', '01_23', '02_01', '02_02', '02_03', '02_04', '02_05', '02_06', '03_01',
            '03_02', '04_01', '04_02', '04_03', '04_04', '04_05', '05_01', '05_02', '05_03', '05_04',
            '05_05', '05_06', '05_07', '05_08', '06_01', '06_02', '06_03', '07_01', '07_02', '07_03',
            '07_04', '07_05', '07_06', '07_07', '07_08', '07_09', '07_10', '07_11', '08_01', '08_02']
_CATEGORIES = ['01', '02', '03', '04', '05', '06', '07', '08']


def evaluate(recognitions: pd.DataFrame, annotations: pd.DataFrame, conf: str = 'conf', pred: str = 'class_id',
             iou_threshold: float = 0.5, conf_threshold: float = 0.25, discard_disagreements: bool = False,
             ignore_unusual: bool = True, area_range=None, ap_method: str = 'interp') -> (pd.DataFrame, pd.DataFrame):
    """
    Evaluate recognition results w.r.t. ground truth annotations.

    Arguments:
        recognitions: DataFrame with recognition results.
        annotations: DataFrame with ground-truth annotations.
        conf: Name of the column in `recognitions` containing the recognition confidence score.
        pred: Name of the column in `recognitions` containing the predicted traffic signs categories or classes.
        iou_threshold: IoU threshold for matching recognitions and ground-truth annotations.
        conf_threshold: Confidence threshold. Only recognitions whose confidence score lies above this threshold are
                    considered.
        discard_disagreements: If column `pred` contains traffic sign classes, but `recognitions` also has a column
                    "cat_id" with traffic sign categories: Whether to discard all recognitions where the category of
                    the class in `pred` does not agree with the category in "cat_id".
        ignore_unusual: If column `pred` contains traffic sign classes: Whether to ignore ground truth annotations
                    where at least one of the attributes "unusual_sign", "crossed_out", "multiple_signs_visible" or
                    "caption" is set to True.
        area_range: Ground truth area range as a pair `(min, max)`, or None. If not None, all ground truth annotations
                    whose total area, in pixels, lies outside the closed interval `[min, max]`.
        ap_method: The method used for computing average precision. Should be set to "interp" to employ MS-COCO's
                    101-point interpolation method and reproduce the results from the paper.
    Returns:
        Pair `(matches, metrics)`, where `matches` is a DataFrame containing the found matches between recognitions
        and annotations, including their IoU, and `metrics` is a DataFrame with per-class performance metrics.
    """

    classes = _CLASSES

    assert recognitions.index.is_unique
    assert annotations.index.is_unique
    assert 'image_id' in recognitions.columns
    assert conf is None or conf in recognitions.columns
    assert pred in recognitions.columns
    assert 'image_id' in annotations.columns

    if discard_disagreements and pred != 'cat_id' and 'cat_id' in recognitions.columns:
        if recognitions['cat_id'].dtype.kind == 'i':
            cat = '0' + recognitions['cat_id'].astype('str')
        else:
            cat = recognitions['cat_id']
        df = recognitions[recognitions[pred].str[:2] == cat].copy()
    else:
        df = recognitions.copy()

    cols = ['class_id', 'xtl', 'ytl', 'xbr', 'ybr', 'annotation_id', 'image_id', 'unusual_sign', 'crossed_out',
            'multiple_signs_visible', 'caption']
    annotations = annotations[[c for c in cols if c in annotations.columns]].copy()

    if df[pred].dtype.kind == 'O' and (df[pred].str.len() == 5).all():
        # evaluate whole pipeline
        all_classes = annotations['class_id'].unique()
        mapping = {k: i for i, k in enumerate(classes)}
        mapping.update({k: len(classes) for k in all_classes if k not in classes})
        # the last class contains all classes to be ignored
        df['class_id'] = df[pred].replace(mapping)
        annotations['class_id'] = annotations['class_id'].replace(mapping).astype(annotations['xtl'].dtype)
        if ignore_unusual:
            # set true class of unusual signs to "other"
            cols = [c for c in ('crossed_out', 'unusual_sign', 'multiple_signs_visible', 'caption')
                    if c in annotations.columns]
            mask = (annotations[cols] == True).any(axis=1)
            annotations.loc[mask, 'class_id'] = len(classes)
        pred = 'class_id'
    else:
        # evaluate detector only
        classes = _CATEGORIES
        if df[pred].dtype.kind == 'i':
            df['cat_id'] = df[pred] - 1
        else:
            df['cat_id'] = df[pred].replace({k: i for i, k in enumerate(classes)})
        annotations['cat_id'] = annotations['class_id'].str[:2].replace({k: i for i, k in enumerate(classes)}) \
            .astype(annotations['xtl'].dtype)
        pred = 'cat_id'

    if area_range is not None:
        area = (annotations['xbr'] - annotations['xtl']) * (annotations['ybr'] - annotations['ytl'])
        annotations.loc[(area < area_range[0]) | (area > area_range[1]), pred] = len(classes)

    all_imgs = set(annotations['image_id'])
    all_imgs.update(df['image_id'])

    matcher = RecognitionAnnotationMatcher(IOU_THRESHOLD=iou_threshold, CONF_THRESHOLD=conf_threshold)
    matches = matcher.match_images(df, annotations, list(all_imgs), conf=conf)

    stats = Statistics(len(classes), CONF_THRESHOLD=conf_threshold, ap_method=ap_method)
    stats.process_data(df, annotations, matches, conf=conf, cls=pred)

    metrics = pd.DataFrame(
        data=dict(
            TP=stats.get_tp(),
            FP=stats.get_fp(),
            FN=stats.get_fn(),
            Precision=stats.get_precision_per_class(),
            Recall=stats.get_recall_per_class(),
            AP=stats.get_average_precision_by_class()
        ),
        index=classes
    )

    return matches, metrics
