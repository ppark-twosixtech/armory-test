import numpy as np
from armory.logs import log
from collections import Counter


def _intersection_over_union(box_1, box_2):
    """
    Assumes each input has shape (4,) and format [y1, x1, y2, x2] or [x1, y1, x2, y2]
    """
    assert box_1[2] >= box_1[0]
    assert box_2[2] >= box_2[0]
    assert box_1[3] >= box_1[1]
    assert box_2[3] >= box_2[1]

    if all(i <= 1.0 for i in box_1[np.where(box_1 > 0)]) ^ all(
        i <= 1.0 for i in box_2[np.where(box_2 > 0)]
    ):
        log.warning("One set of boxes appears to be normalized while the other is not")

    # Determine coordinates of intersection box
    x_left = max(box_1[1], box_2[1])
    x_right = min(box_1[3], box_2[3])
    y_top = max(box_1[0], box_2[0])
    y_bottom = min(box_1[2], box_2[2])

    intersect_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    if intersect_area == 0:
        return 0

    box_1_area = (box_1[3] - box_1[1]) * (box_1[2] - box_1[0])
    box_2_area = (box_2[3] - box_2[1]) * (box_2[2] - box_2[0])

    iou = intersect_area / (box_1_area + box_2_area - intersect_area)
    assert iou >= 0
    assert iou <= 1
    return iou


def object_detection_get_tpr_mr_dr_hr(
    y_list, y_pred_list, iou_threshold=0.5, score_threshold=0.5, class_list=None
):
    """
    Helper function to compute true positive rate, disappearance rate, misclassification rate, and
    hallucinations per image as defined below:
    true positive rate: the percent of ground-truth boxes which are predicted with iou > iou_threshold,
        score > score_threshold, and the correct label
    misclassification rate: the percent of ground-truth boxes which are predicted with iou > iou_threshold,
        score > score_threshold, and the incorrect label
    disappearance rate: 1 - true_positive_rate - misclassification rate
    hallucinations per image: the number of predicted boxes per image that have score > score_threshold and
        iou(predicted_box, ground_truth_box) < iou_threshold for each ground_truth_box
    y_list (list): of length equal to the number of input examples. Each element in the list
        should be a dict with "labels" and "boxes" keys mapping to a numpy array of
        shape (N,) and (N, 4) respectively where N = number of boxes.
    y_pred_list (list): of length equal to the number of input examples. Each element in the
        list should be a dict with "labels", "boxes", and "scores" keys mapping to a numpy
        array of shape (N,), (N, 4), and (N,) respectively where N = number of boxes.
    class_list (list, optional): a list of classes, such that all predictions with labels and ground-truths
        with labels NOT in class_list are to be ignored.
    returns: a tuple of length 4 (TPR, MR, DR, HR) where each element is a list of length equal
    to the number of images.
    """

    true_positive_rate_per_img = []
    misclassification_rate_per_img = []
    disappearance_rate_per_img = []
    hallucinations_per_img = []
    for img_idx, (y, y_pred) in enumerate(zip(y_list, y_pred_list)):
        if class_list:
            # Filter out ground-truth classes with labels not in class_list
            indices_to_keep = np.where(np.isin(y["labels"], class_list))
            gt_boxes = y["boxes"][indices_to_keep]
            gt_labels = y["labels"][indices_to_keep]
        else:
            gt_boxes = y["boxes"]
            gt_labels = y["labels"]

        # initialize count of hallucinations
        num_hallucinations = 0
        num_gt_boxes = len(gt_boxes)

        # Initialize arrays that will indicate whether each respective ground-truth
        # box is a true positive or misclassified
        true_positive_array = np.zeros((num_gt_boxes,))
        misclassification_array = np.zeros((num_gt_boxes,))

        # Only consider the model's confident predictions
        conf_pred_indices = np.where(y_pred["scores"] > score_threshold)[0]
        if class_list:
            # Filter out predictions from classes not in class_list kwarg
            conf_pred_indices = conf_pred_indices[
                np.isin(y_pred["labels"][conf_pred_indices], class_list)
            ]

        # For each confident prediction
        for y_pred_idx in conf_pred_indices:
            y_pred_box = y_pred["boxes"][y_pred_idx]

            # Compute the iou between the predicted box and the ground-truth boxes
            ious = np.array([_intersection_over_union(y_pred_box, a) for a in gt_boxes])

            # Determine which ground-truth boxes, if any, the predicted box overlaps with
            overlap_indices = np.where(ious > iou_threshold)[0]

            # If the predicted box doesn't overlap with any ground-truth boxes, increment
            # the hallucination counter and move on to the next predicted box
            if len(overlap_indices) == 0:
                num_hallucinations += 1
                continue

            # For each ground-truth box that the prediction overlaps with
            for y_idx in overlap_indices:
                # If the predicted label is correct, mark that the ground-truth
                # box has a true positive prediction
                if gt_labels[y_idx] == y_pred["labels"][y_pred_idx]:
                    true_positive_array[y_idx] = 1
                else:
                    # Otherwise mark that the ground-truth box has a misclassification
                    misclassification_array[y_idx] = 1

        # Convert these arrays to binary to avoid double-counting (i.e. when multiple
        # predicted boxes overlap with a single ground-truth box)
        true_positive_rate = (true_positive_array > 0).mean()
        misclassification_rate = (misclassification_array > 0).mean()

        # Any ground-truth box that had no overlapping predicted box is considered a
        # disappearance
        disappearance_rate = 1 - true_positive_rate - misclassification_rate

        true_positive_rate_per_img.append(true_positive_rate)
        misclassification_rate_per_img.append(misclassification_rate)
        disappearance_rate_per_img.append(disappearance_rate)
        hallucinations_per_img.append(num_hallucinations)

    return (
        true_positive_rate_per_img,
        misclassification_rate_per_img,
        disappearance_rate_per_img,
        hallucinations_per_img,
        true_positive_array,
        misclassification_array,
    )


def object_detection_get_tp_fp(y_list, y_pred_list, iou_threshold=0.5, class_list=None):
    """
    Mean average precision for object detection. The mAP can be computed by taking the mean
    of the AP's across all classes. This metric is computed over all evaluation samples,
    rather than on a per-sample basis.

    y_list (list): of length equal to the number of input examples. Each element in the list
        should be a dict with "labels" and "boxes" keys mapping to a numpy array of
        shape (N,) and (N, 4) respectively where N = number of boxes.
    y_pred_list (list): of length equal to the number of input examples. Each element in the
        list should be a dict with "labels", "boxes", and "scores" keys mapping to a numpy
        array of shape (N,), (N, 4), and (N,) respectively where N = number of boxes.
    class_list (list, optional): a list of classes, such that all predictions and ground-truths
        with labels NOT in class_list are to be ignored.

    returns: a dictionary mapping each class to the average precision (AP) for the class.
    """
    # _check_object_detection_input(y_list, y_pred_list)

    # Precision will be computed at recall points of 0, 0.1, 0.2, ..., 1
    RECALL_POINTS = np.linspace(0, 1, 11)

    # Converting all boxes to a list of dicts (a list for predicted boxes, and a
    # separate list for ground truth boxes), where each dict corresponds to a box and
    # has the following keys "img_idx", "label", "box", as well as "score" for predicted boxes
    pred_boxes_list = []
    gt_boxes_list = []
    for img_idx, (y, y_pred) in enumerate(zip(y_list, y_pred_list)):
        img_labels = y["labels"].flatten()
        img_boxes = y["boxes"].reshape((-1, 4))
        for gt_box_idx in range(img_labels.flatten().shape[0]):
            label = img_labels[gt_box_idx]
            box = img_boxes[gt_box_idx]
            gt_box_dict = {"img_idx": img_idx, "label": label, "box": box}
            gt_boxes_list.append(gt_box_dict)

        for pred_box_idx in range(y_pred["labels"].flatten().shape[0]):
            pred_label = y_pred["labels"][pred_box_idx]
            pred_box = y_pred["boxes"][pred_box_idx]
            pred_score = y_pred["scores"][pred_box_idx]
            pred_box_dict = {
                "img_idx": img_idx,
                "label": pred_label,
                "box": pred_box,
                "score": pred_score,
            }
            pred_boxes_list.append(pred_box_dict)

    # Union of (1) the set of all true classes and (2) the set of all predicted classes
    set_of_class_ids = set([i["label"] for i in gt_boxes_list]) | set(
        [i["label"] for i in pred_boxes_list]
    )

    if class_list:
        # Filter out classes not in class_list
        set_of_class_ids = set(i for i in set_of_class_ids if i in class_list)

    # # Remove the class ID that corresponds to a physical adversarial patch in APRICOT
    # # dataset, if present
    # set_of_class_ids.discard(ADV_PATCH_MAGIC_NUMBER_LABEL_ID)

    # Initialize dict that will store AP for each class
    average_precisions_by_class = {}

    tp_list = []
    fp_list = []

    # Compute AP for each class
    for class_id in set_of_class_ids:

        # Build lists that contain all the predicted/ground-truth boxes with a
        # label of class_id
        class_predicted_boxes = []
        class_gt_boxes = []
        for pred_box in pred_boxes_list:
            if pred_box["label"] == class_id:
                class_predicted_boxes.append(pred_box)
        for gt_box in gt_boxes_list:
            if gt_box["label"] == class_id:
                class_gt_boxes.append(gt_box)

        # Determine how many gt boxes (of class_id) there are in each image
        num_gt_boxes_per_img = Counter([gt["img_idx"] for gt in class_gt_boxes])

        # Initialize dict where we'll keep track of whether a gt box has been matched to a
        # prediction yet. This is necessary because if multiple predicted boxes of class_id
        # overlap with a single gt box, only one of the predicted boxes can be considered a
        # true positive
        img_idx_to_gtboxismatched_array = {}
        for img_idx, num_gt_boxes in num_gt_boxes_per_img.items():
            img_idx_to_gtboxismatched_array[img_idx] = np.zeros(num_gt_boxes)

        # Sort all predicted boxes (of class_id) by descending confidence
        class_predicted_boxes.sort(key=lambda x: x["score"], reverse=True)

        # Initialize arrays. Once filled in, true_positives[i] indicates (with a 1 or 0)
        # whether the ith predicted box (of class_id) is a true positive. Likewise for
        # false_positives array
        true_positives = np.zeros(len(class_predicted_boxes))
        false_positives = np.zeros(len(class_predicted_boxes))

        # Iterating over all predicted boxes of class_id
        for pred_idx, pred_box in enumerate(class_predicted_boxes):
            # Only compare gt boxes from the same image as the predicted box
            gt_boxes_from_same_img = [
                gt_box
                for gt_box in class_gt_boxes
                if gt_box["img_idx"] == pred_box["img_idx"]
            ]

            # If there are no gt boxes in the predicted box's image that have the predicted class
            if len(gt_boxes_from_same_img) == 0:
                false_positives[pred_idx] = 1
                continue

            # Iterate over all gt boxes (of class_id) from the same image as the predicted box,
            # determining which gt box has the highest iou with the predicted box
            highest_iou = 0
            for gt_idx, gt_box in enumerate(gt_boxes_from_same_img):
                iou = _intersection_over_union(pred_box["box"], gt_box["box"])
                if iou >= highest_iou:
                    highest_iou = iou
                    highest_iou_gt_idx = gt_idx

            if highest_iou > iou_threshold:
                # If the gt box has not yet been covered
                if (
                    img_idx_to_gtboxismatched_array[pred_box["img_idx"]][
                        highest_iou_gt_idx
                    ]
                    == 0
                ):
                    true_positives[pred_idx] = 1

                    # Record that we've now covered this gt box. Any subsequent
                    # pred boxes that overlap with it are considered false positives
                    img_idx_to_gtboxismatched_array[pred_box["img_idx"]][
                        highest_iou_gt_idx
                    ] = 1
                else:
                    # This gt box was already covered previously (i.e a different predicted
                    # box was deemed a true positive after overlapping with this gt box)
                    false_positives[pred_idx] = 1
            else:
                false_positives[pred_idx] = 1

        tp_list.append(true_positives)
        fp_list.append(false_positives)

    return tp_list, fp_list


#         # Cumulative sums of false/true positives across all predictions which were sorted by
#         # descending confidence
#         tp_cumulative_sum = np.cumsum(true_positives)
#         fp_cumulative_sum = np.cumsum(false_positives)


#         # Total number of gt boxes with a label of class_id
#         total_gt_boxes = len(class_gt_boxes)

#         if total_gt_boxes > 0:
#             recalls = tp_cumulative_sum / total_gt_boxes
#         else:
#             recalls = np.zeros_like(tp_cumulative_sum)

#         precisions = tp_cumulative_sum / (tp_cumulative_sum + fp_cumulative_sum + 1e-8)

#         interpolated_precisions = np.zeros(len(RECALL_POINTS))
#         # Interpolate the precision at each recall level by taking the max precision for which
#         # the corresponding recall exceeds the recall point
#         # See http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.5766&rep=rep1&type=pdf
#         for i, recall_point in enumerate(RECALL_POINTS):
#             precisions_points = precisions[np.where(recalls >= recall_point)]
#             # If there's no cutoff at which the recall > recall_point
#             if len(precisions_points) == 0:
#                 interpolated_precisions[i] = 0
#             else:
#                 interpolated_precisions[i] = max(precisions_points)

#         # Compute mean precision across the different recall levels
#         average_precision = interpolated_precisions.mean()
#         average_precisions_by_class[int(class_id)] = np.around(
#             average_precision, decimals=2
#         )

#     return average_precisions_by_class


def object_detection_AP_per_class_test(
    y_list, y_pred_list, iou_threshold=0.5, class_list=None
):
    """
    Mean average precision for object detection. The mAP can be computed by taking the mean
    of the AP's across all classes. This metric is computed over all evaluation samples,
    rather than on a per-sample basis.

    y_list (list): of length equal to the number of input examples. Each element in the list
        should be a dict with "labels" and "boxes" keys mapping to a numpy array of
        shape (N,) and (N, 4) respectively where N = number of boxes.
    y_pred_list (list): of length equal to the number of input examples. Each element in the
        list should be a dict with "labels", "boxes", and "scores" keys mapping to a numpy
        array of shape (N,), (N, 4), and (N,) respectively where N = number of boxes.
    class_list (list, optional): a list of classes, such that all predictions and ground-truths
        with labels NOT in class_list are to be ignored.

    returns: a dictionary mapping each class to the average precision (AP) for the class.
    """
    # _check_object_detection_input(y_list, y_pred_list)

    # Precision will be computed at recall points of 0, 0.1, 0.2, ..., 1
    # RECALL_POINTS = np.linspace(0, 1, 11)
    RECALL_POINTS = np.linspace(
        0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
    )

    # Converting all boxes to a list of dicts (a list for predicted boxes, and a
    # separate list for ground truth boxes), where each dict corresponds to a box and
    # has the following keys "img_idx", "label", "box", as well as "score" for predicted boxes
    pred_boxes_list = []
    gt_boxes_list = []
    for img_idx, (y, y_pred) in enumerate(zip(y_list, y_pred_list)):
        img_labels = y["labels"].flatten()
        img_boxes = y["boxes"].reshape((-1, 4))
        for gt_box_idx in range(img_labels.flatten().shape[0]):
            label = img_labels[gt_box_idx]
            box = img_boxes[gt_box_idx]
            gt_box_dict = {"img_idx": img_idx, "label": label, "box": box}
            gt_boxes_list.append(gt_box_dict)

        for pred_box_idx in range(y_pred["labels"].flatten().shape[0]):
            pred_label = y_pred["labels"][pred_box_idx]
            pred_box = y_pred["boxes"][pred_box_idx]
            pred_score = y_pred["scores"][pred_box_idx]
            pred_box_dict = {
                "img_idx": img_idx,
                "label": pred_label,
                "box": pred_box,
                "score": pred_score,
            }
            pred_boxes_list.append(pred_box_dict)

    # Union of (1) the set of all true classes and (2) the set of all predicted classes
    set_of_class_ids = set([i["label"] for i in gt_boxes_list]) | set(
        [i["label"] for i in pred_boxes_list]
    )

    if class_list:
        # Filter out classes not in class_list
        set_of_class_ids = set(i for i in set_of_class_ids if i in class_list)

    # Remove the class ID that corresponds to a physical adversarial patch in APRICOT
    # dataset, if present
    # set_of_class_ids.discard(ADV_PATCH_MAGIC_NUMBER_LABEL_ID)

    # Initialize dict that will store AP for each class
    average_precisions_by_class = {}

    # Compute AP for each class
    for class_id in set_of_class_ids:

        # Build lists that contain all the predicted/ground-truth boxes with a
        # label of class_id
        class_predicted_boxes = []
        class_gt_boxes = []
        for pred_box in pred_boxes_list:
            if pred_box["label"] == class_id:
                class_predicted_boxes.append(pred_box)
        for gt_box in gt_boxes_list:
            if gt_box["label"] == class_id:
                class_gt_boxes.append(gt_box)

        # Determine how many gt boxes (of class_id) there are in each image
        num_gt_boxes_per_img = Counter([gt["img_idx"] for gt in class_gt_boxes])

        # Initialize dict where we'll keep track of whether a gt box has been matched to a
        # prediction yet. This is necessary because if multiple predicted boxes of class_id
        # overlap with a single gt box, only one of the predicted boxes can be considered a
        # true positive
        img_idx_to_gtboxismatched_array = {}
        for img_idx, num_gt_boxes in num_gt_boxes_per_img.items():
            img_idx_to_gtboxismatched_array[img_idx] = np.zeros(num_gt_boxes)

        # Sort all predicted boxes (of class_id) by descending confidence
        class_predicted_boxes.sort(key=lambda x: x["score"], reverse=True)

        # Initialize arrays. Once filled in, true_positives[i] indicates (with a 1 or 0)
        # whether the ith predicted box (of class_id) is a true positive. Likewise for
        # false_positives array
        true_positives = np.zeros(len(class_predicted_boxes))
        false_positives = np.zeros(len(class_predicted_boxes))

        # Iterating over all predicted boxes of class_id
        for pred_idx, pred_box in enumerate(class_predicted_boxes):
            # Only compare gt boxes from the same image as the predicted box
            gt_boxes_from_same_img = [
                gt_box
                for gt_box in class_gt_boxes
                if gt_box["img_idx"] == pred_box["img_idx"]
            ]

            # If there are no gt boxes in the predicted box's image that have the predicted class
            if len(gt_boxes_from_same_img) == 0:
                false_positives[pred_idx] = 1
                continue

            # Iterate over all gt boxes (of class_id) from the same image as the predicted box,
            # determining which gt box has the highest iou with the predicted box
            highest_iou = 0
            for gt_idx, gt_box in enumerate(gt_boxes_from_same_img):
                iou = _intersection_over_union(pred_box["box"], gt_box["box"])
                if iou >= highest_iou:
                    highest_iou = iou
                    highest_iou_gt_idx = gt_idx

            if highest_iou > iou_threshold:
                # If the gt box has not yet been covered
                if (
                    img_idx_to_gtboxismatched_array[pred_box["img_idx"]][
                        highest_iou_gt_idx
                    ]
                    == 0
                ):
                    true_positives[pred_idx] = 1

                    # Record that we've now covered this gt box. Any subsequent
                    # pred boxes that overlap with it are considered false positives
                    img_idx_to_gtboxismatched_array[pred_box["img_idx"]][
                        highest_iou_gt_idx
                    ] = 1
                else:
                    # This gt box was already covered previously (i.e a different predicted
                    # box was deemed a true positive after overlapping with this gt box)
                    false_positives[pred_idx] = 1
            else:
                false_positives[pred_idx] = 1

        # Cumulative sums of false/true positives across all predictions which were sorted by
        # descending confidence
        tp_cumulative_sum = np.cumsum(true_positives)
        fp_cumulative_sum = np.cumsum(false_positives)

        # Total number of gt boxes with a label of class_id
        total_gt_boxes = len(class_gt_boxes)

        if total_gt_boxes > 0:
            recalls = tp_cumulative_sum / total_gt_boxes
        else:
            recalls = np.zeros_like(tp_cumulative_sum)

        precisions = tp_cumulative_sum / (tp_cumulative_sum + fp_cumulative_sum + 1e-8)

        interpolated_precisions = np.zeros(len(RECALL_POINTS))
        # Interpolate the precision at each recall level by taking the max precision for which
        # the corresponding recall exceeds the recall point
        # See http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.5766&rep=rep1&type=pdf
        for i, recall_point in enumerate(RECALL_POINTS):
            precisions_points = precisions[np.where(recalls >= recall_point)]
            # If there's no cutoff at which the recall > recall_point
            if len(precisions_points) == 0:
                interpolated_precisions[i] = 0
            else:
                interpolated_precisions[i] = max(precisions_points)

        # Compute mean precision across the different recall levels
        average_precision = interpolated_precisions.mean()
        # average_precisions_by_class[int(class_id)] = np.around(
        #     average_precision, decimals=2
        # )
        average_precisions_by_class[int(class_id)] = average_precision

    return average_precisions_by_class
