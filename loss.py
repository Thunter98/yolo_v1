import tensorflow as tf
import numpy as np
from utils import iou

def yolo_loss(predict,
              labels,
              each_object_num,
              num_classes,
              boxes_per_cell,
              cell_size,
              input_width,
              input_height,
              coord_scale,
              object_scale,
              noobject_scale,
              class_scale
              ):
  '''
  Args:
    predict: 3 - D tensor [cell_size, cell_size, num_classes + 5 * boxes_per_cell]
    labels: 2-D list [object_num, 5] (xcenter (Absolute coordinate), ycenter (Absolute coordinate), w (Absolute coordinate), h (Absolute coordinate), class_num)
    each_object_num: 해당 오브젝트에 대한 number
    num_classes: 예측된 class의 수
    boxes_per_cell: 하나의 box당 몇개의 cell이 있는지
    cell_size: 각 cell size
    input_width : 원본 이미지의 너비
    input_height : 원본 이미지의 높이
    coord_scale : coordination의 coefficient
    object_scale : 오브젝트가 있는 cell의 coefficient
    noobject_scale : 오브젝트가 없는 cell의 coefficient
    class_scale : 자유도를 높이기 위해 추가적인 람다를 적용한 class의 coefficient
  Returns:
    total_loss: coord_loss  + object_loss + noobject_loss + class_loss
    coord_loss
    object_loss
    noobject_loss
    class_loss
  '''

  #coordinate vector 값 parsing 
  predict_boxes = predict[:, :, num_classes + boxes_per_cell:]
  predict_boxes = tf.reshape(predict_boxes, [cell_size, cell_size, boxes_per_cell, 4])

  #coordinate 절대값 예측
  pred_xcenter = predict_boxes[:, :, :, 0]
  pred_ycenter = predict_boxes[:, :, :, 1]
  pred_sqrt_w = tf.sqrt(tf.minimum(input_width * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 2])))
  pred_sqrt_h = tf.sqrt(tf.minimum(input_height * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 3])))
  pred_sqrt_w = tf.cast(pred_sqrt_w, tf.float32)
  pred_sqrt_h = tf.cast(pred_sqrt_h, tf.float32)

  #parse label
  labels = np.array(labels)
  labels = labels.astype('float32')
  label = labels[each_object_num, :]
  xcenter = label[0]
  ycenter = label[1]
  sqrt_w = tf.sqrt(label[2])
  sqrt_h = tf.sqrt(label[3])

  #YOLO가 예측한 bounding box와 정답 bounding box의 iou 계산
  iou_predict_truth = iou(predict_boxes, label[0:4])

  #iou 계산을 토대로 best_box_mak 찾기
  I = iou_predict_truth
  max_I = tf.reduce_max(I, 2, keepdims=True)
  best_box_mask = tf.cast((I >= max_I), tf.float32)

  #오브젝트 loss 
  C = iou_predict_truth
  pred_C = predict[:, :, num_classes:num_classes + boxes_per_cell]

  #class loss 
  P = tf.one_hot(tf.cast(label[4], tf.int32), num_classes, dtype=tf.float32)
  pred_P = predict[:, :, 0:num_classes]

  #오브젝트가 존재한 cell 찾아 mask map 생성
  object_exists_cell = np.zeros([cell_size, cell_size, 1])
  object_exists_cell_i, object_exists_cell_j = int(cell_size * ycenter / input_height), int(cell_size * xcenter / input_width)
  object_exists_cell[object_exists_cell_i][object_exists_cell_j] = 1

  #coord_loss
  coord_loss = (tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_xcenter - xcenter) / (input_width / cell_size)) +
                tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_ycenter - ycenter) / (input_height / cell_size)) +
                tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_sqrt_w - sqrt_w)) / input_width +
                tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_sqrt_h - sqrt_h)) / input_height ) \
               * coord_scale

  #object_loss
  object_loss = tf.nn.l2_loss(object_exists_cell * best_box_mask * (pred_C - C)) * object_scale

  #noobject_loss
  noobject_loss = tf.nn.l2_loss((1 - object_exists_cell) * (pred_C)) * noobject_scale

  #class loss
  class_loss = tf.nn.l2_loss(object_exists_cell * (pred_P - P)) * class_scale

  #sum every loss
  total_loss = coord_loss + object_loss + noobject_loss + class_loss

  return total_loss, coord_loss, object_loss, noobject_loss, class_loss