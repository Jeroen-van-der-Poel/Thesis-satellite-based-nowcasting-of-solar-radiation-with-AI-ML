import numpy as np
import tensorflow as tf

# Count number of tensors in each tfrecord file
def count_tensors(files):
    count = 0
    for record in files:
        example = tf.train.Example()
        example.ParseFromString(record.numpy())
        count += len(example.features.feature)
    return count

# Count total number of records in tfrecords
def countRecords(ds: tf.data.Dataset):
    count = 0
    if tf.executing_eagerly():
        # TF v2 or v1 in eager mode
        for r in ds:
            count = count + 1
    else:
        # TF v1 in non-eager mode
        iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
        next_batch = iterator.get_next()
        with tf.compat.v1.Session() as sess:
            try:
                while True:
                    sess.run(next_batch)
                    count = count + 1
            except tf.errors.OutOfRangeError:
                pass
    return count

# Loading TFRecord dataset
def Dataset(tfr_dir, batch_size, crop = 0, prob=False,augment = True, norma_value = 1, shuffle=True):
    files = tf.io.gfile.glob(str(tfr_dir) + '/*.tfrecords')
    files = tf.data.Dataset.from_tensor_slices(files)
    if shuffle:
        # You should increase this one
        files = files.shuffle(buffer_size=len(files) // 2)

    # Read TFRecord files using parallel loading
    train_dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'), num_parallel_calls=tf.data.AUTOTUNE)
    if prob:
        train_dataset = train_dataset.map(parse_tfr_element_with_prob_simple, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        train_dataset = train_dataset.map(parse_tfr_element_simple, num_parallel_calls=tf.data.AUTOTUNE)

    # FIXME increase for better shuffling
    # If shuffle: you should increase this one
    # dataset = dataset.shuffle(buffer_size=1)
    if augment:
        train_dataset_aug = train_dataset.map(augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch the dataset and optimize loading
    train_dataset_aug = train_dataset_aug.batch(batch_size, drop_remainder=True)
    train_dataset = train_dataset.batch(batch_size,drop_remainder=True)
    train_dataset_aug = train_dataset_aug.prefetch(buffer_size=tf.data.AUTOTUNE).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE).repeat()

    return train_dataset, train_dataset_aug

# Adjust image contrast randomly within a given range
def random_contrast_augmentation(image, contrast_lower, contrast_upper,random_seed):
    tf.random.set_seed(random_seed)
    contrast_factor = tf.random.uniform([], minval=contrast_lower, maxval=contrast_upper)
    return tf.image.adjust_contrast(image, contrast_factor)

# Perform random cropping on images
def random_crop_images(target_data, label_data, crop_height, crop_width):
    target_shape = tf.shape(target_data)
    target_y = tf.random.uniform(shape=[], maxval=target_shape[1] - crop_height + 1, dtype=tf.int32)
    target_x = tf.random.uniform(shape=[], maxval=target_shape[2] - crop_width + 1, dtype=tf.int32)
    label_y = target_y
    label_x = target_x
    target_cropped = tf.image.crop_to_bounding_box(target_data, target_y, target_x, crop_height, crop_width)
    label_cropped = tf.image.crop_to_bounding_box(label_data, label_y, label_x, crop_height, crop_width)
    return target_cropped, label_cropped

# Combines flipping, contrast adjustment, and random cropping in one function
def augmentation(input_img, target_img, time_stamp):
  """Perform random augmentations"""
  # Spatial flips
  if tf.random.uniform([]) < 0.5:
    input_img = tf.reverse(input_img, axis=[1])
    target_img = tf.reverse(target_img, axis=[1])
  if tf.random.uniform([]) < 0.5:
    input_img = tf.reverse(input_img, axis=[2])
    target_img = tf.reverse(target_img, axis=[2])

  # Contrast
  random_seed = np.random.randint(1, 100)
  input_img = random_contrast_augmentation(input_img, 0.9, 1.1,random_seed)
  target_img = random_contrast_augmentation(target_img, 0.9, 1.1, random_seed)

  # Random crop and resize
  if tf.random.uniform([]) > 0.2:
    input_img,target_img = random_crop_images(input_img,target_img, 343, 225)
    input_img = tf.image.resize(input_img, size=[390, 256])
    target_img = tf.image.resize(target_img, size=[390, 256])

  return (input_img, target_img, time_stamp)

# Parse TFRecords with probability maps
def parse_tfr_element_with_prob_simple(element):
    data = {
        'window_cond': tf.io.FixedLenFeature([], tf.float32),
        'height_cond': tf.io.FixedLenFeature([], tf.float32),
        'width_cond': tf.io.FixedLenFeature([], tf.float32),
        'raw_image_cond': tf.io.FixedLenFeature([], tf.string),
        'depth_cond': tf.io.FixedLenFeature([], tf.float32),
        'window_targ': tf.io.FixedLenFeature([], tf.float32),
        'height_targ': tf.io.FixedLenFeature([], tf.float32),
        'width_targ': tf.io.FixedLenFeature([], tf.float32),
        'raw_image_targ': tf.io.FixedLenFeature([], tf.string),
        'depth_targ': tf.io.FixedLenFeature([], tf.float32),
        'prob': tf.io.FixedLenFeature([], tf.string),
        'start_date': tf.io.FixedLenFeature([], tf.string)
    }

    content = tf.io.parse_single_example(element, data)

    window_cond = content['window_cond']
    height_cond = content['height_cond']
    width_cond = content['width_cond']
    depth_cond = content['depth_cond']
    raw_image_cond = content['raw_image_cond']
    window_targ = content['window_targ']
    height_targ = content['height_targ']
    width_targ = content['width_targ']
    depth_targ = content['depth_targ']
    raw_image_targ = content['raw_image_targ']
    prob = content['prob']
    start_date = content['start_date']

    # Get our 'feature'-- our image -- and reshape it appropriately
    feature_cond = tf.io.parse_tensor(raw_image_cond, out_type=tf.float32)
    feature_cond = tf.reshape(feature_cond, shape=[window_cond, height_cond, width_cond, depth_cond])
    feature_targ = tf.io.parse_tensor(raw_image_targ, out_type=tf.float32)
    feature_targ = tf.reshape(feature_targ, shape=[window_targ, height_targ, width_targ, depth_targ])
    feature_prob = tf.io.parse_tensor(prob, out_type=tf.float32)
    feature_date = tf.io.parse_tensor(start_date, out_type=tf.string)

    return (feature_cond, feature_targ, feature_date)

# Parse TFRecords without probability maps
def parse_tfr_element_simple(element):
    data = {
        'window_cond': tf.io.FixedLenFeature([], tf.float32),
        'height_cond': tf.io.FixedLenFeature([], tf.float32),
        'width_cond': tf.io.FixedLenFeature([], tf.float32),
        'depth_cond': tf.io.FixedLenFeature([], tf.float32),
        'raw_image_cond': tf.io.FixedLenFeature([], tf.string),
        'window_targ': tf.io.FixedLenFeature([], tf.float32),
        'height_targ': tf.io.FixedLenFeature([], tf.float32),
        'width_targ': tf.io.FixedLenFeature([], tf.float32),
        'depth_targ': tf.io.FixedLenFeature([], tf.float32),
        'raw_image_targ': tf.io.FixedLenFeature([], tf.string),
        'start_date': tf.io.FixedLenFeature([], tf.string)
    }

    content = tf.io.parse_single_example(element, data)

    window_cond = content['window_cond']
    height_cond = content['height_cond']
    width_cond = content['width_cond']
    depth_cond = content['depth_cond']
    raw_image_cond = content['raw_image_cond']
    window_targ = content['window_targ']
    height_targ = content['height_targ']
    width_targ = content['width_targ']
    depth_targ = content['depth_targ']
    raw_image_targ = content['raw_image_targ']
    start_date = content['start_date']

    # Get our 'feature'-- our image -- and reshape it appropriately
    feature_cond = tf.io.parse_tensor(raw_image_cond, out_type=tf.float32)
    feature_cond = tf.reshape(feature_cond, shape=[window_cond, height_cond, width_cond, depth_cond])
    feature_targ = tf.io.parse_tensor(raw_image_targ, out_type=tf.float32)
    feature_targ = tf.reshape(feature_targ, shape=[window_targ, height_targ, width_targ, depth_targ])
    feature_date = tf.io.parse_tensor(start_date, out_type=tf.string)
    
    return (feature_cond,feature_targ, feature_date)