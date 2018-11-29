#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 00:37:33 2018

DataLoader

"""

im_s       = [30, 30, 1]   # target image size
BATCH_SIZE = 16

# image reader 
# - fnl_queue: queue with [fn l] pairs 
# Notes 
# - to resize:  image_tensor = tf.image.resize_image_with_crop_or_pad(image_tensor, HEIGHT, WIDTH)
# - how about image preprocessing?
def img_reader_jpg(fnl_queue, ch = 3, keep = False):
    fn, label = fnl_queue.dequeue()

    if keep:
        fnl_queue.enqueue([fn, label])

    img_bytes = tf.read_file(fn)
    img_u8    = tf.image.decode_jpeg(img_bytes, channels=ch) 
    img_f32   = tf.cast(img_u8, tf.float32)/256.0  
    #img_4     = tf.expand_dims(img_f32,0)
    return img_f32, label

#  load [feature, label] and enqueue to processing queue
# - sess:             tf session 
# - sess:             tf Coordinator
# - [fr_op, lr_op ]:  feature_read_op label_read_op
# - enqueue_op:       [f l] pairs enqueue op
def load_and_enqueue(sess, coord, feature_read_op, label_read_op , enqueue_op):
    i = 0
    while not coord.should_stop():
        # for testing purpose
        time.sleep(0.1)                     
        #print 'load_and_enqueue i=',i
        #i = i +1

        feature, label = sess.run([feature_read_op, label_read_op ])

        feed_dict = {feature_input: feature,
                     label_input  : label}

        sess.run(enqueue_op, feed_dict=feed_dict)


# --- TF part ---

# filenames and labels are pre-loaded
fv = tf.constant(fnames)
lv = tf.constant(ohl)

#fnl_q    = tf.FIFOQueue(len(fnames), [tf.string, tf.float32])
fnl_q    = tf.RandomShuffleQueue(len(fnames), 0, [tf.string, tf.float32])
do_enq = fnl_q.enqueue_many([fv, lv])

# reading_op: feature_read_op label_read_op 
feature_read_op, label_read_op = img_reader_jpg(fnl_q, ch = im_s[2])

# samples queue
f_s = im_s
l_s = 2
feature_input = tf.placeholder(tf.float32, shape=f_s, name='feature_input')
label_input   = tf.placeholder(tf.float32, shape=l_s, name='label_input')

#proc_q     = tf.RandomShuffleQueue(len(fnames), 0, [tf.float32, tf.float32], shapes=[f_s, l_s])
proc_q     = tf.FIFOQueue(len(fnames), [tf.float32, tf.float32], shapes=[f_s, l_s])
enqueue_op = proc_q.enqueue([feature_input, label_input])

# test: 
# - some op
img_batch, lab_batch = proc_q.dequeue_many(BATCH_SIZE)
some_op   = [img_batch, lab_batch]

# service ops
init_op   = tf.initialize_all_variables()

# let run stuff
with tf.Session() as sess:

    sess.run(init_op)
    sess.run(do_enq)

    print "fnl_q.size:", fnl_q.size().eval()
    print "proc_q.size:", proc_q.size().eval()

    # --- test thread stuff ---
    #  - fill proc_q
    coord = tf.train.Coordinator()
    t = threading.Thread(target=load_and_enqueue, args = (sess, coord, feature_read_op, label_read_op , enqueue_op))
    t.start()

    time.sleep(2.1)

    coord.request_stop()
    coord.join([t])

    print "fnl_q.size:", fnl_q.size().eval()
    print "proc_q.size:", proc_q.size().eval()

    #  - process a bit 
    ss = sess.run(some_op)
    print 'ss[0].shape', ss[0].shape 
    print ' ss[1]:\n', ss[1]

    print "fnl_q.size:", fnl_q.size().eval()
    print "proc_q.size:", proc_q.size().eval() 

print 'ok'