import os
import shutil
import sys
import time
import random
import tensorflow as tf


from loader import *
from GanModel import CycleGANModel

import cv2


tf.app.flags.DEFINE_string('data_path', '../data/horse2zebra', 'Directory path to read the data files')
tf.app.flags.DEFINE_string('checkpoint_path', 'model', 'Directory path to save checkpoint files')

tf.app.flags.DEFINE_boolean('train_continue', False, 'flag for continue training from previous checkpoint')
tf.app.flags.DEFINE_boolean('valid_only', False, 'flag for validation only. this will make train_continue flag ignored')

tf.app.flags.DEFINE_integer('batch_size', 1, 'mini-batch size for training')
tf.app.flags.DEFINE_float('lr_disc', 2e-4, 'initial learning rate for discriminator')
tf.app.flags.DEFINE_float('lr_gen', 2e-4, 'initial learning rate for generator')
tf.app.flags.DEFINE_integer('n_epoch', 20, '# of epochs for maintaining learning rate')
tf.app.flags.DEFINE_integer('n_epoch_for_decay', 20, '# of epochs for decaying learning rate')
tf.app.flags.DEFINE_integer('train_log_interval', 100, 'step interval for triggering print logs of train')
tf.app.flags.DEFINE_integer('valid_log_interval', 1000, 'step interval for triggering validation')

FLAGS = tf.app.flags.FLAGS

print("Learning rate for discriminator = %e" % FLAGS.lr_disc)
print("Learning rate for generator = %e" % FLAGS.lr_gen)


class GanLearner:
    def __init__(self):
        self.sess = tf.Session()
        self.batch_size = FLAGS.batch_size
        self.model = CycleGANModel(batch_size=self.batch_size)

        self.train_loader = Loader(data_path=os.path.join(FLAGS.data_path, "train"), batch_size=self.batch_size)
        self.valid_loader = Loader(data_path=os.path.join(FLAGS.data_path, "valid"), batch_size=self.batch_size)

        self.max_pool_size = 50
        self.fake_image_pool_a = []
        self.fake_image_pool_b = []

        self.epoch_counter = 1
        self.lr_gen = FLAGS.lr_gen
        self.lr_disc = FLAGS.lr_disc
        self.n_epoch = FLAGS.n_epoch
        self.n_epoch_for_decay = FLAGS.n_epoch_for_decay
        self.train_log_interval = FLAGS.train_log_interval
        self.valid_log_interval = FLAGS.valid_log_interval

        self.keep_prob = 0.7

        self.train_continue = FLAGS.train_continue or FLAGS.valid_only
        self.checkpoint_dirpath = FLAGS.checkpoint_path
        self.checkpoint_filepath = os.path.join(self.checkpoint_dirpath, 'model.ckpt')
        self.log_dirpath = "log"

        if not self.train_continue and os.path.exists(self.checkpoint_dirpath):
            shutil.rmtree(self.log_dirpath, ignore_errors=True)
            shutil.rmtree(self.checkpoint_dirpath, ignore_errors=True)
            shutil.rmtree("imgs", ignore_errors=True)
        if not os.path.exists(self.checkpoint_dirpath):
            os.makedirs(self.checkpoint_dirpath)
            os.makedirs("imgs")

        self.train_summary_writer = tf.summary.FileWriter(
            os.path.join(self.log_dirpath, 'train'),
            self.sess.graph,
        )

        self.valid_summary_writer = tf.summary.FileWriter(
            os.path.join(self.log_dirpath, 'valid'),
            self.sess.graph
        )

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=3)

        if self.train_continue:
            print("======== Restoring from saved checkpoint ========")
            save_path = self.checkpoint_dirpath
            ckpt = tf.train.get_checkpoint_state(save_path)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                print("======>" + ckpt.model_checkpoint_path)
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)

                # Reset log of steps after model is saved.
                last_step = self.model.global_step_disc.eval(self.sess)
                session_log = tf.SessionLog(status=tf.SessionLog.START)
                self.train_summary_writer.add_session_log(session_log, last_step+1)
                self.valid_summary_writer.add_session_log(session_log, last_step+1)
        return


    def get_fake_image_from_pool(self, fake_image, pool):
        if len(pool) < self.max_pool_size:
            pool.append(fake_image)
            return fake_image
        else:
            p = random.random()
            if p > 0.2:
                rand_idx = random.randint(0, len(pool)-1)
                tmp = pool[rand_idx]
                pool[rand_idx] = fake_image
                return tmp
            else:
                return fake_image

    def save_images(self, epoch):

        # self.valid_loader.reset()
        self.train_loader.reset()
        dir_name = "imgs/" + str(epoch)
        os.makedirs(dir_name)
        for idx in range(10):
            batch_data_a = self.train_loader.get_batch('A')
            batch_data_b = self.train_loader.get_batch('B')
            sess_output = self.sess.run(
                fetches=[
                    self.model.fake_image_a_t,
                    self.model.fake_image_b_t,
                    self.model.cyc_image_a_t,
                    self.model.cyc_image_b_t,
                ],
                feed_dict={
                    self.model.input_image_a_ph: batch_data_a.images,
                    self.model.input_image_b_ph: batch_data_b.images,
                    self.model.batch_size_ph: self.batch_size,
                }
            )
            fake_image_a = ((sess_output[0][0]+1)*127.5).astype(np.uint8)
            fake_image_b = ((sess_output[1][0]+1)*127.5).astype(np.uint8)
            cyc_image_a = ((sess_output[2][0]+1)*127.5).astype(np.uint8)
            cyc_image_b = ((sess_output[3][0]+1)*127.5).astype(np.uint8)
            image_a = np.hstack((batch_data_a.images[0], fake_image_b, cyc_image_a))
            image_b = np.hstack((batch_data_b.images[0], fake_image_a, cyc_image_b))
            cv2.imwrite(dir_name + "/sample_A_" + str(idx) + ".jpg", image_a)
            cv2.imwrite(dir_name + "/sample_B_" + str(idx) + ".jpg", image_b)

    def train(self):
        self.train_loader.reset()

        cur_step = 0
        while True:
            start_time = time.time()
            batch_data_a = self.train_loader.get_batch('A')
            batch_data_b = self.train_loader.get_batch('B')
            if batch_data_a is None or batch_data_b is None:
                print('%d epoch complete' % self.epoch_counter)
                self.train_loader.reset()

                # Save the variables to disk.
                save_path = self.saver.save(
                    self.sess,
                    self.checkpoint_filepath,
                    global_step=cur_step
                )
                print("Model saved in file: %s" % save_path)

                self.save_images(self.epoch_counter)

                self.epoch_counter += 1
                if self.epoch_counter > self.n_epoch:
                    over_epoch = self.epoch_counter - self.n_epoch
                    self.lr_disc = (self.n_epoch_for_decay - over_epoch) / self.n_epoch_for_decay * FLAGS.lr_disc
                    self.lr_gen = (self.n_epoch_for_decay - over_epoch) / self.n_epoch_for_decay * FLAGS.lr_gen
                print("\t===> current learning rate = %f, %f" % (self.lr_disc, self.lr_gen))
                if self.epoch_counter == (self.n_epoch + self.n_epoch_for_decay):
                    break
                continue

            # Step 1. train generator from A to B
            sess_output = self.sess.run(
                fetches=[
                    self.model.train_op_gen_a,
                    self.model.fake_image_b_t,
                    self.model.gen_loss_a_summ,
                ],
                feed_dict={
                    self.model.lr_gen_ph: self.lr_gen,
                    self.model.input_image_a_ph: batch_data_a.images,
                    self.model.input_image_b_ph: batch_data_b.images,
                    self.model.batch_size_ph: self.batch_size,
                }
            )
            self.train_summary_writer.add_summary(sess_output[-1], cur_step)
            fake_pool_image_b = self.get_fake_image_from_pool(sess_output[1], self.fake_image_pool_b)

            # Step 2. train discriminator for B
            sess_output = self.sess.run(
                fetches=[
                    self.model.train_op_disc_b,
                    self.model.disc_loss_b_summ,
                ],
                feed_dict={
                    self.model.lr_disc_ph: self.lr_disc,
                    self.model.input_image_b_ph: batch_data_b.images,
                    self.model.fake_pool_image_b_ph: fake_pool_image_b,
                    self.model.batch_size_ph: self.batch_size,
                }
            )
            self.train_summary_writer.add_summary(sess_output[-1], cur_step)

            # Step 3. train generator from B to A
            sess_output = self.sess.run(
                fetches=[
                    self.model.train_op_gen_b,
                    self.model.fake_image_a_t,
                    self.model.gen_loss_b_summ,
                ],
                feed_dict={
                    self.model.lr_gen_ph: self.lr_gen,
                    self.model.input_image_a_ph: batch_data_a.images,
                    self.model.input_image_b_ph: batch_data_b.images,
                    self.model.batch_size_ph: self.batch_size,
                }
            )
            self.train_summary_writer.add_summary(sess_output[-1], cur_step)
            fake_pool_image_a = self.get_fake_image_from_pool(sess_output[1], self.fake_image_pool_a)

            # Step 4. train discriminator for A
            sess_output = self.sess.run(
                fetches=[
                    self.model.train_op_disc_a,
                    self.model.disc_loss_a_summ,
                ],
                feed_dict={
                    self.model.lr_disc_ph: self.lr_disc,
                    self.model.input_image_a_ph: batch_data_a.images,
                    self.model.fake_pool_image_a_ph: fake_pool_image_a,
                    self.model.batch_size_ph: self.batch_size,
                }
            )
            self.train_summary_writer.add_summary(sess_output[-1], cur_step)

            cur_step += 1

            if cur_step % self.train_log_interval == 0:
                print('Current step ===> %d' % cur_step)

        return

    def valid(self, step=None):
        self.valid_loader.reset()

        step_counter = 0
        accum_disc_loss_real = .0
        accum_cls_loss_real = .0

        accum_disc_loss_fake = .0
        accum_fake_cls_loss = .0
        accum_gen_loss = .0
        accum_disc_loss = .0

        accum_correct_count = .0
        accum_conf_matrix = None

        fake_image = None
        fake_label = None

        valid_batch_size = self.batch_size
        while True:
            batch_data = self.valid_loader.get_batch(valid_batch_size)
            if batch_data is None:
                # print('%d validation complete' % self.epoch_counter)
                break

            # Validation for real & fake samples concurrently.
            sess_input = [
                self.model.gen_loss,
                self.model.disc_loss_fake,
                self.model.disc_loss_real,
                self.model.disc_loss,
                self.model.correct_count,
                self.model.most_realistic_fake_image,
                self.model.most_realistic_fake_class,
            ]
            sess_output = self.sess.run(
                fetches=sess_input,
                feed_dict={
                    self.model.keep_prob_ph: self.keep_prob,
                    self.model.is_training_gen_ph: False,
                    self.model.is_training_disc_ph: False,
                    self.model.input_image_ph: batch_data.images,
                    self.model.label_cls_ph: batch_data.labels,
                    self.model.batch_size_ph: valid_batch_size,
                }
            )
            accum_gen_loss += sess_output[0]
            accum_disc_loss_fake += sess_output[1]
            accum_disc_loss_real += sess_output[2]
            accum_disc_loss += sess_output[3]
            accum_correct_count += sess_output[4]

            fake_label = sess_output[-1][0]
            fake_image = (sess_output[-2][0] + 1.0) * (255.0/2.0)

            step_counter += 1

        # global_step = self.sess.run(self.model.global_step_disc)
        cv2.imwrite("fake_images/step_%d_%s_valid.jpg" % (step, self.valid_loader.label_name[fake_label]), fake_image)

        disc_loss_real = accum_disc_loss_real / step_counter
        disc_loss_fake = accum_disc_loss_fake / step_counter
        # cls_loss_real = accum_cls_loss_real / step_counter
        # cls_loss_fake = accum_fake_cls_loss / step_counter
        gen_loss = accum_gen_loss / step_counter
        disc_loss = accum_disc_loss /step_counter
        accuracy = accum_correct_count / (valid_batch_size * step_counter)

        # log for tensorboard
        cur_step = self.sess.run(self.model.global_step_disc)
        custom_summaries = [
            tf.Summary.Value(tag='disc_loss_real', simple_value=disc_loss_real),
            tf.Summary.Value(tag='disc_loss_fake', simple_value=disc_loss_fake),
            # tf.Summary.Value(tag='cls_loss_real', simple_value=cls_loss_real),
            # tf.Summary.Value(tag='cls_loss_fake', simple_value=cls_loss_fake),
            tf.Summary.Value(tag='gen_loss', simple_value=gen_loss),
            tf.Summary.Value(tag='disc_loss', simple_value=disc_loss),
            tf.Summary.Value(tag='accuracy', simple_value=accuracy),
        ]
        self.valid_summary_writer.add_summary(tf.Summary(value=custom_summaries), cur_step)
        self.valid_summary_writer.flush()

        # return disc_loss_real, cls_loss_real, disc_loss_fake, cls_loss_fake, gen_loss, accuracy, accum_conf_matrix
        return disc_loss_real, disc_loss_fake, gen_loss


def main(argv):

    learner = GanLearner()

    if not FLAGS.valid_only:
        learner.train()
    else:
        loss, accuracy, accum_conf_matrix = learner.valid()
        print(">> Validation result")
        print("\tloss = %f" % loss)
        print("\taccuracy = %f" % accuracy)
        print("\t==== confusion matrix ====")
        print(accum_conf_matrix)

    return

if __name__ == '__main__':
    tf.app.run()
