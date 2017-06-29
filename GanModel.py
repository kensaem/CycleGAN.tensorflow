from layer import *


class CycleGANModel:
    def __init__(self, batch_size=64):

        self.noise_size = 100
        self.cond_size = 5
        self.latent_cls_size = 10
        self.batch_size = batch_size

        self.global_step_disc = tf.Variable(0, trainable=False, name='global_step_discriminator')
        self.global_step_gen = tf.Variable(0, trainable=False, name='global_step_generator')
        self.keep_prob_ph = tf.placeholder(dtype=tf.float32, name='keep_probability')
        self.lr_gen_ph = tf.placeholder(dtype=tf.float32, name='learning_rate_for_generator')
        self.lr_disc_ph = tf.placeholder(dtype=tf.float32, name='learning_rate_for_discriminator')
        self.is_training_gen_ph = tf.placeholder(dtype=tf.bool, name='is_training_for_generator')
        self.is_training_disc_ph = tf.placeholder(dtype=tf.bool, name='is_training_for_discriminator')

        # FIXME temporary placeholder for convolutional dropout.
        self.batch_size_ph = tf.placeholder(
            dtype=tf.int32,
            name='batch_size_ph'
        )

        self.input_image_a_ph = tf.placeholder(
            dtype=tf.uint8,
            shape=[None, 256, 256, 3],
            name='input_image_a_ph')

        self.input_image_b_ph = tf.placeholder(
            dtype=tf.uint8,
            shape=[None, 256, 256, 3],
            name='input_image_b_ph')

        self.fake_pool_image_a_ph = tf.placeholder(
            dtype=tf.float32,
            shape=[None, 256, 256, 3],
            name='fake_pool_image_a_ph')

        self.fake_pool_image_b_ph = tf.placeholder(
            dtype=tf.float32,
            shape=[None, 256, 256, 3],
            name='fake_pool_image_b_ph')

        # NOTE Prepare images
        self.real_image_a_t = tf.div(tf.to_float(self.input_image_a_ph), 127.5) - 1.0
        self.real_image_b_t = tf.div(tf.to_float(self.input_image_b_ph), 127.5) - 1.0

        # NOTE Build model for generator
        self.fake_image_b_t = self.build_generator(
            input_t=self.real_image_a_t,
            batch_size=self.batch_size,
            name="generator_a_to_b"
        )
        self.fake_image_a_t = self.build_generator(
            input_t=self.real_image_b_t,
            batch_size=self.batch_size,
            name="generator_b_to_a"
        )
        print(self.fake_image_a_t)
        print(self.fake_image_b_t)

        self.cyc_image_b_t = self.build_generator(
            input_t=self.fake_image_a_t,
            batch_size=self.batch_size,
            name="generator_a_to_b",
            reuse=True,
        )

        self.cyc_image_a_t = self.build_generator(
            input_t=self.fake_image_b_t,
            batch_size=self.batch_size,
            name="generator_b_to_a",
            reuse=True,
        )
        print(self.cyc_image_a_t)
        print(self.cyc_image_b_t)

        tf.summary.histogram("real_image_a", self.real_image_a_t)
        tf.summary.histogram("real_image_b", self.real_image_b_t)
        tf.summary.histogram("fake_image_a", self.fake_image_a_t)
        tf.summary.histogram("fake_image_b", self.fake_image_b_t)
        tf.summary.histogram("cyc_image_a", self.cyc_image_a_t)
        tf.summary.histogram("cyc_image_b", self.cyc_image_b_t)

        # NOTE Build model for discriminator

        discriminator = self.build_patch_discriminator
        self.real_disc_a_t = discriminator(input_t=self.real_image_a_t, name="discriminator_a")
        self.real_disc_b_t = discriminator(input_t=self.real_image_b_t, name="discriminator_b")

        self.fake_disc_a_t = discriminator(input_t=self.fake_image_a_t, name="discriminator_a", reuse=True)
        self.fake_disc_b_t = discriminator(input_t=self.fake_image_b_t, name="discriminator_b", reuse=True)

        self.fake_pool_disc_a_t = discriminator(input_t=self.fake_pool_image_a_ph, name="discriminator_a", reuse=True)
        self.fake_pool_disc_b_t = discriminator(input_t=self.fake_pool_image_b_ph, name="discriminator_b", reuse=True)

        # NOTE Build loss functions
        self.build_loss()

        # NOTE build optimizers
        optimizer = tf.train.AdamOptimizer

        t_vars = tf.trainable_variables()
        d_a_vars = [var for var in t_vars if "discriminator_a" in var.name]
        d_b_vars = [var for var in t_vars if "discriminator_b" in var.name]
        g_a_vars = [var for var in t_vars if "generator_a_to_b" in var.name]
        g_b_vars = [var for var in t_vars if "generator_b_to_a" in var.name]
        g_vars = [var for var in t_vars if "generator_" in var.name]
        print("\n==== Variables for discriminator_a ====")
        for var in d_a_vars:
            print(var.name)
        self.train_op_disc_a = optimizer(self.lr_disc_ph, beta1=0.5).minimize(
            loss=self.disc_loss_a,
            var_list=d_a_vars,
        )

        print("\n==== Variables for discriminator_b ====")
        for var in d_b_vars:
            print(var.name)
        self.train_op_disc_b = optimizer(self.lr_disc_ph, beta1=0.5).minimize(
            loss=self.disc_loss_b,
            var_list=d_b_vars,
        )

        print("\n==== Variables for generator A to B ====")
        for var in g_a_vars:
            print(var.name)
        self.train_op_gen_a = optimizer(self.lr_gen_ph, beta1=0.5).minimize(
            loss=self.gen_loss_a,
            var_list=g_a_vars,
        )

        print("\n==== Variables for generator B to A ====")
        for var in g_b_vars:
            print(var.name)
        self.train_op_gen_b = optimizer(self.lr_gen_ph, beta1=0.5).minimize(
            loss=self.gen_loss_b,
            var_list=g_b_vars,
        )

        self.gen_loss_a_summ = tf.summary.scalar("gen_loss_a", self.gen_loss_a)
        self.gen_loss_b_summ = tf.summary.scalar("gen_loss_b", self.gen_loss_b)
        self.disc_loss_a_summ = tf.summary.scalar("disc_loss_a", self.disc_loss_a)
        self.disc_loss_b_summ = tf.summary.scalar("disc_loss_b", self.disc_loss_b)

        return

    def build_generator(
            self,
            input_t,
            batch_size,
            reuse=False,
            name="generator",
    ):
        with tf.variable_scope(name, reuse=reuse):
            output_t = input_t

            ks = 3
            f = 7
            output_t = tf.pad(output_t, [[0, 0], [ks, ks], [ks, ks], [0, 0]], "REFLECT")

            # Convolution blocks
            output_t = conv_block(
                input_t=output_t,
                output_channel=64,
                kernel_size=f,
                padding='VALID',
                name="conv_1",
                norm=instance_norm,
                activation=tf.nn.relu,
            )

            output_t = conv_block(
                input_t=output_t,
                output_channel=128,
                kernel_size=ks,
                stride_size=2,
                padding='SAME',
                name="conv_2",
                norm=instance_norm,
                activation=tf.nn.relu,
            )

            output_t = conv_block(
                input_t=output_t,
                output_channel=256,
                kernel_size=ks,
                stride_size=2,
                padding='SAME',
                name="conv_3",
                norm=instance_norm,
                activation=tf.nn.relu,
            )

            # Residual blocks
            for res_idx in range(9):
                output_t = res_block(
                    input_t=output_t,
                    output_shape=256,
                    layer_name="res_block"+str(res_idx),
                )

            # Deconv blocks
            output_t = deconv_block(
                output_t,
                [batch_size, 128, 128, 128],
                layer_name="conv_tp_1",
                kernel_size=ks,
                stride_size=2,
                norm=instance_norm,
                activation=tf.nn.relu,
            )

            output_t = deconv_block(
                output_t,
                [batch_size, 256, 256, 64],
                layer_name="conv_tp_2",
                kernel_size=ks,
                stride_size=2,
                norm=instance_norm,
                activation=tf.nn.relu,
            )

            output_t = tf.pad(output_t, [[0, 0], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
            output_t = conv_block(
                input_t=output_t,
                output_channel=3,
                kernel_size=f,
                stride_size=1,
                padding='VALID',
                name="conv_last",
                activation=tf.nn.tanh,
            )

        print(output_t)

        return output_t

    def build_discriminator(
            self,
            input_t,
            reuse=False,
            do_aug=False,
            name="discriminator"
    ):
        with tf.variable_scope(name, reuse=reuse):

            if do_aug:
                output_t = tf.cond(
                    self.is_training_disc_ph,
                    lambda: tf.map_fn(lambda img: tf.image.random_flip_left_right(img), input_t),
                    lambda: input_t
                )
            else:
                output_t = input_t

            # feature size 256
            output_t = conv_block(
                input_t=output_t,
                output_channel=64,
                kernel_size=4,
                stride_size=2,
                name="layer1",
                activation=lrelu,
            )

            # feature size 128
            output_t = conv_block(
                input_t=output_t,
                output_channel=128,
                name="layer2",
                norm=instance_norm,
                activation=lrelu,
                is_training=self.is_training_disc_ph,
            )

            # feature size 64
            output_t = conv_block(
                input_t=output_t,
                output_channel=256,
                name="layer3",
                norm=instance_norm,
                is_training=self.is_training_disc_ph,
            )

            # feature size 32
            output_t = conv_block(
                input_t=output_t,
                output_channel=512,
                name="layer4",
                norm=instance_norm,
                is_training=self.is_training_disc_ph,
            )

            # feature size 16
            output_t = conv_block(
                input_t=output_t,
                output_channel=512,
                name="layer5",
                norm=instance_norm,
                is_training=self.is_training_disc_ph,
            )

            # feature size 8
            output_t = conv_block(
                input_t=output_t,
                output_channel=512,
                name="layer6",
                norm=instance_norm,
                is_training=self.is_training_disc_ph,
            )

            # feature size 4
            output_t = conv_block(
                input_t=output_t,
                output_channel=1,
                padding='VALID',
                name="layer7",
            )

        return output_t

    def build_patch_discriminator(
            self,
            input_t,
            reuse=False,
            do_aug=False,
            do_crop=False,
            name="discriminator"
    ):
        with tf.variable_scope(name, reuse=reuse):

            if do_aug:
                output_t = tf.cond(
                    self.is_training_disc_ph,
                    lambda: tf.map_fn(lambda img: tf.image.random_flip_left_right(img), input_t),
                    lambda: input_t
                )
            else:
                output_t = input_t

            if do_crop:
                output_t = tf.random_crop(output_t, [self.batch_size, 70, 70, 3])

            kw = 4

            # feature size 70
            output_t = conv_block(
                input_t=output_t,
                output_channel=64,
                kernel_size=kw,
                stride_size=2,
                name="layer1",
                activation=lrelu,
            )

            # feature size 35
            output_t = conv_block(
                input_t=output_t,
                output_channel=128,
                kernel_size=kw,
                stride_size=2,
                name="layer2",
                norm=instance_norm,
                activation=lrelu,
            )

            # feature size 17
            output_t = conv_block(
                input_t=output_t,
                output_channel=256,
                kernel_size=kw,
                stride_size=2,
                name="layer3",
                norm=instance_norm,
                activation=lrelu,
            )

            # feature size 8
            output_t = conv_block(
                input_t=output_t,
                output_channel=512,
                kernel_size=kw,
                stride_size=1,
                name="layer4",
                norm=instance_norm,
                activation=lrelu,
            )

            # feature size 4
            output_t = conv_block(
                input_t=output_t,
                output_channel=1,
                kernel_size=kw,
                stride_size=1,
                padding='VALID',
                name="layer5",
            )

        return output_t

    def build_loss(self):
        scale_factor = 10.0
        self.cyc_loss = tf.reduce_mean(tf.abs(self.real_image_a_t - self.cyc_image_a_t)) \
            + tf.reduce_mean(tf.abs(self.real_image_b_t - self.cyc_image_b_t))

        label = 0.9
        self.gen_loss_a = self.cyc_loss * scale_factor + tf.reduce_mean(tf.squared_difference(self.fake_disc_b_t, label))
        self.gen_loss_b = self.cyc_loss * scale_factor + tf.reduce_mean(tf.squared_difference(self.fake_disc_a_t, label))

        self.disc_loss_a = (tf.reduce_mean(tf.square(self.fake_pool_disc_a_t)) + tf.reduce_mean(tf.squared_difference(self.real_disc_a_t, label)))/2.0
        self.disc_loss_b = (tf.reduce_mean(tf.square(self.fake_pool_disc_b_t)) + tf.reduce_mean(tf.squared_difference(self.real_disc_b_t, label)))/2.0
        return
