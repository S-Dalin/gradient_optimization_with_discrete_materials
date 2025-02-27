import os
import numpy as np
import random
import tensorflow as tf
import keras
from keras import Model
from keras.layers import Dense, Dropout, Input, BatchNormalization, LeakyReLU, ReLU
from keras.optimizers import Adam
from tqdm import trange
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.layers import Layer

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
os.environ["KERAS_BACKEND"] = "tensorflow"


class WGAN_GP:
    def __init__(
        self,
        latent_dim,
        N_numerical,
        N_mat_core,
        N_mat_shell,
        g_lr,
        d_lr,
        beta_1,
        beta_2,
        n_critic,
        n_generator,
        gradient_penalty_weight,
    ):
        self.latent_dim = latent_dim
        self.N_numerical = N_numerical
        self.N_mat_core = N_mat_core
        self.N_mat_shell = N_mat_shell
        self.data_dim = N_numerical + N_mat_core + N_mat_shell
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.n_critic = n_critic
        self.n_generator = n_generator
        self.gradient_penalty_weight = gradient_penalty_weight

        self.generator = self.build_generator()
        self.critic = self.build_critic()

        self.g_optimizer = Adam(
            learning_rate=self.g_lr,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            clipvalue=1.0,
        )
        self.c_optimizer = Adam(
            learning_rate=self.d_lr,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            clipvalue=1.0,
        )

        self.generator_loss = []
        self.critic_loss = []

    def build_generator(self):
        in_layer = tf.keras.Input(shape=(self.latent_dim,))

        x = tf.keras.layers.Dense(512, kernel_initializer="he_normal")(in_layer)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(256, kernel_initializer="he_normal")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(128, kernel_initializer="he_normal")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        out_1 = tf.keras.layers.Dense(
            self.N_numerical, activation="tanh", kernel_initializer="he_normal"
        )(x)

        out_2 = tf.keras.layers.Dense(self.N_mat_core, kernel_initializer="he_normal")(
            x
        )
        out_2 = tf.keras.layers.Softmax()(out_2)
        out_3 = tf.keras.layers.Dense(self.N_mat_shell, kernel_initializer="he_normal")(
            x
        )
        out_3 = tf.keras.layers.Softmax()(out_3)

        model = tf.keras.Model(inputs=in_layer, outputs=[out_1, out_2, out_3])
        return model

    def build_critic(self):
        in_1 = tf.keras.Input(shape=(self.N_numerical,))
        in_2 = tf.keras.Input(shape=(self.N_mat_core,))
        in_3 = tf.keras.Input(shape=(self.N_mat_shell,))

        x = tf.keras.layers.Concatenate()([in_1, in_2, in_3])

        x = tf.keras.layers.Dense(512, kernel_initializer="he_normal")(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Dense(256, kernel_initializer="he_normal")(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Dense(128, kernel_initializer="he_normal")(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)

        out = tf.keras.layers.Dense(1)(x)

        model = tf.keras.Model(inputs=[in_1, in_2, in_3], outputs=out)
        return model

    def gradient_penalty(self, real, fake):
        epsilon = tf.random.uniform(
            [real[0].shape[0], 1], minval=0.0, maxval=1.0, dtype=tf.dtypes.float32
        )
        x_hat = [
            tf.cast(epsilon, tf.float32) * tf.cast(real_i, tf.float32)
            + (1 - tf.cast(epsilon, tf.float32)) * tf.cast(fake_i, tf.float32)
            for real_i, fake_i in zip(real, fake)
        ]
        with tf.GradientTape() as t:
            t.watch(x_hat)
            d_hat = self.critic(x_hat)
        gradients = t.gradient(d_hat, x_hat)
        ddx = [
            tf.sqrt(tf.reduce_sum(gradients_i**2, axis=[1]) + 1e-10)
            for gradients_i in gradients
        ]
        d_regularizer = tf.reduce_mean([(ddx_i - 1.0) ** 2 for ddx_i in ddx])
        return d_regularizer

    @tf.function
    def train_step(self, real_data):
        real_data = [
            tf.cast(real_data[..., : self.N_numerical], tf.float32),
            tf.cast(
                real_data[..., self.N_numerical : self.N_numerical + self.N_mat_core],
                tf.float32,
            ),
            tf.cast(real_data[..., self.N_numerical + self.N_mat_core :], tf.float32),
        ]

        ##

        for _ in range(self.n_critic):
            with tf.GradientTape() as d_tape:
                noise = tf.random.normal(
                    [real_data[0].shape[0], self.latent_dim], dtype=tf.dtypes.float32
                )
                fake_numerical, fake_mat_core, fake_mat_shell = self.generator(noise)
                fake_data = [fake_numerical, fake_mat_core, fake_mat_shell]
                d_loss = self.critic_loss_fn(real_data, fake_data)
            d_gradients = d_tape.gradient(d_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(
                zip(d_gradients, self.critic.trainable_variables)
            )

        for _ in range(self.n_generator):
            with tf.GradientTape() as g_tape:
                noise = tf.random.normal(
                    [real_data[0].shape[0], self.latent_dim], dtype=tf.dtypes.float32
                )
                g_loss = self.generator_loss_fn(noise)
            g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
            self.g_optimizer.apply_gradients(
                zip(g_gradients, self.generator.trainable_variables)
            )

        return d_loss, g_loss

    def critic_loss_fn(self, real, fake):
        logits_real = self.critic(real)
        logits_fake = self.critic(fake)
        gp = self.gradient_penalty(real, fake)
        c_loss = (
            tf.reduce_mean(logits_fake)
            - tf.reduce_mean(logits_real)
            + self.gradient_penalty_weight * gp
        )
        return c_loss

    def generator_loss_fn(self, noise):
        fake = self.generator(noise)
        logits_fake = self.critic(fake)
        g_loss = -tf.reduce_mean(logits_fake)
        return g_loss

    def train(self, data, epochs, sample_interval):
        for epoch in trange(epochs):
            d_loss, g_loss = self.train_step(data)
            self.critic_loss.append(d_loss.numpy())
            self.generator_loss.append(g_loss.numpy())

            if epoch % sample_interval == 0:
                print(
                    f"Epoch: {epoch}, Discriminator Loss: {d_loss.numpy()}, Generator Loss: {g_loss.numpy()}"
                )

        self.plot_losses()

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.generator_loss, color="#E34E5A", label="Generator Loss")
        plt.plot(self.critic_loss, color="#06346B", label="Discriminator Loss")
        plt.axvline(0, color="black", linewidth=0.7)
        plt.axhline(0, color="black", linewidth=1)
        plt.axhline(2, color="black", linewidth=0.5)
        plt.axhline(-2, color="black", linewidth=0.5)
        plt.xlabel("Iterations", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.title("Generator and Discriminator Losses", fontsize=14)
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.show()

        # Save the losses
        loss_data = {
            "generator_loss": self.generator_loss,
            "critic_loss": self.critic_loss,
        }
        with open("models/wgangp_loss_data.pkl", "wb") as f:
            pickle.dump(loss_data, f)


# Function to load the generator
def load_generator(model_path):
    generator = keras.models.load_model(model_path)
    return generator


# Generate synthetic data
def generate_synthetic_data(generator, latent_dim, num_samples):
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    synthetic_data = generator(noise)  # keras3
    return synthetic_data


# Inverse transform synthetic data to the original scale
def inverse_transform_synthetic_data(
    preprocessor, synthetic_data, save_path="models/synthetic_data.npy"
):
    """
    Inverse transform synthetic data to the original scale and save to a file.

    Parameters:
    preprocessor (ColumnTransformer): The preprocessor used to scale the data.
    synthetic_data (list): List of arrays containing the generated synthetic data.
    save_path (str): The path where the transformed data will be saved.

    Returns:
    numpy.ndarray: The synthetic data transformed back to the original scale.
    """

    synthetic_data = np.concatenate(synthetic_data, axis=1)
    num_features_len = len(preprocessor.transformers_[0][2])

    synthetic_data_num = synthetic_data[:, :num_features_len]
    synthetic_data_cat = synthetic_data[:, num_features_len:]

    # Inverse transform the numerical data
    num_scaler = preprocessor.named_transformers_["num"]
    synthetic_data_num_original = num_scaler.inverse_transform(synthetic_data_num)

    # Inverse transform the categorical data
    cat_encoder = preprocessor.named_transformers_["cat"]
    synthetic_data_cat_original = cat_encoder.inverse_transform(synthetic_data_cat)

    # Combine the results back into a single dataset
    synthetic_data_original_scale = np.hstack(
        (synthetic_data_num_original, synthetic_data_cat_original)
    )

    # Save synthetic data
    np.save(save_path, synthetic_data_original_scale)

    return synthetic_data_original_scale
