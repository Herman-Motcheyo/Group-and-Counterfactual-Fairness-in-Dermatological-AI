import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def augment_image(image):
    # Flip horizontal et vertical
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # Zoom central aléatoire (recadrage puis resize)
    scale = tf.random.uniform([], 0.9, 1.0)
    crop_height = tf.cast(scale * tf.cast(tf.shape(image)[0], tf.float32), tf.int32)
    crop_width = tf.cast(scale * tf.cast(tf.shape(image)[1], tf.float32), tf.int32)
    image = tf.image.random_crop(image, size=[crop_height, crop_width, 3])
    image = tf.image.resize(image, [224, 224])

    # Brightness, contrast, saturation, hue
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    image = tf.image.random_hue(image, 0.05)

    # Bruit gaussien
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=2.0, dtype=tf.float32)
    image = tf.cast(image, tf.float32) + noise
    image = tf.clip_by_value(image, 0.0, 255.0)

    return tf.cast(image, tf.float32)

def make_multimodal_dataset(images, metadata, labels, batch_size=32, shuffle=True, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices(((images, metadata), labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images))

    if augment:
        def augment_fn(inputs, label):
            img, meta = inputs
            img = augment_image(img)
            return (img, meta), label

        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)




def save_history_to_csv(history, fold_number, models_name, save_dir='histories'):
    """
    Sauvegarde l'historique d'entraînement dans un fichier CSV.

    :param history: objet `History` retourné par `model.fit`
    :param fold_number: entier indiquant le numéro du pli
    :param save_dir: dossier de sauvegarde
    """
    os.makedirs(f'{save_dir}/{models_name}/fold_{fold_number}/', exist_ok=True)

    hist_df = pd.DataFrame(history.history)    
    hist_df['epoch'] = history.epoch

    file_path = f'{save_dir}/{models_name}/fold_{fold_number}/history_fold_{fold_number}.csv'

    hist_df.to_csv(file_path, index=False)
    print(f"Historique sauvegardé : {file_path}")


def plot_training(hist):
    tr_acc = hist.history['accuracy']
    tr_loss = hist.history['loss']
    val_acc = hist.history['val_accuracy']
    val_loss = hist.history['val_loss']
    index_loss = np.argmin(val_loss)
    val_lowest = val_loss[index_loss]
    index_acc = np.argmax(val_acc)
    acc_highest = val_acc[index_acc]

    plt.figure(figsize= (20, 8))
    plt.style.use('fivethirtyeight')
    Epochs = [i+1 for i in range(len(tr_acc))]
    loss_label = f'best epoch= {str(index_loss + 1)}'
    acc_label = f'best epoch= {str(index_acc + 1)}'
    
    plt.subplot(1, 2, 1)
    plt.plot(Epochs, tr_loss, 'r', label= 'Training loss')
    plt.plot(Epochs, val_loss, 'g', label= 'Validation loss')
    plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'blue', label= loss_label)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(Epochs, tr_acc, 'r', label= 'Training Accuracy')
    plt.plot(Epochs, val_acc, 'g', label= 'Validation Accuracy')
    plt.scatter(index_acc + 1 , acc_highest, s= 150, c= 'blue', label= acc_label)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout
    plt.show()