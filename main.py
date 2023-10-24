import siamese_lstm_cnn_model
import argparse
import data_utils
import sys
import os
import pickle
import tensorflow as tf
import scipy.stats as meas
import numpy as np
from tqdm import tqdm


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser()
parser.add_argument("--is_load_model", type=str2bool, default=False,
                    help="Bool to load a pretrained model.")
parser.add_argument("--is_train", type=str2bool, default=True,
                    help="Bool to train the system.")
parser.add_argument("--is_test", type=str2bool, default=False,
                    help="Bool to test the system.")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size")
parser.add_argument("--nb_epochs", type=int, default=201,
                    help="Number of epochs")
parser.add_argument("--max_length", type=int, default=50,
                    help="Max sentence length")
parser.add_argument("--hidden_size", type=int, default=50,
                    help="Hidden layer size.")
parser.add_argument("--forget_bias", type=str, default=2.5,
                    help="Forget bias.")
parser.add_argument("--learning_rate", type=float, default=0.1,
                    help="Learning rate.")
parser.add_argument("--number_layers", type=int, default=1,
                    help="Number of layers.")
parser.add_argument("--dropout", type=float, default=0.5,
                    help="Dropout.")
parser.add_argument("--word_emb_size", type=int, default=300,
                    help="word embedding size.")
parser.add_argument("--local_context_size", type=int, default=5,
                    help="Local context size.")
args = parser.parse_args()


def load_dataset():
    data = data_utils.load_data_from_json()
    train = data[:5900]
    val = data[5900:7900]
    test = data[7900:len(data)]

    return np.asanyarray(train), np.asanyarray(val), np.asanyarray(test)


def calculate_correlation(prediction, reference):
    """
        Calculate the error and correlations of predictions.
    """
    predictions, references = [], []
    prediction = list((np.asarray(prediction) * 4.0) + 1.0)
    predictions.extend(prediction)
    references.extend(reference)

    predictions = np.array(predictions)
    references = np.array(references)
    print("Error\tCorrelation_Pearson\t\tCorrelation_Spearman:")
    print(str(np.mean(np.square(predictions-references))) + "\t" + str(meas.pearsonr(predictions,
          references)[0]) + "\t\t" + str(meas.spearmanr(references, predictions)[0]))

    return meas.pearsonr(predictions, references)[0]


def test_network(sess, network, test):
    # Initialize iterator with test data
    feed_dict = {network.x1:          test[0],
                 network.len1:        test[1],
                 network.x2:          test[2],
                 network.len2:        test[3],
                 network.y:           test[4],
                 network.batch_size:  np.array(test[0]).shape[0]}
    sess.run(network.iter.initializer, feed_dict=feed_dict)
    for _ in tqdm(range(1)):
        loss, prediction, reference = sess.run(
            [network.loss_test, network.prediction_test, network.reference])

    return loss, prediction, reference


def training_network(sess, network, train, val, path_save, nb_epochs):
    old_correlation, count = 0.0, 0
    n_batches = np.array(train[0]).shape[0] // args.batch_size

    # Initialise iterator with train data
    feed_dict = {network.x1:         train[0],
                 network.len1:       train[1],
                 network.x2:         train[2],
                 network.len2:       train[3],
                 network.y:          train[4],
                 network.batch_size: args.batch_size}
    sess.run(network.iter.initializer, feed_dict=feed_dict)

    for eidx in range(nb_epochs):
        tot_loss = 0
        for _ in tqdm(range(n_batches)):
            _, loss_value = sess.run([network.train_op, network.loss])
            tot_loss += loss_value
        print("Iter: {}, Loss: {:.4f}".format(eidx, tot_loss / n_batches))

        # Early stopping
        if not eidx % 10:
            print("Early stopping")
            loss, prediction, reference = test_network(sess, network, val)
            correlation = calculate_correlation(prediction, reference)

            if correlation > old_correlation:
                old_correlation = correlation
                count = 0
                print("Saving model...")
                network.saver.save(sess, path_save)
                print("Model saved in file: %s" % path_save)
            else:
                count += 1
                if count >= 10:
                    break

            # Reload training dataset
            feed_dict = {network.x1:         train[0],
                         network.len1:       train[1],
                         network.x2:         train[2],
                         network.len2:       train[3],
                         network.y:          train[4],
                         network.batch_size: args.batch_size}
            sess.run(network.iter.initializer, feed_dict=feed_dict)


def main():
    # Path to model
    path_save = 'models/se_' + str(args.hidden_size) + 'fb_' + \
        str(args.forget_bias) + 'nl_' + str(args.number_layers)
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    path_save += '/model'

    # Load dataset train: 5900, val: 2000, test: 2027
    train, val, test = load_dataset()

    # Get emb vector [embs1, lenghts1, embs2, lenghts2, score]
    train = data_utils.prepare_data(train, maxlen=args.max_length)
    val = data_utils.prepare_data(val, maxlen=args.max_length)
    test = data_utils.prepare_data(test, maxlen=args.max_length)

    network = siamese_lstm_cnn_model.SiameseLSTMCNN(
        sequence_embedding=args.hidden_size,
        forget_bias=args.forget_bias,
        learning_rate=args.learning_rate,
        number_layers=args.number_layers,
        max_length=args.max_length,
        word_emb_size=args.word_emb_size,
        local_context_size=args.local_context_size,
        dropout=args.dropout
    )

    with tf.compat.v1.Session() as sess:
        sess.run(network.initialize_variables)
        if args.is_load_model and os.path.isfile(path_save+".index"):
            network.saver.restore(sess, path_save)
            print("Model restored!")
        if args.is_train:
            print('Training network........')
            training_network(sess, network, train, val, path_save, args.nb_epochs)
        if args.is_test:
            print('Testing network.......')
            loss, pre, ref = test_network(sess, network, test)
            calculate_correlation(prediction=pre, reference=ref)

if __name__ == '__main__':
    main()
