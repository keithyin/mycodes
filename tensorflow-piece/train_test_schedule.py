
"""
This code is used for train test schedule, some of them is psedo code
When using tensorflow input pipeline, the following code can be applied to the application
"""
GLOBAL_STEP = 0

def train_n_iteration(sess, train_op, summary, writer, loss, accuracy, num_iteration):
    """
    for train n iteration
    :param sess: session
    :param train_op: train_op used for update the weights
    :param summary: summary op
    :param writer: train writer
    :param loss: loss op
    :param accuracy: accuracy op
    :param num_iteration: n iteration
    :return: Nothing
    """
    losses = 0.  # using to track the sum of the n iteration training state
    accuracies = 0.  # using to track the sum of the n iteration training state

    # we can using progressbar to track the progress of the code

    for i in range(num_iteration):
        _, loss_, accu_ = sess.run([train_op, loss, accuracy])
        losses += loss_
        accuracies += accu_
    losses /= float(num_iteration)
    accuracies /= float(num_iteration)
    summ = sess.run([summary], feed_dict={"loss_tracker:0":losses,
                                          "accuracy_tracker:0": accuracies,
                                          "train_flag:0": True})
    writer.add_summary(summ, GLOBAL_STEP)


def test_n_iteration(sess, summary, writer, loss, accuracy, num_iteration):
    """
    for train n iteration, we have fixed the batch size, so the num_iteration should equal to num_test_file/batch_size
    :param sess: session
    :param summary: summary op
    :param writer: test writer
    :param loss: loss op
    :param accuracy: accuracy op
    :param num_iteration: n iteration
    :return: Nothing
    """

    losses = 0.  # using to track the sum of the n iteration training state
    accuracies = 0.  # using to track the sum of the n iteration training state

    # we can using progressbar to track the progress of the code

    for i in range(num_iteration):
        loss_, accu_ = sess.run([ loss, accuracy])
        losses += loss_
        accuracies += accu_
    losses /= float(num_iteration)
    accuracies /= float(num_iteration)
    summ = sess.run([summary], feed_dict={"loss_tracker:0":losses,
                                          "accuracy_tracker:0": accuracies,
                                          "train_flag:0":False})
    writer.add_summary(summ, GLOBAL_STEP)







