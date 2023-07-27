def avg_per_epoch(loss,epoch):
    loss = loss[0:epoch] + [(sum(loss[epoch:])) / len(loss[epoch:])]

    return loss