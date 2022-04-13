from tensorboardX import SummaryWriter

class TensorboardWriter(SummaryWriter):
    def __init__(self, logdir):
        super(TensorboardWriter, self).__init__(logdir)

    def log_training(self, train_loss, step):
        self.add_scalar('train_loss', train_loss, step)

    def log_evaluation(self, test_loss, test_acc, step, test_balanced_loss=None, test_balanced_acc=None):

        self.add_scalar('validation_loss', test_loss, step)
        self.add_scalar('validation_acc', test_acc, step)

        if test_balanced_loss is not None and test_balanced_acc is not None:
            self.add_scalar('validation_balanced_loss', test_balanced_loss, step)
            self.add_scalar('validation_balanced_loss', test_balanced_acc, step)