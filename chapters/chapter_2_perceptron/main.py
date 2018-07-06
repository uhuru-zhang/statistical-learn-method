from chapters.chapter_2_perceptron.algorithmic.perceptron import Perceptron
from commen.loaddata import DataLoader

if __name__ == '__main__':
    perceptron = Perceptron(64)
    train_set, test_set, validation_set = DataLoader.load_digit(training_rate=0.8, test_rate=0.2,
                                                                fn=lambda y: (y % 2) * 2 - 1)
    perceptron.train(train_set=train_set, test_set=test_set, ela=0.01)
