import numpy


def validate(weight, bia, x, y):
    r = (weight.transpose().dot(x) + bia) * y

    return r > 0


class Perceptron(object):
    def __init__(self, feature_num):
        self.feature_num = feature_num
        self.weight = numpy.random.rand(feature_num, 1)
        self.bia = numpy.random.random_sample()

    def train(self, train_set, test_set, ela, epochs=20):
        e = 0
        for x, y in test_set:
            if not validate(self.weight, self.bia, x, y):
                e += 1
        print("epochs: {}, {:.2f}".format("begin", 1 - (1.0 * e / len(test_set))))

        for i in range(epochs):

            e_t = 0
            for x, y in train_set:
                if not validate(self.weight, self.bia, x, y):
                    self.weight = self.weight - ela * self.delta_w(x, y)
                    self.bia = self.bia - ela * self.delta_b(x, y)
                    e_t += 1

            print("train epochs: {}, {:.2f}".format(i, 1 - (1.0 * e_t / len(train_set))))

            e = 0
            for x, y in test_set:
                if not validate(self.weight, self.bia, x, y):
                    e += 1
            print("epochs: {}, {:.2f}".format(i, 1 - (1.0 * e / len(test_set))))

    def delta_w(self, x, y):
        """
        代价函数：C = y(w * x + b)
        """
        return - x * y

    def delta_b(self, x, y):
        """
        代价函数：C = y(w * x + b)
        """
        return - y
