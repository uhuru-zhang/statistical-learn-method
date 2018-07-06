import numpy
import random

from sklearn import datasets


class DataLoader(object):
    @staticmethod
    def load_digit(training_rate=0.6, test_rate=0.2, fn=lambda a: a):
        """
        获取数字图片数据集
        :param fn:
        :param training_rate: 训练数据所占比例
        :param test_rate: 测试数据所占比例
        :return: 返回三组（x, y）training_rate 训练集，test_rate 测试集，1 - training_rate - test_rate 验证集
            x 为 64 * 1 的矩阵（8 * 8 像素图片），y初始数值为图片中数字
        """
        digits = datasets.load_digits()

        couple = []
        for x, y in zip(digits.data, digits.target):
            couple.append((numpy.ndarray(shape=(64, 1), buffer=x), fn(y)))

        random.shuffle(couple)
        return couple[:int(len(couple) * training_rate)], \
               couple[int(len(couple) * training_rate):int(len(couple) * (training_rate + test_rate))], \
               couple[int(len(couple) * (training_rate + test_rate)):]
