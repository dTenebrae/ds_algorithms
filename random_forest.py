# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

from decision_tree import DecisionTree, accuracy_metric, balanced_accuracy_metric

# from collections import Counter

import numpy as np


# import pandas as pd

class RandomForest:
    def __init__(self, n_trees=5, oobs=False):
        self.n_trees = n_trees
        self.min_leaf = 1
        self._oobs = oobs
        self.forest = []
        self._out_of_bag = []
        self.out_of_bag_list = []

    def _get_bootstrap(self, data, labels, n_trees: int) -> list:
        n_samples = data.shape[0]
        bootstrap = []

        for i in range(n_trees):
            b_data = np.zeros(data.shape)
            b_labels = np.zeros(labels.shape)
            oob_list = []
            for j in range(n_samples):
                sample_index = np.random.randint(0, n_samples)
                b_data[j] = data[sample_index]
                b_labels[j] = labels[sample_index]
                # запишем рандомные значения выборки для того, чтобы потом найти, что не вошло
                oob_list.append(sample_index)
            bootstrap.append((b_data, b_labels))
            # запишем невошедшие индексы
            self._out_of_bag.append(list(set([_k for _k in range(n_samples)]) - set(oob_list)))

        return bootstrap

    def fit(self, data, labels):
        data = np.array(data)
        labels = np.array(labels)
        bootstrap = self._get_bootstrap(data, labels, self.n_trees)
        _i = 0

        for b_data, b_labels in bootstrap:
            mod = DecisionTree(min_leaf=1, ensemble=True)
            mod.fit(b_data, b_labels)
            # посчитаем balanced accuracy на oob выборке
            oob_acc = self.balanced_accuracy_metric(labels[self._out_of_bag[_i]],
                                                    mod.predict(data[self._out_of_bag[_i]]))
            self.forest.append(mod)
            self.out_of_bag_list.append(oob_acc)
            _i += 1

    def out_of_bag_score(self):
        # вернем среднее значение balanced accuracy на out-of-bag выборках
        return np.mean(self.out_of_bag_list)

    def predict(self, data):
        data = np.array(data)
        # добавим предсказания всех деревьев в список
        predictions = []
        for tree in self.forest:
            predictions.append(tree.predict(data))

        # сформируем список с предсказаниями для каждого объекта
        predictions_per_object = list(zip(*predictions))

        # выберем в качестве итогового предсказания для каждого объекта то,
        # за которое проголосовало большинство деревьев
        voted_predictions = []
        if not self._oobs:
            for obj in predictions_per_object:
                voted_predictions.append(max(set(obj), key=obj.count))
        else:
            # В качестве выбора метки класса пока выбрал сумму OOBS по каждому классу. Если
            # у отрицательного класса сумма метрик каждого проголосовавшего больше чем у
            # положительного, то в результат пишем 0, иначе 1
            acc = np.array(self.out_of_bag_list)
            for obj in predictions_per_object:
                obj = np.array(obj)
                zero_index = (obj <= 0).nonzero()
                non_zero_index = (obj > 0).nonzero()
                if sum(acc[zero_index]) >= sum(acc[non_zero_index]):
                    voted_predictions.append(0)
                else:
                    voted_predictions.append(1)

        return voted_predictions

    @staticmethod
    def balanced_accuracy_metric(actual, predicted):
        true_predicted = (actual == predicted).astype(np.int8)
        sensitivity = np.sum(actual * true_predicted) / np.sum(actual == 1)
        specificity = np.sum((1 - actual) * true_predicted) / np.sum(actual == 0)
        return 0.5 * (sensitivity + specificity)


if __name__ == '__main__':
    rnd_state = 42
    np.random.seed(rnd_state)
    dataset = load_breast_cancer()
    X = dataset['data']
    y = dataset['target']

    train_data, test_data, train_labels, test_labels = train_test_split(X,
                                                                        y,
                                                                        test_size=0.3,
                                                                        random_state=rnd_state)

    rf_model = RandomForest(10, oobs=True)
    rf_model.fit(train_data, train_labels)
    train_answers = rf_model.predict(train_data)
    test_answers = rf_model.predict(test_data)
    print(rf_model.out_of_bag_score())

    print('Random forest')
    print('-' * 50)
    print(f'Train -> Accuracy: {accuracy_metric(train_labels, train_answers):.3f}, '
          f'Balanced accuracy: {balanced_accuracy_metric(train_labels, train_answers):.3f}, '
          f'f1-score: {f1_score(train_labels, train_answers):.3f}')
    print(f'Test --> Accuracy: {accuracy_metric(test_labels, test_answers):.3f}, '
          f'Balanced accuracy: {balanced_accuracy_metric(test_labels, test_answers):.3f}, '
          f'f1-score: {f1_score(test_labels, test_answers):.3f}')

    single_tree = DecisionTree(min_leaf=1)
    single_tree.fit(train_data, train_labels)
    train_1_answers = single_tree.predict(train_data)
    test_1_answers = single_tree.predict(test_data)

    print('=' * 100)

    print('Decision tree')
    print('-' * 50)
    print(f'Train -> Accuracy: {accuracy_metric(train_labels, train_1_answers):.3f}, '
          f'Balanced accuracy: {balanced_accuracy_metric(train_labels, train_1_answers):.3f}, '
          f'f1-score: {f1_score(train_labels, train_1_answers):.3f}, '
          f'ROC-AUC: {roc_auc_score(train_labels, train_1_answers):.3f}')
    print(f'Test --> Accuracy: {accuracy_metric(test_labels, test_1_answers):.3f}, '
          f'Balanced accuracy: {balanced_accuracy_metric(test_labels, test_1_answers):.3f}, '
          f'f1-score: {f1_score(test_labels, test_1_answers):.3f}, '
          f'ROC-AUC: {roc_auc_score(test_labels, test_1_answers):.3f}')

    # test_answers = my_tree.predict(test_data, proba=True)
    # print(test_answers)
    #
    # train_accuracy = accuracy_metric(train_labels, train_answers)
    # test_accuracy = accuracy_metric(test_labels, test_answers)
    # colors = ListedColormap(['red', 'blue'])
    # light_colors = ListedColormap(['lightcoral', 'lightblue'])
    # plt.figure(figsize=(16, 7))
    #
    # # график обучающей выборки
    # plt.subplot(1, 2, 1)
    # xx, yy = get_meshgrid(train_data)
    # mesh_predictions = np.array(my_tree.predict(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
    # plt.pcolormesh(xx, yy, mesh_predictions, cmap=light_colors, shading='auto')
    # plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels,
    #             cmap=colors)
    # plt.title(f'Train accuracy={train_accuracy:.2f}')
    #
    # # график тестовой выборки
    # plt.subplot(1, 2, 2)
    # plt.pcolormesh(xx, yy, mesh_predictions, cmap=light_colors, shading='auto')
    # plt.scatter(test_data[:, 0], test_data[:, 1], c=test_labels, cmap=colors)
    # plt.title(f'Test accuracy={test_accuracy:.2f}')
    # plt.show()
