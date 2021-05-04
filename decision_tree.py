from collections import Counter
import numpy as np


class DecisionTree:
    def __init__(self, crit='gini', min_leaf=5, n_depth=None, ensemble=True):
        self.crit = crit
        self.min_leaf = min_leaf
        self.n_depth = n_depth
        self._tree = None
        self.ensemble = ensemble

    class Node:

        def __init__(self, index, t, true_branch, false_branch):
            # индекс признака, по которому ведется сравнение с порогом в этом узле
            self.index = index
            self.t = t  # значение порога
            # поддерево, удовлетворяющее условию в узле
            self.true_branch = true_branch
            # поддерево, не удовлетворяющее условию в узле
            self.false_branch = false_branch

    class Leaf:

        def __init__(self, data, labels):
            self.data = data
            self.labels = labels
            self.prediction = self.predict()
            self.predict_proba = self.proba()

        def predict(self):
            # подсчет количества объектов разных классов
            classes = Counter(self.labels)

            #  найдем класс, количество объектов которого будет максимальным
            #  в этом листе и вернем его
            return max(classes, key=classes.get)

        def proba(self):
            classes = Counter(self.labels)
            result = classes[1] / len(self.labels)
            return result

    @staticmethod
    def _get_subsample(len_sample: int) -> np.ndarray:
        return np.random.choice(np.arange(len_sample), size=int(np.sqrt(len_sample)))

    def _build_tree(self, data, labels, depth):
        quality, t, index = self._find_best_split(data, labels)

        #  Базовый случай - прекращаем рекурсию, когда нет прироста в качества
        if quality == 0 or (self.n_depth is not None) and (depth >= self.n_depth):
            return self.Leaf(data, labels)

        true_data, false_data, true_labels, false_labels = self._split(data, labels, index, t)

        # Рекурсивно строим два поддерева
        depth += 1
        true_branch = self._build_tree(true_data, true_labels, depth)
        false_branch = self._build_tree(false_data, false_labels, depth)

        # Возвращаем класс узла со всеми поддеревьями, то есть целого дерева
        return self.Node(index, t, true_branch, false_branch)

    def _info_crit(self, labels):
        #  подсчет количества объектов разных классов
        classes = Counter(labels)

        #  расчет критерия Джини
        if self.crit == 'gini':
            impurity = 1
            impurity -= np.sum([(classes[_i] / len(labels)) ** 2 for _i in classes])
        #  расчет энтропии Шэннона
        elif self.crit == 'shannon':
            impurity = 0
            for obj in classes:
                p = classes[obj] / len(labels)
                impurity += -p * np.log2(p) if p > 0 else 0
        else:
            raise ValueError('Wrong information criterion')

        return impurity

    def _quality(self, left_lbl, right_lbl, current_info):
        # доля выборки, ушедшая в левое поддерево
        p = float(left_lbl.shape[0]) / (left_lbl.shape[0] + right_lbl.shape[0])
        return current_info - p * self._info_crit(left_lbl) - (1 - p) * self._info_crit(right_lbl)

    @staticmethod
    def _split(data, labels, index, t):
        left = np.where(data[:, index] <= t)
        right = np.where(data[:, index] > t)
        return data[left], data[right], labels[left], labels[right]

    def _find_best_split(self, data, labels):
        #  обозначим минимальное количество объектов в узле
        current_info = self._info_crit(labels)

        best_quality = 0
        best_t = None
        best_index = None

        features = self._get_subsample(data.shape[1]) if self.ensemble else [_ for _ in
                                                                             range(data.shape[1])]

        for index in features:
            # будем проверять только уникальные значения признака,
            # исключая повторения
            t_values = np.unique([row[index] for row in data])

            for t in t_values:
                true_data, false_data, true_lbl, false_lbl = self._split(data, labels, index, t)
                #  пропускаем разбиения, в которых в узле остается менее min_leaf объектов
                if len(true_data) < self.min_leaf or len(false_data) < self.min_leaf:
                    continue

                current_quality = self._quality(true_lbl, false_lbl, current_info)

                #  выбираем порог, на котором получается максимальный
                #  прирост качества
                if current_quality > best_quality:
                    best_quality, best_t, best_index = current_quality, t, index

        return best_quality, best_t, best_index

    def _classify_object(self, obj, node, proba=False):
        #  Останавливаем рекурсию, если достигли листа
        if isinstance(node, self.Leaf):
            if proba:
                answer = node.predict_proba
            else:
                answer = node.prediction
            return answer

        if obj[node.index] <= node.t:
            return self._classify_object(obj, node.true_branch, proba)
        else:
            return self._classify_object(obj, node.false_branch, proba)

    def fit(self, data, labels):
        data = np.array(data)
        labels = np.array(labels)
        self._tree = self._build_tree(data, labels, depth=0)
        return self

    def predict(self, data, proba=False):
        data = np.array(data)
        return np.array([self._classify_object(obj, self._tree, proba=proba) for obj in data])

    def _max_depth(self, node):
        if isinstance(node, self.Leaf):
            return 0
        ldepth = self._max_depth(node.true_branch)
        rdepth = self._max_depth(node.false_branch)
        return ldepth + 1 if ldepth > rdepth else rdepth + 1

    def _p_tree(self, node, level=0):
        if not isinstance(node, self.Leaf):
            self._p_tree(node.true_branch, level + 1)
            print(' ' * 5 * level + '-->', f'{node.t:.3f}')
            self._p_tree(node.false_branch, level + 1)

    def get_depth(self):
        return self._max_depth(self._tree)

    def print_tree(self):
        self._p_tree(self._tree)


def accuracy_metric(actual, predicted):
    return np.sum(actual == predicted) / len(actual)


def balanced_accuracy_metric(actual, predicted):
    true_predicted = (actual == predicted).astype(np.int8)
    sensitivity = np.sum(actual * true_predicted) / np.sum(actual == 1)
    specificity = np.sum((1 - actual) * true_predicted) / np.sum(actual == 0)
    return 0.5 * (sensitivity + specificity)
