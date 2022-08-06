import numpy as np

class Node:
    def __init__(self, right, left, level : int, value):
        self.right = right
        self.left = left
        self.level = level
        self.value = value


class RegressionTree:
    def __init__(self, max_level=3, min_objects=10):
        self.max_level = max_level
        self.min_objects = min_objects
        self.root = Node(None, None, 0, None)
    
    def train(self, X, Y):
        self.root = self.__train__(X, Y, 0)

    def predict_list(self, X):
        res = []
        for x in X:
            res.append(self.predict(x))
        res = np.array(res)
        return res
    
    def predict(self, x):
        return self.__predict__(x, self.root)

    def __predict__(self, x, node):
        if node == None:
            return None
        elif node.left == None and node.right == None:
            return node.value
        else:
            if x[node.value[1]] < node.value[0]:
                return self.__predict__(x, node.left)
            else:
                return self.__predict__(x, node.right)

    def __train__(self, X, Y, level=0) -> Node:
        if level >= self.max_level:
            return None if len(Y) == 0 else Node(None, None, level, Y.mean())
        elif len(Y) < self.min_objects:
            return None if len(Y) == 0 else Node(None, None, level, Y.mean())
        else:
            data = self.__best_column__(X, Y)
            node = Node(None, None, level + 1, (data["mean"], data["index"]))
            left_x = []
            left_y = []
            right_x = []
            right_y = []
            # Разделение данных для веток
            i = 0
            while i < len(Y):
                if X[i][data["index"]] < data["mean"]:
                    left_x.append(X[i])
                    left_y.append(Y[i])
                else:
                    right_x.append(X[i])
                    right_y.append(Y[i])
                i += 1
            
            left_x = np.array(left_x)
            left_y = np.array(left_y)
            right_x = np.array(right_x)
            right_y = np.array(right_y)
            
            node.left = self.__train__(left_x, left_y, level + 1)
            node.right = self.__train__(right_x, right_y, level + 1)
            return node

    # Возвращает номер столбца для лучшего разбиения и значение mse
    def __best_column__(self, X, Y):
        row_n = len(X)
        column_n = len(X[0])
        column_id = 0

        MSE_L = []
        MEAN = 0
        while column_id < column_n:
            row_id = 0
            mse = X.max()*100000000000
            while row_id < row_n - 1:
                mean = 0.5 * (X[row_id][column_id] + X[row_id + 1][column_id])
                mean_left = 0
                mean_right = 0
                count_left = 0
                count_right = 0

                # Расчёт средних значений для левой и правой части
                i = 0
                while i < row_n:
                    if X[i][column_id] < mean:
                        mean_left += Y[i]
                        count_left += 1
                    else:
                        mean_right += Y[i]
                        count_right += 1
                    i += 1

                if count_left != 0:
                    mean_left /= count_left
                if count_right != 0:
                    mean_right /= count_right
                #print(f"mean_left = {mean_left}; mean_right = {mean_right}")

                # Расчёт локального mse
                mse_local = 0
                i = 0
                while i < row_n:
                    if X[i][column_id] < mean:
                        mse_local += pow(Y[i] - mean_left, 2)
                    else:
                        mse_local += pow(Y[i] - mean_right, 2)
                    i += 1
                #print(f"mse_local = {mse_local}")
                if mse_local < mse:
                    mse = mse_local
                    MEAN = mean

                row_id += 1
            MSE_L.append((mse, MEAN, column_id))
            column_id += 1
        
        # Поиск минимального mse
        needed_tuple = MSE_L[0]
        i = 1
        while i < column_n:
            if MSE_L[i][0] < needed_tuple[0]:
                needed_tuple = MSE_L[i]
            i += 1
        #print(MSE_L)
        return {"MSE" : needed_tuple[0], "index" : needed_tuple[2], "mean" : needed_tuple[1]}


class GradientBoosting:
    def __init__(self, params):
        self.amount_of_trees = params["amount_of_trees"]
        self.trees = []
        self.max_level_tree = params["max_level_tree"]
    
    def c(self, k):
        return 1 / pow(k, 0.5)

    def error(self, fact, predicted):
        return pow(fact - predicted, 2)
    
    def derivate_error(self, fact, predicted):
        return -2 * (predicted - fact)

    def score(self, X, Y):
        res = 0
        i = 0
        n = len(Y)
        
        i = 0
        while i < n:
            res += pow(self.predict(X[i]) - Y[i], 2)
            i += 1
        return res

    def train(self, dataset):
        X = dataset[:, 0]
        Y = dataset[:, 1]
        X = np.array([np.array([x]) for x in X])

        i = 0
        while i < self.amount_of_trees:
            self.trees.append(RegressionTree(self.max_level_tree, 7))
            self.trees[-1].train(X, Y)
            predicted = self.trees[-1].predict_list(X)
            Y = self.derivate_error(Y, predicted)
            i += 1
    
    def predict(self, x):
        res = 0
        i = 1
        for tree in self.trees:
            #res += self.c(i) * tree.predict(x)
            res += tree.predict(x)
            i += 1
        return res

