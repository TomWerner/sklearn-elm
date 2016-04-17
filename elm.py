from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import logging
import numpy as np
import scipy.sparse
import numexpr as ne
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import StandardScaler


class NeuronGroup:
    def __init__(self, input_dimensions, neuron_type, neuron_count):
        if neuron_type not in ['linear', 'tanh', 'sigmoid']:
            message = "%s is an unsupported neuron type. Try 'linear', 'tanh', or 'sigmoid'" % str(neuron_type)
            logging.error(message)
            raise ValueError(message)
        self.neuron_type = neuron_type
        self.neuron_count = neuron_count

        if neuron_type == 'linear':
            if neuron_count == -1:
                logging.info("-1 linear neurons requested, will match the number of input dimensions.")
                self.weight_matrix = np.eye(input_dimensions, input_dimensions)
                self.bias_vector = np.zeros(input_dimensions,)
            else:
                if neuron_count > input_dimensions:
                    message = "Requested %d linear neurons, but only %d dimensions. Only adding %d neurons." \
                              % (neuron_count, input_dimensions, input_dimensions)
                    logging.warning(message)
                self.weight_matrix = np.eye(input_dimensions, neuron_count)
                self.bias_vector = np.zeros(neuron_count,)
        else:
            self.weight_matrix = np.random.randn(input_dimensions, neuron_count)
            self.weight_matrix *= 3 / input_dimensions ** 0.5
            self.bias_vector = np.random.randn(neuron_count,)

    def transform(self, X):
        # The input to the neuron function is x * weights + bias, data.dot(weight_matrix) does the matrix mult
        activation_function_input = safe_sparse_dot(X, self.weight_matrix) + self.bias_vector
        activation_function_output = np.zeros(activation_function_input.shape)

        if self.neuron_type == "linear":
            activation_function_output = activation_function_input # We already computed this with the dot + bias

        else:
            if scipy.sparse.issparse(activation_function_input):
                if self.neuron_type == 'sigmoid':
                    activation_function_output = 1 / (1 + np.exp(-activation_function_input))
                elif self.neuron_type == 'tanh':
                    activation_function_output = np.tanh(activation_function_input)
            else:
                if self.neuron_type == "sigmoid":
                    ne.evaluate("1/(1+exp(-activation_function_input))", out=activation_function_output)
                elif self.neuron_type == "tanh":
                    ne.evaluate('tanh(activation_function_input)', out=activation_function_output)
        return activation_function_output


class BaseELM(BaseEstimator):
    def __init__(self, neuron_dictionary, alpha_l2=1e-6, solver=None):
        self.alpha_l2 = alpha_l2
        self.neuron_groups = []
        self.neuron_dictionary = neuron_dictionary
        self.coefs_ = None
        self.fitted_ = False
        self.solver = solver

    def get_params(self, deep=True):
        return {'alpha_l2': self.alpha_l2, 'neuron_dictionary': self.neuron_dictionary, 'solver': self.solver}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        """
        Fit the model using X, y as training data.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape [n_samples, n_outputs]
            Target values (class labels in classification, real numbers in
            regression)
        Returns
        -------
        self : object
            Returns an instance of self.
        """
        for neuron_type, neuron_count in self.neuron_dictionary.items():
            self.neuron_groups.append(NeuronGroup(X.shape[1], neuron_type, neuron_count))
        self.scaler = StandardScaler(with_mean=scipy.sparse.issparse(X))
        X = self.scaler.fit_transform(X)

        H = np.hstack(neuron_group.transform(X) for neuron_group in self.neuron_groups)

        if self.solver is None:
            Ht_H = safe_sparse_dot(H.T, H, dense_output=True)
            Ht_H += np.identity(Ht_H.shape[0]) * self.alpha_l2
            Ht_Y = safe_sparse_dot(H.T, y, dense_output=True)
            self.coefs_ = scipy.linalg.solve(Ht_H, Ht_Y, sym_pos=True)
        else:
            self.solver = self.solver.fit(H, y)
        self.fitted_ = True
        return self

    def predict(self, X):
        """
        Predict values using the model
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]
        Returns
        -------
        C : numpy array of shape [n_samples, n_outputs]
            Predicted values.
        """

        if not self.fitted_:
            raise ValueError("ELM not fitted")
        X = self.scaler.transform(X)
        H = np.hstack(neuron_group.transform(X) for neuron_group in self.neuron_groups)

        if self.solver is None:
            return safe_sparse_dot(H, self.coefs_, dense_output=True)
        else:
            return self.solver.predict(H)


class ClassificationELM(BaseELM, ClassifierMixin):
    def __init__(self, neuron_dictionary, alpha_l2=1e-6, solver=None):
        super(ClassificationELM, self).__init__(neuron_dictionary, alpha_l2, solver)

    def predict(self, X):
        prediction = super(ClassificationELM, self).predict(X)
        prediction[prediction < .5] = 0
        prediction[prediction >= .5] = 1

        return prediction


class RegressionELM(BaseELM, RegressorMixin):
    def __init__(self, neuron_dictionary, alpha_l2=1e-6, solver=None):
        super(RegressionELM, self).__init__(neuron_dictionary, alpha_l2, solver)


# from sklearn import datasets
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.cross_validation import cross_val_score
# # inputs, outputs = datasets.make_classification(n_samples=25000, n_features=20, n_informative=8, n_classes=2)
# inputs, outputs = datasets.make_regression() #(n_samples=25000, n_features=20, n_informative=8, n_targets=1)
#
#
# # log_reg = LogisticRegression()
# # score = cross_val_score(log_reg, inputs, outputs, cv=10)
# # print("Logistic Regression: ", np.mean(score), np.std(score))
# lin_reg = LinearRegression()
# score = cross_val_score(lin_reg, inputs, outputs, cv=10)
# print("Linear Regression: ", np.mean(score), np.std(score))
# for i in range(1, 100):
#     # elm = ClassificationELM({'linear': -1, 'tanh': i}, alpha_l2=1e-5)
#     elm = RegressionELM({'linear': -1, 'tanh': i}, alpha_l2=1e-5)
#     score = cross_val_score(elm, inputs, outputs, cv=10)
#     print("ELM with %d neurons: " % i, np.mean(score), np.std(score))
#
# # inputs = np.array(list(range(0, 100))).reshape(100, 1)
# # outputs = np.array([np.sin(x) for x in inputs])
# #
# # for i in range(1, 100):
# #     elm = RegressionELM({'linear': -1, 'tanh': i})
# #     elm.fit(inputs, outputs)
# #     print(elm.score(inputs, outputs))