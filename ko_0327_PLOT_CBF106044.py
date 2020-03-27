"""
==========================================
One-class SVM with non-linear kernel (RBF)
==========================================

An example using a one-class SVM for novelty detection.

:ref:`One-class SVM <svm_outlier_detection>` is an unsupervised
algorithm that learns a decision function for novelty detection:
classifying new data as similar or different to the training set.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm

# np.meshgrid_xx代表x軸，yy代表y軸，做2D平面圖
xx, yy = np.meshgrid(np.linspace(-7, 7, 500), np.linspace(-7, 7, 500))
# Generate train data
X = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X + 3, X - 2]
X_trainthree = np.r_[X, X]
# Generate some regular novel observations
X = 0.3 * np.random.randn(20, 2)
X_test = np.r_[X + 3, X - 2]
X_three = np.r_[X, X] #新增一群正常資料
# Generate some abnormal novel observations
# 從low跟high中間取等差值，大小為(20,2)
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# fit the model
# nu=判斷為異常的不超過10%，gumma越小，越分散，越不準確
clf = svm.OneClassSVM(nu=0.1, kernel="poly", gamma=0.1)
# fit是訓練資料，pred是返回預測值
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

# plot the line, the points, and the nearest vectors to the plane
# decision_function是返回個樣本點到超平面的函數距離
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("Novelty Detection")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s, edgecolors='k')
# 新增第三群的繪圖資料
b3 = plt.scatter(X_trainthree[:, 0], X_trainthree[:, 1], c='white', s=s, edgecolors='k')
b4 = plt.scatter(X_three[:, 0], X_three[:, 1], c='blueviolet', s=s, edgecolors='k')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,
                edgecolors='k')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([a.collections[0], b1, b2, c],
           ["learned frontier", "training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=11))
plt.xlabel( # 在x軸寫下列這些字
    "error train: %d/200 ; errors novel regular: %d/40 ; "
    "errors novel abnormal: %d/40"
    % (n_error_train, n_error_test, n_error_outliers))
plt.show()
