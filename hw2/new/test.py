from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Imputer


def cross_validate_single_kernel(x, y, kernel, n_folds, args):

    ErrorList = []
    nsv = 0
    nsvm = 0
    # Do a k folds Stratified cross_validation
    skf=StratifiedKFold(n_splits=n_folds, shuffle=True)
    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index].values.ravel(), y.iloc[test_index].values.ravel()

        
        svm = SVC(C=args["c_val"], kernel=kernel, tol=1e-7, shrinking=False, degree = args["d_val"], gamma = 'auto')
        svm.fit(x_train, y_train)

        ErrorList.append(len(y_test)-((svm.score(x_test, y_test))*len(y_test)))
        sv_list   =  svm.support_
        svx       =  svm.support_vectors_
        beta_list = abs(np.float16((svm.decision_function(svx))))
        index = [i for i, j in enumerate(beta_list) if j == 1]

        #pred   =   svm.predict(x_test)
        #auc.append(roc_auc_score(y_test, pred))
        nsv   +=   np.mean(svm.n_support_)/n_folds
        nsvm  +=   len(index)/n_folds

    return ErrorList, nsv, nsvm
