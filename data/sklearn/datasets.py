#exec(open('.\\trees\\sklearn\\datasets.py').read())
import subprocess as sp
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_digits
from sklearn.datasets import load_linnerud
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer

if __name__ == '__main__':
    sp.call('cls', shell = True)

    # load the iris dataset
    ds = dict()
    ds['iris'] = load_iris()
    ds['boston'] = load_boston()
    ds['iris'] = load_iris()
    ds['diabetes'] = load_diabetes()
    ds['digits'] = load_digits()
    ds['linnerud'] = load_linnerud()
    ds['wine'] = load_wine()
    ds['breastcancer'] = load_breast_cancer()

    # print the keys of every dataset
    for key in ds:
        print(key)
        for key2 in ds[key]:
            print('{0}{1}'.format('    ', key2))
