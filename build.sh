pyinstaller app.py --onefile --clean --hidden-import \
scipy.special.cython_special --hidden-import sklearn.neighbors._typedefs \
--hidden-import sklearn.utils._cython_blas --hidden-import \
sklearn.neighbors._quad_tree --hidden-import sklearn.tree._utils
