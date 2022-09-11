import lightgbm as _lgb

class LightGbm:
    def train(self, X_train, y_train, X_val, y_val, **kwargs):
        """学習

        **kwargs:
            https://lightgbm.readthedocs.io/en/latest/Parameters.html#core-parameters
        """
        ds_train = _lgb.Dataset(X_train, label=y_train)
        ds_val = _lgb.Dataset(X_val, label=y_val, reference=ds_train)
        self._model = _lgb.train(params=kwargs,train_set=ds_train, valid_sets=[ds_val])

    def predict(self, data):
        return self._model.predict(data, num_iteration=self._model.best_iteration)