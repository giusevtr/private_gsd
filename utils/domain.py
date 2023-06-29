from functools import reduce
import jax.numpy as jnp

class Domain:
    def __init__(self, config: dict, null_cols: list = (), bin_edges: dict = None):
        """ Construct a Domain object
        
        :param attrs: a list or tuple of attribute names
        :param shape: a list or tuple of domain sizes for each attribute
        """
        self.attrs = list(config.keys())
        self.config = config


        self.null_cols = null_cols
        self._is_col_null = self._set_null_cols(null_cols)

        # Edges of real-value features used for discretization.
        self._bin_edges = bin_edges

    def has_nulls(self, col):
        return self._is_col_null[col]

    def _set_null_cols(self, null_cols):
        is_col_null = {}
        for col in self.attrs:
            is_col_null[col] = col in null_cols
        return is_col_null

    def get_bin_edges(self, col):
        """ Returns: First argument is a boolean indicating if col has bin edges defined. """
        if self._bin_edges is None:
            return False, None
        if col not in self._bin_edges:
            return False, None
        return True, jnp.array(self._bin_edges[col])

    def project(self, attrs):
        """ project the domain onto a subset of attributes
        
        :param attrs: the attributes to project onto
        :return: the projected Domain object
        """
        # return the projected domain
        if type(attrs) is str:
            attrs = [attrs]
        # shape = tuple(self.config[a] for a in attrs)
        new_config = {}
        for a in attrs:
            new_config[a] = self.config[a]
        return Domain(new_config, null_cols=self.null_cols, bin_edges=self._bin_edges)

    def axes(self, attrs):
        """ return the axes tuple for the given attributes

        :param attrs: the attributes
        :return: a tuple with the corresponding axes
        """
        return tuple(self.attrs.index(a) for a in attrs)

    def transpose(self, attrs):
        """ reorder the attributes in the domain object """
        return self.project(attrs)

    def invert(self, attrs):
        """ returns the attributes in the domain not in the list """
        return [a for a in self.attrs if a not in attrs]

    def contains(self, other):
        """ determine if this domain contains another

        """
        return set(other.attrs) <= set(self.attrs)

    def __contains__(self, attr):
        return attr in self.attrs

    def __getitem__(self, a):
        """ return the size of an individual attribute
        :param a: the attribute
        """
        return self.config[a]

    def __iter__(self):
        """ iterator for the attributes in the domain """
        return self.attrs.__iter__()

    def __len__(self):
        return len(self.attrs)

    def __eq__(self, other):
        return self.config == other.config

    def get_numerical_cols(self):
        n_cols = []
        for c in self.attrs:
            if self.config[c]['type'] == 'numerical':
                n_cols.append(c)
        return n_cols

    def get_ordinal_cols(self):
        n_cols = []
        for c in self.attrs:
            if self.config[c]['type'] == 'ordinal':
                n_cols.append(c)
        return n_cols

    def get_categorical_cols(self):
        c_cols = []
        for c in self.attrs:
            if self.config[c]['type'] == 'categorical':
                c_cols.append(c)
        return c_cols


    def type(self, att):
        return self.config[att]['type']

    def size(self, att):
        return self.config[att]['size']

    def get_attribute_indices(self, atts):
        indices = []
        for i, temp in enumerate(self.attrs):
            if temp not in atts:
               continue
            indices.append(i)
        return jnp.array(indices)
