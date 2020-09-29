from keras import backend as K
from keras.optimizers import *


class Ada_heavy_ball(Optimizer):

    def __init__(self, lr=0.001, gamma=0.1, delta=1e-8,
                 epsilon=None, decay=0., **kwargs):
        super(Ada_heavy_ball, self).__init__(**kwargs)
        self.iterations = K.variable(0, dtype='int64', name='iterations')
        self.lr = K.variable(lr, name='lr')
        self.decay = K.variable(decay, name='decay')
        self.delta = delta
        self.gamma = K.variable(gamma, name='gamma')
        self.initial_decay = decay

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        t = K.cast(self.iterations, K.floatx()) + 1
        epoch = (t // 98) + 1  # CIFAR10, CIFAR100
        # epoch = (t//118) + 1    #  MNIST
        beta2 = 1 - (self.gamma / epoch)
        alpha_t = lr / ((epoch + 2) * (epoch ** (1 / 2)))
        momentum = epoch / (epoch + 2)

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            v_t = (beta2 * v) + (1 - beta2) * K.square(g)
            v_hat = (K.sqrt(v_t)) + (self.delta / (epoch ** (1 / 2)))
            m_t = momentum * m + alpha_t * g / v_hat

            p_t = p - m_t

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'gamma': float(K.get_value(self.gamma)),
                  'decay': float(K.get_value(self.decay))}
        base_config = super(Ada_heavy_ball, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SGD_mom(Optimizer):

    def __init__(self, lr=0.01, momentum=0.9, decay=0.,
                 nesterov=False, **kwargs):
        super(SGD_mom, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        # if self.initial_decay > 0:
        #     lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
        #                                               K.dtype(self.decay))))

        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m - lr * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(SGD_mom, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
