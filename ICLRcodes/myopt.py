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


class SAdam(Optimizer):
    def __init__(self, beta_1=0.9, lr=0.01, delta=1e-2, xi_1=0.1, xi_2=0.1,
                 gamma=0.9, decay=0., vary_delta=False, **kwargs):
        super(SAdam, self).__init__(**kwargs)
        self.iterations = K.variable(0, name='iterations')
        self.beta_1 = K.variable(beta_1, name='beta_1')
        self.lr = K.variable(lr, name='lr')
        self.delta = delta
        self.xi_1 = K.variable(xi_1, name='xi_1')
        self.xi_2 = K.variable(xi_2, name='xi_2')
        self.gamma = K.variable(gamma, name='gamma')
        self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.vary_delta = vary_delta

    def get_updates(self, params, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        t = K.cast(self.iterations, K.floatx()) + 1

        vs = [K.zeros(K.get_variable_shape(p)) for p in params]
        gs = [K.zeros(K.get_variable_shape(p)) for p in params]
        self.weights = [self.iterations] + vs + gs
        lr_t = self.lr
        for p, g, v, gg in zip(params, grads, vs, gs):
            hat_g_t = self.beta_1 * gg + (1 - self.beta_1) * g
            v_t = (1 - (self.gamma / t)) * v + (self.gamma / t) * K.square(g)
            if self.vary_delta:
                p_t = p - lr_t * hat_g_t / (t * v_t + self.xi_2 * K.exp(-self.xi_1 * t * v_t))
            else:
                p_t = p - lr_t * hat_g_t / (t * v_t + self.delta)

            self.updates.append(K.update(v, v_t))
            self.updates.append(K.update(gg, hat_g_t))
            new_p = p_t
            # apply constraints
            if getattr(p, 'constraints', None) is not None:
                new_p = p.constraints(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'xi_1': float(K.get_value(self.xi_1)),
                  'xi_2': float(K.get_value(self.xi_2)),
                  'gamma': float(K.get_value(self.gamma)),
                  'decay': float(K.get_value(self.decay))}
        base_config = super(SAdam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Amsgrad(Optimizer):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0., amsgrad=True, **kwargs):
        super(Amsgrad, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

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
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad}
        base_config = super(Amsgrad, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SC_Adagrad(Optimizer):

    def __init__(self, lr=0.01, xi_1=0.1, xi_2=0.1,
                 decay=0., **kwargs):
        super(SC_Adagrad, self).__init__(**kwargs)
        self.iterations = K.variable(0, name='iterations')
        self.lr = K.variable(lr, name='lr')
        self.xi_1 = K.variable(xi_1, name='xi_1')
        self.xi_2 = K.variable(xi_2, name='xi_2')
        self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay

    def get_updates(self, params, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        vs = [K.zeros(K.get_variable_shape(p)) for p in params]
        self.weights = [self.iterations] + vs

        for p, g, v in zip(params, grads, vs):
            v_t = v + K.square(g)
            p_t = p - self.lr * g / (v_t + self.xi_2 * K.exp(-self.xi_1 * v_t))

            self.updates.append(K.update(v, v_t))

            new_p = p_t
            # apply constraints
            if getattr(p, 'constraints', None) is not None:
                new_p = p.constraints(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'xi_1': float(K.get_value(self.xi_1)),
                  'xi_2': float(K.get_value(self.xi_2)),
                  'decay': float(K.get_value(self.decay))}
        base_config = super(SC_Adagrad, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SC_RMSprop(Optimizer):

    def __init__(self, lr=0.01, xi_1=0.1, xi_2=0.1,
                 gamma=0.9, decay=0., **kwargs):
        super(SC_RMSprop, self).__init__(**kwargs)
        self.iterations = K.variable(0, name='iterations')
        self.lr = K.variable(lr, name='lr')
        self.xi_1 = K.variable(xi_1, name='xi_1')
        self.xi_2 = K.variable(xi_2, name='xi_2')
        self.gamma = K.variable(gamma, name='gamma')
        self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay

    def get_updates(self, params, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 1

        vs = [K.zeros(K.get_variable_shape(p)) for p in params]
        self.weights = [self.iterations] + vs

        for p, g, v in zip(params, grads, vs):
            v_t = (1 - (self.gamma / t)) * v + (self.gamma / t) * K.square(g)
            p_t = p - self.lr * g / (t * v_t + self.xi_2 * K.exp(-self.xi_1 * t * v_t))

            self.updates.append(K.update(v, v_t))

            new_p = p_t
            # apply constraints
            if getattr(p, 'constraints', None) is not None:
                new_p = p.constraints(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'xi_1': float(K.get_value(self.xi_1)),
                  'xi_2': float(K.get_value(self.xi_2)),
                  'gamma': float(K.get_value(self.gamma)),
                  'decay': float(K.get_value(self.decay))}
        base_config = super(SC_RMSprop, self).get_config()
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


class AdamNC(Optimizer):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0., **kwargs):
        super(AdamNC, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        self.epsilon = epsilon

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        t = K.cast(self.iterations, K.floatx()) + 1

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs
        beta_2_t = 1 - (1 / t)
        lr_t = self.lr * (K.sqrt(1. - K.pow(beta_2_t, t)) /
                          (1. - K.pow(self.beta_1, t)))
        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (beta_2_t * v) + (1. - beta_2_t) * K.square(g)
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)
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
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(AdamNC, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))