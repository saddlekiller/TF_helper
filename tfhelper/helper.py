import tensorflow as tf
import os

class Summary:

    def __init__(self, logdir, max_outputs=5, sampling_rate=None):
        self._summary_dict = {
            'audio':{},
            'image':{},
            'histogram':{},
            'scalar':{}
        }
        self.sampling_rate = sampling_rate
        self.max_outputs = max_outputs
        self.logdir = logdir
        self.merged = False
        os.makedirs(logdir, exist_ok=True)

    def add(self, summary_type, summary_scope, summary_name, tensor):
        if self.merged:
            raise ValueError
        if summary_scope not in self._summary_dict[summary_type].keys():
            self._summary_dict[summary_type][summary_scope] = []
        self._summary_dict[summary_type][summary_scope].append((summary_name, tensor))

    def merge(self):
        if len(self._summary_dict['audio'].keys()) > 0 and self.sampling_rate is None:
            raise ValueError
        for _scope, values in self._summary_dict['audio'].items():
            with tf.variable_scope(_scope):
                for _name, _tensor in values:
                    _name = _name.replace(':', '_')
                    if len(_tensor.get_shape().as_list()) == 3:
                        _tensor = tf.squeeze(_tensor, -1)
                    elif len(_tensor.get_shape().as_list()) == 1:
                        _tensor = tf.expand_dims(_tensor, 0)
                    if len(_tensor.get_shape().as_list()) != 2:
                        tf.logging.error('Audio Error, Wrong shape for tensor {}, please check the dimensionality!'.format(_tensor))
                        raise ValueError
                    tf.summary.audio(_name, _tensor, self.sampling_rate, max_outputs=self.max_outputs)
        for _scope, values in self._summary_dict['image'].items():
            with tf.variable_scope(_scope):
                for _name, _tensor in values:
                    _name = _name.replace(':', '_')
                    if len(_tensor.get_shape().as_list()) == 2:
                        _tensor = tf.expand_dims(_tensor, 0)
                        _tensor = tf.expand_dims(_tensor, -1)
                    elif len(_tensor.get_shape().as_list()) == 3:
                        _tensor = tf.expand_dims(_tensor, -1)
                    elif len(_tensor.get_shape().as_list()) == 4:
                        pass
                    else:
                        tf.logging.error('Image Error, Wrong shape for tensor {}, please check the dimensionality!'.format(_tensor))
                        raise ValueError
                    tf.summary.image(_name, _tensor, max_outputs=self.max_outputs)
        for _scope, values in self._summary_dict['histogram'].items():
            with tf.variable_scope(_scope):
                for _name, _tensor in values:
                    _name = _name.replace(':', '_')
                    if len(_tensor.get_shape().as_list()) == 2:
                        _tensor = tf.expand_dims(_tensor, 0)
                        _tensor = tf.expand_dims(_tensor, -1)
                    elif len(_tensor.get_shape().as_list()) == 3:
                        _tensor = tf.expand_dims(_tensor, -1)
                    elif len(_tensor.get_shape().as_list()) == 4:
                        pass
                    elif len(_tensor.get_shape().as_list()) == 1:
                        _tensor = tf.reshape(_tensor, [1, 1, -1, 1])
                    else:
                        tf.logging.error('Histogram Error, Wrong shape for tensor {}, please check the dimensionality!'.format(_tensor))
                        raise ValueError
                    tf.summary.histogram(_name, _tensor)
        for _scope, values in self._summary_dict['scalar'].items():
            with tf.variable_scope(_scope):
                for _name, _tensor in values:
                    _name = _name.replace(':', '_')
                    if len(_tensor.get_shape().as_list()) == 0 or len(_tensor.get_shape().as_list()) == 1:
                        pass
                    else:
                        tf.logging.error('Scalar Error, Wrong shape for tensor {}, please check the dimensionality!'.format(_tensor))
                        raise ValueError
                    tf.summary.scalar(_name, _tensor)
        self.stats = tf.summary.merge_all()
        self.filewriter = tf.summary.FileWriter(self.logdir)
        self.merged = True

    def save(self, sess, global_step, var_list=None):
        if not self.merged:
            tf.logging.error('Must merge all summaries before saving summaries!')
            raise ValueError
        if var_list is None:
            self.filewriter.add_summary(sess.run(self.stats), global_step)
            var = None
        else:
            stats, var = sess.run([self.stats, var_list])
            self.filewriter.add_summary(stats, global_step)
        tf.logging.info('Saved summary at {} '.format(global_step))
        return var

class Saver:

    def __init__(self, var_list, max_to_keep=10, keep_checkpoint_every_n_hours=1):
        self.saver = tf.train.Saver(var_list=var_list, max_to_keep=max_to_keep,
                                    keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours)

    def save(self, session, save_path, global_step=None):
        os.makedirs(save_path, exist_ok=True)
        if save_path.find('ckpt') == -1:
            save_path = os.path.join(save_path, 'model.ckpt')
        self.saver.save(sess=session, save_path=save_path, global_step=global_step)

    def restore(self, session, save_path):
        os.makedirs(save_path, exist_ok=True)
        if save_path.find('ckpt') != -1:
            step = save_path.split('-')[-1]
            self._restore(session=session, save_path=save_path)
        else:
            checkpoint_state = tf.train.get_checkpoint_state(save_path)
            if checkpoint_state and checkpoint_state.model_checkpoint_path:
                step = checkpoint_state.model_checkpoint_path.split('-')[-1]
                self._restore(session=session, save_path=checkpoint_state.model_checkpoint_path)
            else:
                tf.logging.info('Found no checkpoints in {}'.format(save_path))
                step = 0
                raise ValueError
        return step

    def _restore(self, session, save_path):
        try:
            self.saver.restore(sess=session, save_path=save_path)
        except Exception as e:
            tf.logging.info('Failed to restore models due to {}'.format(e))
            raise ValueError
