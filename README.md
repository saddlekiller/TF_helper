# TF_helper

***Author: Cheng.Bi; Email: alan_debug@icloud.com***

![Language](https://img.shields.io/badge/language-Python%20%2F%20Shell%20%20%2F%20Tensorflow%20-orange.svg)

***Version: 1.0.0***

---

This python toolkit is supposed to bring convenience to adding summaries or freezing TF models.

### How to use?
    git clone https://github.com/ChengBi/TF_helper
    cd TF_helper
    python setup.py install
    
Then in Python Scripts:
    
    import tfhelper.helper as tfhelper
    summary = tfhelper.Summary(logdir, max_outputs, sampling_rate)
        .
        .
        .
    summary.merge()
    summary.save(sess, global_step, var_list)
    
    saver = tfhelper.Saver(var_list, max_to_keep, keep_checkpoint_every_n_hours)
    if restore:
        step = self.saver_helper.restore(session, save_path)
    if save:
        saver.save(session, save_path, global_step)





