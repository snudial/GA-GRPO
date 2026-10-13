from types import SimpleNamespace

import molt.trainer.rl_trainer as runtime


def test_due_save_waits_for_drain_and_does_not_replace_latest_early(monkeypatch):
    saved=[]
    trainer=object.__new__(runtime.BaseRLTrainer)
    trainer.args=SimpleNamespace(logger=SimpleNamespace(logging_steps=1),ckpt=SimpleNamespace(save_steps=5))
    trainer.wandb_logger=trainer.tensorboard_logger=None
    trainer.best_eval_metric_key=None;trainer.best_eval_metric_value=.3
    trainer._latest_eval_metric_value=None;trainer.critic_model_group=None
    trainer.actor_model_group=SimpleNamespace(async_run_method=lambda **kw:saved.append(kw) or [])
    monkeypatch.setattr(runtime.ray,'get',lambda value:value)
    def state(pending,finished=0):
        return {'rollout_generator_state_dict':{'pending_rollouts':pending,'pending_finished_samples':finished}}
    trainer.save_logs_and_checkpoints(5,client_states=state(15))
    trainer.save_logs_and_checkpoints(6,client_states=state(0,1))
    assert saved==[]
    trainer.save_logs_and_checkpoints(7,client_states=state(0))
    assert [x['tag'] for x in saved]==['global_step7']
    trainer.save_logs_and_checkpoints(8,client_states=state(0))
    assert len(saved)==1
    trainer.save_logs_and_checkpoints(9,client_states=state(0),force=True)
    assert [x['tag'] for x in saved]==['global_step7','global_step9']
