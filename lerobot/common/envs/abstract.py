import abc
from collections import deque
from typing import Optional

from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.envs.utils import _terminated_or_truncated, step_mdp


class EnvBaseWithMultiStepRollouts(EnvBase):
    """Adds handling of policies that output action trajectories to be execute with a fixed horizon."""

    def _rollout_stop_early(
        self,
        *,
        tensordict,
        auto_cast_to_device,
        max_steps,
        policy,
        policy_device,
        env_device,
        callback,
    ):
        """Override adds handling of multi-step policies."""
        tensordicts = []
        step_ix = 0
        do_break = False
        while not do_break:
            if auto_cast_to_device:
                if policy_device is not None:
                    tensordict = tensordict.to(policy_device, non_blocking=True)
                else:
                    tensordict.clear_device_()
            tensordict = policy(tensordict)
            if auto_cast_to_device:
                if env_device is not None:
                    tensordict = tensordict.to(env_device, non_blocking=True)
                else:
                    tensordict.clear_device_()

            for action in tensordict["action"].clone():
                tensordict["action"] = action
                tensordict = self.step(tensordict)
                tensordicts.append(tensordict.clone(False))

                if step_ix == max_steps - 1:
                    # we don't truncated as one could potentially continue the run
                    do_break = True
                    break
                tensordict = step_mdp(
                    tensordict,
                    keep_other=True,
                    exclude_action=False,
                    exclude_reward=True,
                    reward_keys=self.reward_keys,
                    action_keys=self.action_keys,
                    done_keys=self.done_keys,
                )
                # done and truncated are in done_keys
                # We read if any key is done.
                any_done = _terminated_or_truncated(
                    tensordict,
                    full_done_spec=self.output_spec["full_done_spec"],
                    key=None,
                )
                if any_done:
                    break

                if callback is not None:
                    callback(self, tensordict)

                step_ix += 1

        return tensordicts

    def _rollout_nonstop(
        self,
        *,
        tensordict,
        auto_cast_to_device,
        max_steps,
        policy,
        policy_device,
        env_device,
        callback,
    ):
        """Override adds handling of multi-step policies."""
        tensordicts = []
        tensordict_ = tensordict
        for i in range(max_steps):
            if auto_cast_to_device:
                if policy_device is not None:
                    tensordict_ = tensordict_.to(policy_device, non_blocking=True)
                else:
                    tensordict_.clear_device_()
            tensordict_ = policy(tensordict_)
            if auto_cast_to_device:
                if env_device is not None:
                    tensordict_ = tensordict_.to(env_device, non_blocking=True)
                else:
                    tensordict_.clear_device_()
            tensordict, tensordict_ = self.step_and_maybe_reset(tensordict_)
            tensordicts.append(tensordict)
            if i == max_steps - 1:
                # we don't truncated as one could potentially continue the run
                break
            if callback is not None:
                callback(self, tensordict)

        return tensordicts


class AbstractEnv(EnvBaseWithMultiStepRollouts):
    def __init__(
        self,
        task,
        frame_skip: int = 1,
        from_pixels: bool = False,
        pixels_only: bool = False,
        image_size=None,
        seed=1337,
        device="cpu",
        num_prev_obs=1,
        num_prev_action=0,
    ):
        super().__init__(device=device, batch_size=[])
        self.task = task
        self.frame_skip = frame_skip
        self.from_pixels = from_pixels
        self.pixels_only = pixels_only
        self.image_size = image_size
        self.num_prev_obs = num_prev_obs
        self.num_prev_action = num_prev_action
        self._rendering_hooks = []

        if pixels_only:
            assert from_pixels
        if from_pixels:
            assert image_size

        self._make_env()
        self._make_spec()
        self._current_seed = self.set_seed(seed)

        if self.num_prev_obs > 0:
            self._prev_obs_image_queue = deque(maxlen=self.num_prev_obs)
            self._prev_obs_state_queue = deque(maxlen=self.num_prev_obs)
        if self.num_prev_action > 0:
            raise NotImplementedError()
            # self._prev_action_queue = deque(maxlen=self.num_prev_action)

    def register_rendering_hook(self, func):
        self._rendering_hooks.append(func)

    def call_rendering_hooks(self):
        for func in self._rendering_hooks:
            func(self)

    def reset_rendering_hooks(self):
        self._rendering_hooks = []

    @abc.abstractmethod
    def render(self, mode="rgb_array", width=640, height=480):
        raise NotImplementedError()

    @abc.abstractmethod
    def _reset(self, tensordict: Optional[TensorDict] = None):
        raise NotImplementedError()

    @abc.abstractmethod
    def _step(self, tensordict: TensorDict):
        raise NotImplementedError()

    @abc.abstractmethod
    def _make_env(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def _make_spec(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def _set_seed(self, seed: Optional[int]):
        raise NotImplementedError()
