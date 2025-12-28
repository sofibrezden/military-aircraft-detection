import warnings
from math import inf, isfinite
from typing import ClassVar

from mmcv.runner import HOOKS, Hook, Runner

DATA_BATCH = dict | tuple | list | None


@HOOKS.register_module()
class EarlyStoppingHook(Hook):
    """Early stop the training when the monitored metric reached a plateau.

    Args:
        monitor (str): The monitored metric key to decide early stopping.
        rule (str, optional): Comparison rule. Options are 'greater',
            'less'. Defaults to None.
        min_delta (float, optional): Minimum difference to continue the
            training. Defaults to 0.01.
        strict (bool, optional): Whether to crash the training when `monitor`
            is not found in the `metrics`. Defaults to False.
        check_finite: Whether to stop training when the monitor becomes NaN or
            infinite. Defaults to True.
        patience (int, optional): The times of validation with no improvement
            after which training will be stopped. Defaults to 5.
        stopping_threshold (float, optional): Stop training immediately once
            the monitored quantity reaches this threshold. Defaults to None.

    Note:
        `New in version 0.7.0.`

    """

    rule_map: ClassVar[dict[str, callable]] = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    _default_greater_keys: ClassVar[list[str]] = ['acc', 'top', 'AR@', 'auc', 'precision', 'mAP', 'mDice', 'mIoU', 'mAcc', 'aAcc']
    _default_less_keys: ClassVar[list[str]] = ['loss']

    def __init__(
        self,
        monitor: str,
        rule: str | None = None,
        min_delta: float = 0.1,
        *,
        strict: bool = False,
        check_finite: bool = True,
        patience: int = 5,
        stopping_threshold: float | None = None,
    ) -> None:
        self.monitor = monitor
        if rule is not None:
            if rule not in ['greater', 'less']:
                raise ValueError(f'`rule` should be either "greater" or "less", but got {rule}')
        else:
            rule = self._init_rule(monitor)
        self.rule = rule
        self.min_delta = min_delta if rule == 'greater' else -1 * min_delta
        self.strict = strict
        self.check_finite = check_finite
        self.patience = patience
        self.stopping_threshold = stopping_threshold

        self.wait_count = 0
        self.best_score = -inf if rule == 'greater' else inf

    def _init_rule(self, monitor: str) -> str:
        greater_keys = {key.lower() for key in self._default_greater_keys}
        less_keys = {key.lower() for key in self._default_less_keys}
        monitor_lc = monitor.lower()
        if monitor_lc in greater_keys:
            rule = 'greater'
        elif monitor_lc in less_keys:
            rule = 'less'
        elif any(key in monitor_lc for key in greater_keys):
            rule = 'greater'
        elif any(key in monitor_lc for key in less_keys):
            rule = 'less'
        else:
            raise ValueError(f'Cannot infer the rule for {monitor}, thus rule must be specified.')
        return rule

    def _check_stop_condition(self, current_score: float) -> tuple[bool, str]:
        compare = self.rule_map[self.rule]
        stop_training = False
        reason_message = ''

        if self.check_finite and not isfinite(current_score):
            stop_training = True
            reason_message = (
                f'Monitored metric {self.monitor} = {current_score} is infinite. Previous best value was {self.best_score:.3f}.'
            )

        elif self.stopping_threshold is not None and compare(current_score, self.stopping_threshold):
            stop_training = True
            self.best_score = current_score
            reason_message = (
                f'Stopping threshold reached: `{self.monitor}` = {current_score} is {self.rule} than {self.stopping_threshold}.'
            )
        else:
            # Check if current_score is an improvement over best_score
            # For 'greater' rule: improvement if current_score > best_score + min_delta
            # For 'less' rule: improvement if current_score < best_score + min_delta (min_delta is negative)
            is_initial = (self.rule == 'greater' and self.best_score == -inf) or (self.rule == 'less' and self.best_score == inf)

            if is_initial:
                # First call: set baseline and start counting
                self.best_score = current_score
                self.wait_count = 1
            else:
                is_improvement = compare(current_score, self.best_score + self.min_delta)

                if is_improvement:
                    self.best_score = current_score
                    self.wait_count = 0
                else:
                    self.wait_count += 1
                    if self.wait_count > self.patience:
                        reason_message = (
                            f'the monitored metric did not improve in the last {self.wait_count} records. '
                            f'best score: {self.best_score:.3f}. '
                            f'patience: {self.patience} exceeded.'
                        )
                        stop_training = True

        return stop_training, reason_message

    def before_run(self, runner: Runner) -> None:
        """Initialize `stop_training` variable on runner if it doesn't exist.

        Args:
            runner (Runner): The runner of the training process.

        """
        if not hasattr(runner, 'stop_training'):
            runner.stop_training = False

    def after_val_epoch(self, runner: Runner, metrics: dict) -> None:
        """Decide whether to stop the training process.

        Args:
            runner (Runner): The runner of the training process.
            metrics (dict): Evaluation results of all metrics

        """
        if self.monitor not in metrics:
            if self.strict:
                raise RuntimeError(
                    'Early stopping conditioned on metric '
                    f'`{self.monitor} is not available. Please check available'
                    f' metrics {metrics}, or set `strict=False` in '
                    '`EarlyStoppingHook`.'
                )
            warnings.warn(
                f'Skip early stopping process since the evaluation results ({metrics.keys()}) do not include `monitor` ({self.monitor}).',
                stacklevel=2,
            )
            return

        current_score = metrics[self.monitor]

        stop_training, message = self._check_stop_condition(current_score)
        if stop_training:
            runner.stop_training = True
            runner.logger.info(message)
        # Log progress if we're waiting for improvement
        elif self.wait_count > 0:
            remaining = self.patience - self.wait_count
            runner.logger.info(
                f'EarlyStopping: {self.monitor} = {current_score:.4f} (best: {self.best_score:.4f}), '
                f'no improvement for {self.wait_count}/{self.patience} epochs. '
                f'Remaining patience: {remaining}'
            )

    def after_train_epoch(self, runner: Runner) -> None:
        """Check if training should be stopped after training epoch.

        Args:
            runner (Runner): The runner of the training process.

        """
        if hasattr(runner, 'stop_training') and runner.stop_training:
            runner.should_stop = True
            runner.logger.info('Early stopping triggered. Training will stop after current epoch.')
