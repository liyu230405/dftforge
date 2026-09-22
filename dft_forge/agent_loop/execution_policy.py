"""Product execution policy applied to planned tools before any dispatch."""

import os


class ExecutionPolicyError(ValueError):
    def __init__(self, message, code="execution_policy"):
        super().__init__(message)
        self.code = code


def check_execution_policy(steps, ctx, registry) -> None:
    config = ctx.get("run_config") or {}
    mode = config.get("mode")
    if mode is None:
        return  # CLI/integration callers without a product mode retain their API.
    compute_steps = [
        step for step in steps
        if (tool := registry.get(step.get("tool"))) is not None
        and tool.side_effect == "external_execution"
    ]
    if not compute_steps:
        return
    backend = os.environ.get("DFT_FORGE_EXECUTOR", "local").strip().lower() or "local"
    if mode == "research":
        if not config.get("approved"):
            raise ExecutionPolicyError(
                "科研模式需要先确认计算方法；请确认方法后重试", code="approval_required",
            )
        if backend == "fake":
            raise ExecutionPolicyError("科研模式不能使用模拟后端；请在连接中心选择本地或 SSH，或切换为演示模式")
    elif mode == "demo":
        if backend != "fake":
            raise ExecutionPolicyError("演示模式需要模拟后端；请在连接中心选择演示后端后重试")
    else:
        raise ExecutionPolicyError("运行模式必须是科研或演示")
    for step in compute_steps:
        # The legacy single-job tool chooses its own CLI executor; it must
        # not bypass the workspace's selected compute destination.
        if step.get("tool") == "job.submit":
            actual = (step.get("args") or {}).get("backend", "local")
            if actual != backend:
                raise ExecutionPolicyError("单任务执行器与连接中心配置不一致；请使用图工作流或选择匹配的计算后端")
