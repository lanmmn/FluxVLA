from abc import ABC, abstractmethod  # ABC: 抽象基类支持; abstractmethod: 标记子类必须实现的方法
from typing import Any               # Any: 类型注解,表示任意类型


class BasePolicy(ABC):
    """策略抽象基类,定义了所有策略(本地推理/远程推理)必须遵循的接口。"""

    @abstractmethod  # 子类必须实现此方法,否则无法实例化
    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """核心推理方法(内部接口),子类在这里实现具体的动作计算逻辑。"""
        pass

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """公开推理接口,调用子类实现的 _get_action。"""
        return self._get_action(observation, options)

    @abstractmethod  # 子类必须实现此方法
    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """重置策略状态(如清除历史帧缓存),返回重置后的信息字典。"""
        pass
