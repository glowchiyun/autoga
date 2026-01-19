"""
智能超参数范围调整器
基于高斯分布的自适应超参数范围调整
"""
import numpy as np
import logging


class IntelligentRangeAdjuster:
    """基于高斯分布的自适应超参数范围调整器"""
    
    def __init__(self, adjustment_factor=0.5):
        self.param_history = {}  # {model_param:[values], fitness:[scores]}
        self.adjustment_factor = adjustment_factor  # 调整激进度(0-1)
        self.adjustment_count = 0  # 调整次数统计
        
        # 定义常见超参数的合法范围约束
        self.param_bounds = {
            'colsample_bytree': (0, 1),
            'subsample': (0, 1),
            'learning_rate': (0.001, 1),
            'gamma': (0, float('inf')),
            'reg_alpha': (0, float('inf')),
            'reg_lambda': (0, float('inf')),
            'min_child_weight': (0, float('inf')),
            'max_depth': (1, 50),
            'n_estimators': (10, 1000),
            'alpha': (0, float('inf')),
            'l1_ratio': (0, 1),
            'C': (0.001, float('inf')),
        }
    
    def record_evaluation(self, model_name, param_name, param_value, fitness):
        """记录每个超参数的评估结果"""
        key = f"{model_name}_{param_name}"
        if key not in self.param_history:
            self.param_history[key] = {'values': [], 'fitness': []}
        
        self.param_history[key]['values'].append(param_value)
        self.param_history[key]['fitness'].append(fitness)
    
    def _get_param_bounds(self, param_name, original_values):
        """获取参数的合法范围约束
        
        :param param_name: 参数名称
        :param original_values: 原始值列表
        :return: (min_bound, max_bound)
        """
        # 如果有预定义的边界，使用预定义的
        if param_name in self.param_bounds:
            return self.param_bounds[param_name]
        
        # 否则从原始值推断
        if not original_values:
            return (0, float('inf'))
        
        min_val = min(original_values)
        max_val = max(original_values)
        
        # 如果所有原始值都在[0,1]之间，假设这是一个比例参数
        if max_val <= 1.0 and min_val >= 0:
            return (0, 1)
        
        # 否则给一个宽松的范围
        if min_val >= 0:
            return (0, max_val * 3)  # 允许扩展到原最大值的3倍
        else:
            return (min_val * 3, max_val * 3)
    
    def compute_optimal_range(self, model_name, param_name, original_values):
        """基于评估历史计算最优范围 - 改进版：在最优值周围生成新的搜索点
        
        :param model_name: 模型名称
        :param param_name: 超参数名称
        :param original_values: 原始参数范围列表
        :return: 调整后的参数范围
        """
        key = f"{model_name}_{param_name}"
        
        if key not in self.param_history or not self.param_history[key]['values']:
            return original_values
        
        values = np.array(self.param_history[key]['values'])
        fitness = np.array(self.param_history[key]['fitness'])
        
        # 过滤失败的评估（-inf值）
        valid_mask = fitness != -np.inf
        if np.sum(valid_mask) < 3:
            return original_values  # 样本过少，不调整
        
        values = values[valid_mask]
        fitness = fitness[valid_mask]
        
        # 检查参数类型
        # 1. 处理包含None的情况
        has_none = None in original_values
        numeric_originals = [v for v in original_values if v is not None and isinstance(v, (int, np.integer, float, np.floating))]
        non_numeric = [v for v in original_values if not isinstance(v, (int, np.integer, float, np.floating))]
        
        # 2. 如果原始值中包含字符串等非数值（不包括None），则不调整
        if non_numeric and not (len(non_numeric) == 1 and non_numeric[0] is None):
            logging.debug(f"{model_name}.{param_name}: 包含非数值参数，跳过范围调整")
            return original_values
        
        # 3. 提取历史中的数值参数
        numeric_values = []
        corresponding_fitness = []
        for v, f in zip(values, fitness):
            if v is not None and isinstance(v, (int, np.integer, float, np.floating)):
                numeric_values.append(float(v))
                corresponding_fitness.append(f)
        
        if len(numeric_values) < 3:
            return original_values
        
        numeric_values = np.array(numeric_values)
        corresponding_fitness = np.array(corresponding_fitness)
        
        # 找到表现最好的top 30%参数值
        top_k = max(1, int(len(numeric_values) * 0.3))
        top_indices = np.argsort(-corresponding_fitness)[:top_k]
        best_values = numeric_values[top_indices]
        
        # 计算最优区域的中心和范围
        optimal_center = np.mean(best_values)
        optimal_std = np.std(best_values) if len(best_values) > 1 else np.std(numeric_values) * 0.5
        optimal_std = max(optimal_std, 0.01)  # 防止std过小
        
        # 判断参数类型：整数还是浮点数
        is_integer = all(isinstance(v, (int, np.integer)) for v in numeric_originals)
        
        # 获取参数的合法范围约束
        param_min, param_max = self._get_param_bounds(param_name, numeric_originals)
        
        # 在最优中心周围生成新的搜索点
        new_values = []
        
        # 保留最优的原始值
        for v in numeric_originals:
            if abs(v - optimal_center) <= 2 * optimal_std:
                new_values.append(v)
        
        # 在最优区域生成新的候选值
        if is_integer:
            # 整数参数：在中心附近生成等间距的整数
            min_val = max(param_min, int(optimal_center - 2 * optimal_std))
            max_val = min(param_max, int(optimal_center + 2 * optimal_std))
            step = max(1, (max_val - min_val) // 4)
            candidates = list(range(min_val, max_val + 1, step))
        else:
            # 浮点数参数：在中心附近生成对数或线性间距的值
            if optimal_center > 0.1:  # 使用对数空间
                log_center = np.log10(optimal_center)
                log_std = optimal_std / optimal_center  # 相对标准差
                candidates = np.logspace(log_center - 2*log_std, log_center + 2*log_std, 5)
            else:  # 使用线性空间
                candidates = np.linspace(max(param_min, optimal_center - 2*optimal_std), 
                                       min(param_max, optimal_center + 2*optimal_std), 5)
            candidates = [round(v, 4) for v in candidates]
            
        # 合并并去重，同时确保在合法范围内
        for c in candidates:
            if is_integer:
                c = int(c)
            if c not in new_values and param_min <= c <= param_max:
                new_values.append(c)
        
        # 如果新值太少，保留一些原始值
        if len(new_values) < 3:
            for v in sorted(numeric_originals):
                if v not in new_values:
                    new_values.append(v)
                if len(new_values) >= 4:
                    break
        
        # 如果原始值包含None，且None对应的特征表现也不错，则保留
        if has_none:
            none_fitness = [f for v, f in zip(values, fitness) if v is None]
            if none_fitness and np.mean(none_fitness) > np.percentile(corresponding_fitness, 30):
                new_values.insert(0, None)
        
        new_values = sorted([v for v in new_values if v is not None], key=float) + ([None] if None in new_values else [])
        
        # 确保新值与原始值有实质性差异，否则不调整
        if set(new_values) == set(original_values):
            return original_values
        
        logging.debug(f"{model_name}.{param_name}: 最优中心={optimal_center:.4f}, "
                     f"std={optimal_std:.4f}, 新范围={new_values}")
        
        return new_values
    
    def suggest_new_values_in_range(self, model_name, param_name, original_values, expand=False):
        """在最优值周围建议新的参数值
        
        :param model_name: 模型名称
        :param param_name: 超参数名称
        :param original_values: 原始参数范围
        :param expand: 是否扩展搜索空间
        :return: 建议的新参数范围
        """
        key = f"{model_name}_{param_name}"
        
        if key not in self.param_history or expand:
            if expand:
                # 扩展搜索空间
                min_val = original_values[0]
                max_val = original_values[-1]
                step = (max_val - min_val) / (len(original_values) - 1) if len(original_values) > 1 else 1
                expanded_range = np.arange(min_val - step, max_val + 2*step, step)
                return [v for v in expanded_range if v >= 0]  # 移除负值
            return original_values
        
        values = np.array(self.param_history[key]['values'])
        fitness = np.array(self.param_history[key]['fitness'])
        
        valid_mask = fitness != -np.inf
        if np.sum(valid_mask) == 0:
            return original_values
        
        best_value = values[valid_mask][np.argmax(fitness[valid_mask])]
        
        # 在最优值周围生成新值
        if isinstance(best_value, (int, np.integer)):
            step = max(1, int(abs(original_values[-1] - original_values[0]) / 10))
            new_range = [int(best_value - 2*step), int(best_value - step), int(best_value),
                        int(best_value + step), int(best_value + 2*step)]
            return [v for v in new_range if v > 0]
        else:
            step = abs(original_values[-1] - original_values[0]) / 10
            return [float(best_value - 2*step), float(best_value - step), float(best_value),
                   float(best_value + step), float(best_value + 2*step)]
    
    def should_adjust(self, current_gen, adjust_interval=5):
        """判断是否应该调整范围
        
        :param current_gen: 当前代数
        :param adjust_interval: 调整间隔（默认每5代调整一次）
        :return: 是否应该调整
        """
        return current_gen > 0 and current_gen % adjust_interval == 0
