"""
nsga-ii.py
实现：基于原论文流程的可运行实验单环（数据加载、SVR集成不确定性、EI、NSGA-II、KMeans选择、实验报告生成）
"""
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import norm
import matplotlib
# use non-interactive backend for headless environments / CI
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.population import Population
ELEMENTS = ['Al','Ti','V','Cr','Zr','Nb','Mo','Hf','Ta','W']
# directory to save reports and figures
REPORT_DIR = 'reports'
# 是否将模型预测的占位结果追加回训练集（仅在有真实测量值时应为 True）
APPEND_PREDICTIONS = False
# NSGA-II / selection settings
POP_SIZE = 400
N_GEN = 100
# 'kmeans' or 'maximin'
SELECT_STRATEGY = 'kmeans'
def load_dataset(local_path='datest.csv'):
	# 仅使用本地相对路径 datest.csv 加载数据，若不存在则抛出错误
	if local_path and os.path.exists(local_path):
		print(f"使用本地数据集: {local_path}")
		df = pd.read_csv(local_path)
	else:
		raise FileNotFoundError(f"未找到本地数据集 {local_path}，请将 datest.csv 放在脚本同目录下。")
	X = df[ELEMENTS].values.astype(float)
	y_strength = df['yield_strength_1000C'].values.astype(float)
	y_ductility = df['fracture_strain_RT'].values.astype(float)
	print(f"样本数: {len(X)}")
	return X, y_strength, y_ductility, df
def train_ensemble(X, y, M=20, svr_params=None):
	"""用 Bootstrap 训练 SVR 集成以估计不确定性"""
	if svr_params is None:
		svr_params = {'kernel':'rbf', 'C':100, 'gamma':'scale'}
	models = []
	n = X.shape[0]
	for i in range(M):
		idx = np.random.choice(n, n, replace=True)
		Xi = X[idx]
		yi = y[idx]
		m = SVR(**svr_params)
		m.fit(Xi, yi)
		models.append(m)
	return models
def predict_ensemble(models, Xq):
	preds = np.vstack([m.predict(Xq) for m in models])
	mu = preds.mean(axis=0)
	sigma = preds.std(axis=0, ddof=1) + 1e-9
	return mu, sigma

def expected_improvement(mu, sigma, f_best):
	z = (mu - f_best) / sigma
	ei = (mu - f_best) * norm.cdf(z) + sigma * norm.pdf(z)
	ei[sigma<=0] = 0.0
	return ei

class EIProblem(Problem):
	def __init__(self, models_s, models_d, scaler, best_s, best_d):
		# 变量在 [0,1] 区间，评估时会将每一行归一化使成分和为1
		super().__init__(n_var=10, n_obj=2, n_constr=1, xl=0.0, xu=1.0)
		self.models_s = models_s
		self.models_d = models_d
		self.scaler = scaler
		self.best_s = best_s
		self.best_d = best_d
	def _evaluate(self, X, out, *args, **kwargs):
		# 对每一行归一化（确保各元素摩尔分数之和为1）
		Xn = X / (X.sum(axis=1, keepdims=True) + 1e-12)
		# 通过不等式约束惩罚未满足 5–35 at% 的解（G<=0 为可行解）
		lower_violation = np.maximum(0.0, 0.05 - Xn)
		upper_violation = np.maximum(0.0, Xn - 0.35)
		violation = (lower_violation + upper_violation).sum(axis=1)
		# 将组成特征按训练时相同方式缩放（若已使用 StandardScaler）
		Xn_scaled = self.scaler.transform(Xn)
		# 使用集成模型预测均值与不确定性
		mu_s, sigma_s = predict_ensemble(self.models_s, Xn_scaled)
		mu_d, sigma_d = predict_ensemble(self.models_d, Xn_scaled)
		# 计算期望改进 EI
		ei_s = expected_improvement(mu_s, sigma_s, self.best_s)
		ei_d = expected_improvement(mu_d, sigma_d, self.best_d)
		# pymoo 为最小化框架，使用 -EI 来最大化期望改进
		out['F'] = np.column_stack([-ei_s, -ei_d])
		out['G'] = violation.reshape(-1,1)
def save_iteration_report(iteration, candidates, pred_s, pred_d):
	# 保存 CSV
	# 确保 reports 目录存在
	os.makedirs(REPORT_DIR, exist_ok=True)
	rows = []
	for i, comp in enumerate(candidates):
		comp_dict = {e: float(round(comp[j], 6)) for j,e in enumerate(ELEMENTS)}
		rows.append({**comp_dict, 'pred_strength_1000C': float(pred_s[i]), 'pred_ductility_RT': float(pred_d[i])})
	dfc = pd.DataFrame(rows)
	csv_path = os.path.join(REPORT_DIR, f'iter_{iteration}_candidates.csv')
	dfc.to_csv(csv_path, index=False)
	# Markdown summary
	md_lines = [f"# 第 {iteration} 轮实验报告", "", "## 推荐候选合金", ""]
	for i, comp in enumerate(candidates):
		md_lines.append(f"### 候选 {i+1}")
		comp_lines = [f"- {e}: {comp[j]:.4f}" for j,e in enumerate(ELEMENTS) if comp[j] > 1e-6]
		md_lines.extend(comp_lines)
		md_lines.append(f"- 预测 1000℃ 屈服强度: {pred_s[i]:.2f} MPa")
		md_lines.append(f"- 预测 室温断裂应变: {pred_d[i]:.2f} %")
		md_lines.append("")
	md_lines.append("## 帕累托前沿样本（预测值）")
	pf_path = os.path.join(REPORT_DIR, f'iter_{iteration}_pareto.png')
	md_lines.append(f"![pareto]({pf_path})")
	md_path = os.path.join(REPORT_DIR, f'iter_{iteration}_report.md')
	with open(md_path, 'w', encoding='utf-8') as f:
		f.write('\n'.join(md_lines))
	print(f"已保存：{csv_path}, {md_path}")

def plot_pareto_and_candidates(iteration, pareto_front, candidates_pred):
	# 绘制帕累托前沿及候选点
	plt.figure(figsize=(6,5))
	plt.scatter(pareto_front[:,0], pareto_front[:,1], c='lightgray', alpha=0.6, label='Pareto')
	plt.scatter(candidates_pred[:,0], candidates_pred[:,1], c='red', s=120, marker='*', label='Candidates')
	plt.xlabel('1000C strength (MPa)')
	plt.ylabel('RT ductility (%)')
	plt.title(f'Iter {iteration} Pareto and Candidates')
	plt.legend()
	plt.grid(alpha=0.3)
	path = os.path.join(REPORT_DIR, f'iter_{iteration}_pareto.png')
	plt.tight_layout()
	plt.savefig(path)
	plt.close()

def main():
	X, y_s, y_d, df_raw = load_dataset()
	# 对初始组成做标准化缩放
	scaler = StandardScaler().fit(X)
	# 为模型训练保留缩放后的特征矩阵
	X_scaled = scaler.transform(X)
	n_iterations = 30
	candidates_per_iter = 4
	history = []
	# 上一轮推荐的候选（决策变量空间，归一化后）用于注入到下一轮的初始种群
	prev_candidates = None
	for it in range(1, n_iterations+1):
		print(f"\n=== 迭代 {it} / {n_iterations} ===")
		# 训练集成模型（在缩放后的特征上训练以保持训练/预测一致）
		models_s = train_ensemble(X_scaled, y_s, M=20)
		models_d = train_ensemble(X_scaled, y_d, M=20)
		# 报告训练性能（用单模型的简单验证作为示意）
		# 用最后一个 ensemble 的模型做一次 train/test eval
		single_s = models_s[-1]
		single_d = models_d[-1]
		Xt, Xt_test, ys_t, ys_test = train_test_split(X_scaled, y_s, test_size=0.2, random_state=42)
		single_s.fit(Xt, ys_t)
		ys_pred = single_s.predict(Xt_test)
		print('强度模型示意 R2=', r2_score(ys_test, ys_pred))
		# 当前最优观测值
		best_s = np.nanmax(y_s)
		best_d = np.nanmax(y_d)
		print(f"当前观测最优：strength={best_s:.2f}, ductility={best_d:.2f}")
		# 运行 NSGA-II
		# 如果存在上一轮的候选，则通过自定义采样器将其注入初始种群
		problem = EIProblem(models_s, models_d, scaler, best_s, best_d)
		# 选择采样器：若有 prev_candidates 则使用 PreSampling 注入，否则使用随机采样
		if prev_candidates is not None and len(prev_candidates) > 0:
			class PreSampling(FloatRandomSampling):
				def __init__(self, prepop):
					self.prepop = np.atleast_2d(prepop)
					super().__init__()
				def _do(self, problem, n_samples, *args, random_state=None, **kwargs):
					# 若预置个体多于种群大小，截断；否则用随机样本补齐
					n_pre = self.prepop.shape[0]
					if n_pre >= n_samples:
						X = self.prepop[:n_samples]
					else:
						# 从父类采样补齐
						rnd_X = super()._do(problem, n_samples - n_pre, random_state=random_state)
						X = np.vstack([self.prepop, rnd_X])
					return X
			sampling = PreSampling(prev_candidates)
		else:
			sampling = FloatRandomSampling()
		# 使用显式代际循环：将父代 P_t 与子代 Q_t 合并为 2N 种群，
		# 对合并种群执行统一的非支配排序与拥挤度计算，选出最优 N 个个体作为 P_{t+1}
		algorithm = NSGA2(pop_size=POP_SIZE, sampling=sampling)
		# 初始化算法（设置 problem、随机种子等）
		algorithm.setup(problem, seed=it, verbose=False)

		# 逐代执行合并选择（显式实现 elitist replacement）
		for gen in range(int(N_GEN)):
			# 生成子代（infill）
			infills = algorithm.infill()
			if infills is None:
				break
			# 评估子代（计算 F, G 等）
			algorithm.evaluator.eval(problem, infills, algorithm=algorithm)

			# 将父代与子代合并为 2N（若父代为空则直接用子代）
			if algorithm.pop is None or len(algorithm.pop) == 0:
				merged = infills
			else:
				merged = Population.merge(algorithm.pop, infills)

			# 在合并种群上执行非支配排序与拥挤度选择，保留最优 POP_SIZE 个个体
			algorithm.pop = algorithm.survival.do(problem, merged, n_survive=POP_SIZE, algorithm=algorithm, random_state=algorithm.random_state)

			# 记录本代子代并执行后处理（更新最优解、终止条件、回调、历史等）
			algorithm.off = infills
			algorithm._post_advance()
		# 运行结束，读取结果
		res = algorithm.result()
		# 用 ensemble 均值预测帕累托点的实际性能
		X_pf = res.X
		X_pf_norm = X_pf / (X_pf.sum(axis=1, keepdims=True) + 1e-12)
		X_pf_scaled = scaler.transform(X_pf_norm)
		mu_s_pf, _ = predict_ensemble(models_s, X_pf_scaled)
		mu_d_pf, _ = predict_ensemble(models_d, X_pf_scaled)
		pareto_front = np.column_stack([mu_s_pf, mu_d_pf])
		# KMeans 选取候选
		# 去重帕累托样本（按决策变量去重），避免重复点导致 KMeans 聚成单簇
		# 先以归一化的决策空间去重（四舍五入减少浮点噪声影响）
		X_pf = X_pf_norm
		_unique_rows, unique_idx = np.unique(np.round(X_pf, decimals=8), axis=0, return_index=True)
		unique_idx = np.sort(unique_idx)
		X_pf_norm_unique = X_pf[unique_idx]
		pareto_front_unique = pareto_front[unique_idx]
		# 若帕累托点数量少于期望聚类数，则降 K 或跳过
		n_pf = pareto_front_unique.shape[0]
		if n_pf == 0:
			print("警告：本轮帕累托前沿为空，跳过候选选择。")
			history.append({'pareto': pareto_front, 'candidates': np.empty((0,10)), 'pred_s': np.array([]), 'pred_d': np.array([])})
			continue
		k = min(candidates_per_iter, int(n_pf))
		if k < candidates_per_iter:
			print(f"提示：帕累托唯一样本较少（{n_pf}），将聚类数降为 {k}。")
		# 如果只有 1 个唯一样本，直接选取该样本为候选
		if n_pf == 1:
			candidates = np.array([X_pf_norm_unique[0]])
			cand_pred_s = np.array([pareto_front_unique[0,0]])
			cand_pred_d = np.array([pareto_front_unique[0,1]])
		else:
			# 选择策略：kmeans 或 maximin，maximin 优先以保证帕累托多样性
			if SELECT_STRATEGY == 'kmeans':
				kmeans = KMeans(n_clusters=k, random_state=42).fit(pareto_front_unique)
				centers = kmeans.cluster_centers_
				candidates = []
				cand_pred_s = []
				cand_pred_d = []
				for c in centers:
					dists = np.linalg.norm(pareto_front_unique - c, axis=1)
					idx = np.argmin(dists)
					comp = X_pf_norm_unique[idx]
					candidates.append(comp)
					cand_pred_s.append(pareto_front_unique[idx,0])
					cand_pred_d.append(pareto_front_unique[idx,1])
			else:
				# maximin: 先选与质心最远的点，然后贪心选择使最小距离最大化
				candidates = []
				cand_pred_s = []
				cand_pred_d = []
				centroid = pareto_front_unique.mean(axis=0)
				first_idx = int(np.argmax(np.linalg.norm(pareto_front_unique - centroid, axis=1)))
				selected = [first_idx]
				while len(selected) < k:
					# 对每个未选点计算到已选集合的最小距离
					not_sel = [i for i in range(pareto_front_unique.shape[0]) if i not in selected]
					min_dists = []
					for i in not_sel:
						d = np.min([np.linalg.norm(pareto_front_unique[i] - pareto_front_unique[j]) for j in selected])
						min_dists.append(d)
					next_idx = not_sel[int(np.argmax(min_dists))]
					selected.append(next_idx)
				for idx in selected:
					candidates.append(X_pf_norm_unique[idx])
					cand_pred_s.append(pareto_front_unique[idx,0])
					cand_pred_d.append(pareto_front_unique[idx,1])
		candidates = np.array(candidates)
		cand_pred_s = np.array(cand_pred_s)
		cand_pred_d = np.array(cand_pred_d)
		# 生成并保存报告（在真实实验中，这里替换为合成并测试）
		save_iteration_report(it, candidates, cand_pred_s, cand_pred_d)
		plot_pareto_and_candidates(it, pareto_front, np.column_stack([cand_pred_s, cand_pred_d]))
		# 将“实验”结果加入训练集（仅在真实测量或明确允许使用预测值时开启）
		if APPEND_PREDICTIONS:
			# 将原始（未缩放）候选加入原始训练集，并同步更新缩放后的矩阵
			X = np.vstack([X, candidates])
			X_scaled = np.vstack([X_scaled, scaler.transform(candidates)])
			y_s = np.hstack([y_s, cand_pred_s])
			y_d = np.hstack([y_d, cand_pred_d])
		history.append({'pareto': pareto_front, 'candidates': candidates, 'pred_s': cand_pred_s, 'pred_d': cand_pred_d})
		print(f"第 {it} 轮：已推荐 {len(candidates)} 个候选并保存报告。")
		# 将本轮候选保留为下一轮的预置个体（决策变量空间）
		if candidates is not None and len(candidates) > 0:
			# candidates 已是归一化的决策变量向量
			prev_candidates = candidates.copy()
	# 输出最终候选
	final = history[-1]
	print("\n=== 最终推荐候选 ===")
	for i, comp in enumerate(final['candidates']):
		comp_dict = {e: float(round(comp[j],4)) for j,e in enumerate(ELEMENTS) if comp[j]>0.0001}
		print(f"候选{i+1}: 成分={comp_dict}, 预测 strength={final['pred_s'][i]:.4f}, ductility={final['pred_d'][i]:.2f}")

if __name__ == '__main__':
	main()

